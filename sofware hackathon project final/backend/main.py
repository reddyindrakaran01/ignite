from __future__ import annotations

import io
import inspect
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from starlette.routing import Router

# Some preinstalled environments pair FastAPI 0.115 with Starlette 1.x,
# whose Router removed FastAPI's legacy startup kwargs.
if "on_startup" not in inspect.signature(Router.__init__).parameters:
    _router_init = Router.__init__

    def _compatible_router_init(self: Router, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("on_startup", None)
        kwargs.pop("on_shutdown", None)
        _router_init(self, *args, **kwargs)

    Router.__init__ = _compatible_router_init  # type: ignore[method-assign]

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from backend.ai import analyze_question
from backend.auth import authenticate_user, create_access_token, current_user, require_admin, require_manager
from backend.data_loader import load_csv_data
from backend.engine import allocate_plan, analyze_shipments, run_simulation, validate_data

app = FastAPI(title="Ignite Shipment Recovery API", version="1.0.0")
# Starlette 1.x expects this application attribute during middleware setup.
if not hasattr(app, "max_body_size"):
    app.max_body_size = None
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501", "http://localhost:8502", "http://localhost:8503"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DATA_NAMES = ("shipments", "vehicles", "hubs", "routes")
DATASET = {name: frame for name, frame in zip(DATA_NAMES, load_csv_data(demo=False))}


class DatasetPayload(BaseModel):
    shipments: list[dict[str, Any]] = Field(default_factory=list)
    vehicles: list[dict[str, Any]] = Field(default_factory=list)
    hubs: list[dict[str, Any]] = Field(default_factory=list)
    routes: list[dict[str, Any]] = Field(default_factory=list)


class PlanRequest(DatasetPayload):
    weights: dict[str, float] | None = None


class SimulationRequest(PlanRequest):
    capacity_factor: float = 1.0
    unavailable_vehicle_ids: list[str] = Field(default_factory=list)
    deadline_shift_hours: float = 0
    route_cost_factor: float = 1.0


class AIRequest(BaseModel):
    question: str
    plan: dict[str, Any] | None = None


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    safe = frame.copy()
    for column in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[column]):
            safe[column] = safe[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return safe.astype(object).where(pd.notna(safe), None).to_dict(orient="records")


def frame_bundle(payload: DatasetPayload) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tuple(pd.DataFrame(getattr(payload, name)) for name in DATA_NAMES)  # type: ignore[return-value]


def serialize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in plan.items() if not isinstance(value, pd.DataFrame)}
    for key in ("shipments", "allocations", "candidate_options", "escalated", "rejected", "vehicles"):
        if isinstance(plan.get(key), pd.DataFrame):
            result[key] = records(plan[key])
    return result


def plan_from_payload(payload: DatasetPayload, weights: dict[str, float] | None = None) -> dict[str, Any]:
    frames = frame_bundle(payload)
    issues = validate_data(*frames)
    if issues:
        raise HTTPException(status_code=422, detail=issues)
    return allocate_plan(*frames, weights)


def update_dataset(name: str, contents: bytes) -> dict[str, Any]:
    frame = pd.read_csv(io.BytesIO(contents))
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    if name == "shipments" and "deadline" not in frame.columns and "deadline_offset_hours" in frame.columns:
        frame["deadline"] = [now + timedelta(hours=float(value)) for value in frame.pop("deadline_offset_hours")]
        if "created_at" not in frame.columns and "created_offset_hours" in frame.columns:
            frame["created_at"] = [now - timedelta(hours=float(value)) for value in frame.pop("created_offset_hours")]
    if name == "vehicles" and "eta" not in frame.columns and "eta_offset_hours" in frame.columns:
        frame["eta"] = [now + timedelta(hours=float(value)) for value in frame.pop("eta_offset_hours")]
        if "departure_time" not in frame.columns and "departure_offset_hours" in frame.columns:
            frame["departure_time"] = [now + timedelta(hours=float(value)) for value in frame.pop("departure_offset_hours")]
    candidate_dataset = {**DATASET, name: frame}
    try:
        issues = validate_data(*(candidate_dataset[key] for key in DATA_NAMES))
    except (KeyError, TypeError) as error:
        raise HTTPException(status_code=422, detail=f"Invalid {name} schema: {error}") from error
    if issues:
        raise HTTPException(status_code=422, detail=issues)
    DATASET[name] = frame
    return {"dataset": name, "records": len(frame), "validation": issues}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends()) -> dict[str, str]:
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid demo credentials")
    return {"access_token": create_access_token(user["username"], user["role"]), "token_type": "bearer", "role": user["role"], "username": user["username"]}


@app.get("/auth/me")
def me(user: dict[str, str] = Depends(current_user)) -> dict[str, str]:
    return user


@app.get("/data")
def data(user: dict[str, str] = Depends(require_manager)) -> dict[str, Any]:
    return {name: records(frame) for name, frame in DATASET.items()}


@app.post("/data/{name}/upload")
async def upload(name: str, file: UploadFile = File(...), user: dict[str, str] = Depends(require_admin)) -> dict[str, Any]:
    if name not in DATA_NAMES:
        raise HTTPException(status_code=404, detail="Unknown dataset")
    return update_dataset(name, await file.read())


@app.post("/data/submit")
def submit_data(payload: DatasetPayload, user: dict[str, str] = Depends(require_admin)) -> dict[str, Any]:
    frames = frame_bundle(payload)
    issues = validate_data(*frames)
    if issues:
        raise HTTPException(status_code=422, detail=issues)
    DATASET.update(dict(zip(DATA_NAMES, frames)))
    return {"records": {name: len(frame) for name, frame in zip(DATA_NAMES, frames)}, "validation": []}


@app.post("/shipments/analyze")
def analyze(payload: PlanRequest, user: dict[str, str] = Depends(require_manager)) -> dict[str, Any]:
    shipments, _, _, _ = frame_bundle(payload)
    return {"shipments": records(analyze_shipments(shipments, payload.weights))}


@app.post("/recovery/plan")
def recovery_plan(payload: PlanRequest, user: dict[str, str] = Depends(require_manager)) -> dict[str, Any]:
    return serialize_plan(plan_from_payload(payload, payload.weights))


@app.post("/simulation/run")
def simulation(payload: SimulationRequest, user: dict[str, str] = Depends(require_manager)) -> dict[str, Any]:
    frames = frame_bundle(payload)
    plan = run_simulation(*frames, payload.weights, payload.capacity_factor, payload.unavailable_vehicle_ids, payload.deadline_shift_hours, payload.route_cost_factor)
    return serialize_plan(plan)


@app.post("/ai/analyze")
def ai_analyze(payload: AIRequest, user: dict[str, str] = Depends(require_manager)) -> dict[str, str]:
    if not payload.plan:
        raise HTTPException(status_code=422, detail="A calculated plan is required")
    frames = tuple(pd.DataFrame(payload.plan.get(name, [])) for name in DATA_NAMES)
    plan = {key: pd.DataFrame(value) if isinstance(value, list) else value for key, value in payload.plan.items()}
    answer, source = analyze_question(plan, payload.question, frames[1], frames[2], frames[3])
    return {"answer": answer, "source": source}
