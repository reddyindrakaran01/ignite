# Intelligent Shipment Recovery Control Tower

A hackathon prototype for **SH-205: Intelligent Shipment Piggybacking**. It treats recovery as a fleet-wide allocation problem: multiple misplaced shipments compete for limited capacity across direct routes and one-hub transfers.

> This is a hackathon prototype using synthetic logistics data. Costs, ETAs, and risk scores are estimates, not operational commitments.

## Run

```powershell
cd "sofware hackathon project final"
python -m pip install -r requirements.txt
streamlit run frontend/app.py
```

The app starts in **HACKATHON DEMO MODE** with a small scenario designed to show shared capacity, one-hop transfer, deadline pressure, and escalation. Turn demo mode off in the sidebar to generate 360 reproducible synthetic shipments and 72 vehicles.

## Architecture

- `data/shipments.csv`, `data/vehicles.csv`, `data/hubs.csv`, `data/routes.csv`: editable demo CSV files used by the application.
- `backend/data_loader.py`: CSV loading, runtime date handling, and larger generated CSV bundle creation.
- `backend/data_generator.py`: reproducible demo and synthetic datasets.
- `backend/engine.py`: validation, risk/priority scoring, direct and one-hub candidate generation, and global greedy allocation.
- `backend/ai.py`: deterministic explanation layer with optional OpenAI-compatible integration point.
- `frontend/app.py`: Streamlit control tower, planner, global plan, network, simulator, and analyzer.
- `ui/visualization.py`: reusable semantic Plotly charts for the RELAYX control tower.
- `tests/test_engine.py`: focused capacity, deadline, routing, allocation, and failure-path tests.

## Optimization approach

1. Validate and normalize records.
2. Score urgency, deadline pressure, priority, delay probability, value, and route risk.
3. Generate feasible direct and one-hub candidates.
4. Rank candidates using deadline safety, priority satisfaction, cost, and utilization.
5. Allocate all shipments in priority order while reserving actual vehicle capacity.
6. Mark anything left without a feasible candidate as escalated with a transparent reason.

The application never lets an explanation layer override hard capacity, route, or deadline constraints.

The frontend navigation includes Control Tower, Shipment Priority, Recovery Planner, Global Recovery Plan, Decision Audit, Network Intelligence, What-If Simulator, and AI Logistics Analyzer. The Decision Audit is sourced directly from deterministic allocation and rejection records.

## AI analyzer

The AI Logistics Analyzer accepts free-form questions through a chat input. It sends the complete loaded shipments, vehicles, hubs, routes, calculated allocations, escalations, rejected candidates, and summary metrics to an OpenAI-compatible chat-completions endpoint when `OPENAI_API_KEY` is configured. Set `OPENAI_BASE_URL` and `OPENAI_MODEL` for another compatible provider. Without a key, the app uses a deterministic fallback and remains fully functional.

## Data files

The default demo reads the four CSV files in `data/`. Demo shipment deadlines and vehicle times use offset columns so they remain relative to the time the app is opened. Disable demo mode to generate and use a 360-shipment, 72-vehicle bundle in `data/generated/`.
