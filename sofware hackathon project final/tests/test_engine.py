from datetime import datetime, timedelta

from backend.data_generator import demo_data
from backend.data_loader import load_csv_data
from backend.engine import allocate_plan, analyze_shipments, candidates_for_shipment


def test_demo_allocates_multiple_shipments_and_escalates_some():
    shipments, vehicles, hubs, routes = demo_data()
    plan = allocate_plan(shipments, vehicles, hubs, routes)
    assert len(plan["allocations"]) >= 2
    assert len(plan["escalated"]) >= 1


def test_capacity_is_not_overallocated():
    shipments, vehicles, hubs, routes = demo_data()
    plan = allocate_plan(shipments, vehicles, hubs, routes)
    for vehicle_id, group in plan["allocations"].groupby("vehicle_id"):
        if " + " not in vehicle_id:
            vehicle = vehicles.loc[vehicles.vehicle_id == vehicle_id].iloc[0]
            assert vehicle.current_load_kg + group.weight_kg.sum() <= vehicle.capacity_kg


def test_priority_score_is_bounded_and_sorted():
    shipments, *_ = demo_data()
    result = analyze_shipments(shipments)
    assert result.priority_score.between(0, 1).all()
    assert result.iloc[0].priority_score >= result.iloc[-1].priority_score


def test_route_matching_finds_direct_and_transfer_candidates():
    shipments, vehicles, hubs, routes = demo_data()
    direct = candidates_for_shipment(shipments.iloc[0], vehicles, hubs, routes)
    transfer = candidates_for_shipment(shipments.iloc[4], vehicles, hubs, routes)
    assert any(item["strategy"] == "DIRECT PIGGYBACK" for item in direct)
    assert any(item["strategy"] == "ONE-HUB PIGGYBACK" for item in transfer)


def test_expired_deadline_is_not_selected():
    shipments, vehicles, hubs, routes = demo_data()
    shipments.loc[0, "deadline"] = datetime.now() - timedelta(hours=1)
    plan = allocate_plan(shipments, vehicles, hubs, routes)
    assert "S101" in set(plan["escalated"].shipment_id)


def test_demo_csv_bundle_loads_with_runtime_dates():
    shipments, vehicles, hubs, routes = load_csv_data(demo=True)
    assert len(shipments) == 6
    assert len(vehicles) == 6
    assert shipments.deadline.notna().all()
    assert vehicles.eta.notna().all()
    assert set(routes.route_status) == {"Active"}


def test_full_csv_mode_uses_same_contract_and_optimizer():
    shipments, vehicles, hubs, routes = load_csv_data(demo=False)
    plan = allocate_plan(shipments, vehicles, hubs, routes)
    assert len(shipments) == 360
    assert len(vehicles) == 72
    assert shipments.deadline.notna().all()
    assert vehicles.available_capacity_kg.ge(0).all()
    assert len(plan["allocations"]) + len(plan["escalated"]) == len(shipments)
    assert plan["validation"] == []


def test_s104_prefers_lower_cost_v101_when_deadlines_are_feasible():
    shipments, vehicles, hubs, routes = load_csv_data(demo=True)
    plan = allocate_plan(shipments, vehicles, hubs, routes)
    selected = plan["allocations"].loc[plan["allocations"].shipment_id == "S104"].iloc[0]
    assert selected.vehicle_id == "V101"
