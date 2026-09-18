from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agent import answer as answer_question
from backend.data_loader import load_csv_data
from backend.engine import DEFAULT_WEIGHTS, allocate_plan
from ui.visualization import fleet_chart, priority_chart, risk_chart, status_chart


st.set_page_config(page_title="RELAYX | AI Shipment Recovery Control Tower", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
h1, h2, h3, h4, h5, h6, p, label, span, li { color: #0F172A !important; }
h1, h2, h3, h4, h5, h6 { letter-spacing: 0; }
.stApp { background: #F5F7FA; }
.block-container { padding-top: .7rem; max-width: 1600px; }
.stSidebar, [data-testid="stSidebar"] { background: #0B1F33; }
[data-testid="stSidebar"] * { color: #F8FAFC !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #A8C0D4 !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea { color: #0F172A !important; background: #FFFFFF !important; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #0F172A !important; }
.stButton > button { color: #FFFFFF; background: #00A6A6; border: 1px solid #087F8C; font-weight: 700; border-radius: 8px; min-height: 2.5rem; }
.stButton > button:hover { color: #FFFFFF; background: #087F8C; border-color: #00A6A6; }
[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: .18rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { background: #102A43; border: 1px solid #1D405B; border-radius: 9px; padding: .62rem .7rem; color: #CBD5E1 !important; font-weight: 600; margin: .22rem 0; transition: background .15s ease, border-color .15s ease; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: #173A56; color: #FFFFFF !important; border-color: #22D3EE; transform: translateX(2px); }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background: #00A6A6; color: #FFFFFF !important; border-color: #22D3EE; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p { color: #FFFFFF !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] input { accent-color: #22D3EE; }
[data-testid="stSidebar"] [data-testid="stRadio"] label p { font-size: .83rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label small { color: #8FB4C8 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(1) { border-left: 3px solid #22D3EE; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(2) { border-left: 3px solid #2563EB; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(3) { border-left: 3px solid #00A6A6; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(4) { border-left: 3px solid #16A34A; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(5) { border-left: 3px solid #F59E0B; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(6) { border-left: 3px solid #64748B; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(7) { border-left: 3px solid #F59E0B; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(8) { border-left: 3px solid #22D3EE; }
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #D9E2EC; border-radius: 12px; padding: .8rem; }
[data-testid="stMetricLabel"] p, [data-testid="stMetricValue"] { color: #0F172A !important; }
[data-testid="stDataFrame"] { border: 1px solid #CBD5E1; border-radius: 8px; }
[data-testid="stDataFrame"] * { color: #0F172A; }
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] { background: #FFFFFF; }
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea { color: #0F172A !important; }
.stCaption, [data-testid="stCaptionContainer"] p { color: #64748B !important; }
.stAlert p, .stAlert span { color: #0F172A !important; }
.hero { background: #0B1F33; color: #F8FAFC; padding: 1.65rem 2rem; border-radius: 12px; margin-bottom: 1rem; border-left: 5px solid #00A6A6; }
.hero h1, .hero h1 span, .hero h1 strong { color: #F8FAFC !important; font-size: 2.1rem; margin: 0; }
.hero p, .hero p span { color: #C7D8E5 !important; margin: .35rem 0 0; font-size: 1rem; }
.kpi { background: #FFFFFF; border: 1px solid #D9E2EC; border-left: 4px solid #00A6A6; border-radius: 12px; padding: .85rem; min-height: 92px; box-shadow: 0 2px 8px rgba(15, 23, 42, .04); }
.kpi-label { color: #64748B; text-transform: uppercase; font-size: .7rem; font-weight: 700; letter-spacing: .08em; }
.kpi-value { color: #0F172A; font-size: 1.65rem; font-weight: 800; margin-top: .25rem; }
.section { color: #0F172A; border-bottom: 2px solid #00A6A6; padding-bottom: .35rem; }
.audit-card { background: #FFFFFF; border: 1px solid #D9E2EC; border-radius: 12px; padding: 1rem; margin-bottom: .75rem; }
.brand-mark { color: #F8FAFC; font-size: 1.25rem; font-weight: 800; letter-spacing: .08em; }
.brand-sub { color: #A8C0D4; font-size: .68rem; margin-left: .55rem; }
.status-online { color: #86EFAC; font-size: .78rem; font-weight: 700; padding-top: .45rem; }
.top-shell { background: #0B1F33; border-radius: 0 0 12px 12px; padding: .5rem .9rem .35rem; margin: -.7rem -1rem 1.1rem; }
.top-shell [data-testid="stRadio"] label { color: #CBD5E1 !important; }
.top-shell [data-testid="stRadio"] label:has(input:checked) { color: #FFFFFF !important; }
.top-shell [data-testid="stRadio"] label:has(input:checked) p { color: #FFFFFF !important; }
.top-search input { background: #FFFFFF !important; color: #0F172A !important; }
.workspace-card { background: #FFFFFF; border: 1px solid #D9E2EC; border-radius: 12px; padding: 1rem; min-height: 100%; box-shadow: 0 2px 8px rgba(15, 23, 42, .04); }
.workspace-title { color: #0F172A; font-size: .78rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; margin-bottom: .6rem; }
.queue-row { display: grid; grid-template-columns: 5.2rem 1fr 4rem; align-items: center; gap: .45rem; padding: .62rem 0; border-bottom: 1px solid #E2E8F0; font-size: .78rem; }
.queue-row:last-child { border-bottom: 0; }
.queue-priority { font-weight: 800; }
.queue-critical { color: #EF4444; }
.queue-high { color: #F59E0B; }
.queue-medium { color: #2563EB; }
.queue-normal { color: #64748B; }
.queue-action { color: #087F8C; font-weight: 700; text-align: right; }
.sidebar-section { color: #8FB4C8; font-size: .68rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; margin: .9rem 0 .35rem; }
.sidebar-status { background: #102A43; border: 1px solid #1D405B; border-radius: 9px; padding: .75rem; color: #DCEAF2 !important; font-size: .76rem; line-height: 1.7; }
.sidebar-status strong { color: #86EFAC !important; }
.page-kicker { color: #087F8C !important; font-size: .7rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .2rem; }
.module-header { display: flex; align-items: center; justify-content: space-between; background: #FFFFFF; border: 1px solid #D9E2EC; border-left: 4px solid #00A6A6; border-radius: 10px; padding: .75rem 1rem; margin-bottom: 1rem; }
.module-header-title { color: #0F172A !important; font-weight: 800; font-size: 1.1rem; }
.module-header-meta { color: #64748B !important; font-size: .78rem; }
.stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: #0F172A !important; }
[data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span { color: #0F172A !important; }
[data-testid="stChatInput"] textarea { color: #0F172A !important; background: #FFFFFF !important; }
.planner-state { background: #E8F1F5; border: 1px solid #8FBCC7; border-left: 4px solid #087F8C; border-radius: 10px; color: #0B1F33 !important; padding: .9rem 1rem; margin: .7rem 0; font-weight: 600; }
.planner-state strong, .planner-state span { color: #0B1F33 !important; }
.planner-note { background: #EEF6F8; border: 1px solid #B7DCE2; border-left: 4px solid #22D3EE; border-radius: 10px; color: #102A43 !important; padding: .9rem 1rem; margin: .7rem 0; line-height: 1.45; }
.planner-note strong, .planner-note span { color: #102A43 !important; }
</style>
""", unsafe_allow_html=True)


def money(value: float) -> str:
    return f"₹{value:,.0f}"


def kpi(label: str, value: str):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)


def network_figure(routes: pd.DataFrame, vehicles: pd.DataFrame):
    """Build the operational network used by the Control Tower and Network page."""
    edge_counts = routes.merge(vehicles.groupby(["route_origin", "route_destination"], as_index=False).size().rename(columns={"size": "vehicles"}), left_on=["origin", "destination"], right_on=["route_origin", "route_destination"], how="left").fillna({"vehicles": 0})
    positions = {city: (i % 4, -(i // 4)) for i, city in enumerate(sorted(set(routes.origin) | set(routes.destination)))}
    figure = go.Figure()
    for row in edge_counts.itertuples():
        x0, y0 = positions[row.origin]
        x1, y1 = positions[row.destination]
        figure.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line={"width": 1 + row.vehicles, "color": "#5B8EA3"}, hovertext=f"{row.origin} → {row.destination} · {int(row.vehicles)} vehicles", showlegend=False))
        figure.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=2, arrowcolor="#00A6A6")
    figure.add_trace(go.Scatter(x=[positions[c][0] for c in positions], y=[positions[c][1] for c in positions], mode="markers", hovertext=list(positions), hovertemplate="%{hovertext}<extra></extra>", marker={"size": 22, "color": "#00A6A6", "line": {"width": 3, "color": "#FFFFFF"}}, showlegend=False))
    for city, (x_position, y_position) in positions.items():
        figure.add_annotation(x=x_position, y=y_position, text=f"<b>{city}</b>", showarrow=False, yshift=24, font={"family": "Inter, sans-serif", "size": 12, "color": "#0F172A"}, bgcolor="#FFFFFF", bordercolor="#00A6A6", borderwidth=1, borderpad=4)
    figure.update_layout(height=390, xaxis={"visible": False, "range": [-.7, 3.7]}, yaxis={"visible": False, "range": [-3.2, .8], "scaleanchor": "x", "scaleratio": 1}, plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF", margin={"l": 8, "r": 8, "t": 8, "b": 8}, hoverlabel={"bgcolor": "#0B1F33", "font": {"color": "#FFFFFF"}})
    return figure


@st.cache_data(show_spinner=False)
def load_data(demo: bool):
    return load_csv_data(demo=demo)


@st.cache_data(show_spinner=False)
def build_plan(demo: bool, weight_values: tuple[tuple[str, float], ...]):
    loaded_data = load_data(demo)
    configured_weights = dict(weight_values)
    return loaded_data, allocate_plan(*loaded_data, configured_weights)


with st.sidebar:
    st.markdown('<div class="brand-mark">◈ RELAYX</div><div class="brand-sub">AI SHIPMENT RECOVERY</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-online">● SYSTEM ONLINE</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="sidebar-section">Operations workspace</div>', unsafe_allow_html=True)
    navigation_labels = ["▦  Control Tower", "◉  Shipment Priority", "⇄  Recovery Planner", "▤  Global Recovery Plan", "✓  Decision Audit", "⌁  Network Intelligence", "◌  What-If Simulator", "✦  AI Logistics Analyzer"]
    navigation_target = {
        "▦  Control Tower": "Control Tower", "◉  Shipment Priority": "Shipment Priority", "⇄  Recovery Planner": "Recovery Planner", "▤  Global Recovery Plan": "Global Recovery Plan",
        "✓  Decision Audit": "Decision Audit", "⌁  Network Intelligence": "Network Intelligence", "◌  What-If Simulator": "What-If Simulator", "✦  AI Logistics Analyzer": "AI Logistics Analyzer",
    }
    page = navigation_target[st.radio("Navigate", navigation_labels, label_visibility="collapsed")]
    st.divider()
    st.markdown('<div class="sidebar-section">Workspace controls</div>', unsafe_allow_html=True)
    st.caption("Live synthetic logistics data · deterministic optimization source of truth")
    with st.expander("Weights"):
        weights = {key: st.slider(key.title(), 0.0, 1.0, float(value), 0.05) for key, value in DEFAULT_WEIGHTS.items()}
    st.markdown('<div class="sidebar-section">System status</div><div class="sidebar-status"><strong>● Optimization engine online</strong><br>● Live simulation data<br>● Analyzer ready</div>', unsafe_allow_html=True)

try:
    (shipments, vehicles, hubs, routes), plan = build_plan(False, tuple(sorted(weights.items())))
except Exception as error:
    st.error(f"The recovery workspace could not be loaded: {error}")
    st.info("Check the CSV data files and restart the app from the project folder.")
    st.stop()

st.markdown('<div class="hero"><h1>RELAYX · AI SHIPMENT RECOVERY CONTROL TOWER</h1><p>Recover more. Use existing capacity. Move smarter.</p></div>', unsafe_allow_html=True)
if plan["validation"]:
    st.warning(" · ".join(plan["validation"]))

shipments_view = plan["shipments"]
allocations = plan["allocations"]
recovered_count = len(allocations)
critical = int((shipments_view.priority == "Critical").sum())
at_risk = int((shipments_view.deadline_risk_label.isin(["HIGH", "CRITICAL"])).sum())
available_vehicles = int((vehicles.vehicle_status != "Unavailable").sum())
utilization = float(((vehicles.current_load_kg.sum() + allocations.weight_kg.sum()) / vehicles.capacity_kg.sum()) * 100) if not allocations.empty else float((vehicles.current_load_kg.sum() / vehicles.capacity_kg.sum()) * 100)

module_descriptions = {
    "Shipment Priority": ("SHIPMENT PRIORITY", "Rank misplaced shipments by urgency, risk, value and deadline.", "#2563EB"),
    "Recovery Planner": ("RECOVERY PLANNER", "Compare feasible direct and one-hub recovery options.", "#00A6A6"),
    "Global Recovery Plan": ("GLOBAL RECOVERY PLAN", "See the fleet-wide allocation across shared capacity.", "#16A34A"),
    "Decision Audit": ("DECISION AUDIT", "Inspect what was selected and why alternatives were rejected.", "#F59E0B"),
    "Network Intelligence": ("NETWORK INTELLIGENCE", "Explore active corridors, hubs, vehicles and route risk.", "#22D3EE"),
    "What-If Simulator": ("WHAT-IF SIMULATOR", "Stress-test the recovery network against disruption.", "#F59E0B"),
    "AI Logistics Analyzer": ("AI LOGISTICS ANALYZER", "Ask the agent about the calculated recovery network.", "#22D3EE"),
}
if page in module_descriptions:
    module_title, module_description, module_color = module_descriptions[page]
    st.markdown(f'<div class="module-header" style="border-left-color:{module_color}"><span class="module-header-title">{module_title}</span><span class="module-header-meta">{module_description}</span></div>', unsafe_allow_html=True)

if page == "Control Tower":
    st.markdown("### Global visibility into misplaced shipments, recovery opportunities, fleet capacity and delivery risk.")
    cols = st.columns(8)
    for col, label, value in zip(cols, ["Misplaced", "Recovered", "Escalated", "At risk", "Critical", "Vehicles online", "Utilization", "Savings"], [len(shipments_view), recovered_count, len(plan["escalated"]), at_risk, critical, available_vehicles, f"{utilization:.1f}%", money(plan["savings"])]):
        with col: kpi(label, str(value))
    st.write("")
    network_column, queue_column = st.columns([7, 5], gap="medium")
    with network_column:
        st.markdown('<div class="workspace-card"><div class="workspace-title">Recovery network · live route intelligence</div>', unsafe_allow_html=True)
        st.plotly_chart(network_figure(routes, vehicles), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with queue_column:
        st.markdown('<div class="workspace-card"><div class="workspace-title">Priority queue · action required</div>', unsafe_allow_html=True)
        for row in shipments_view.head(7).itertuples():
            priority_class = str(row.priority).lower()
            status = "ESCALATED" if row.shipment_id in set(plan["escalated"].shipment_id) else "RECOVERY READY"
            action = "Recover now" if row.recommended_action == "Recover immediately" else "Review route"
            st.markdown(f'<div class="queue-row"><span class="queue-priority queue-{priority_class}">{row.priority.upper()}</span><span><b>{row.shipment_id}</b> · {row.current_location} → {row.destination}<br><small>{row.remaining_hours:.1f}h remaining · {status}</small></span><span class="queue-action">{action} →</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(status_chart(plan), use_container_width=True)
    with right:
        st.plotly_chart(priority_chart(shipments_view), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(risk_chart(shipments_view), use_container_width=True)
    with right:
        st.plotly_chart(fleet_chart(vehicles), use_container_width=True)
    impact_left, impact_right = st.columns(2)
    with impact_left:
        st.markdown("#### BUSINESS IMPACT")
        st.dataframe(pd.DataFrame({"Metric": ["Capacity reused", "Dedicated trips avoided", "On-time recovery", "Estimated recovery cost"], "Value": [f"{allocations.weight_kg.sum() if not allocations.empty else 0:,.0f} kg", str(recovered_count), f"{(allocations.deadline_margin.ge(0).mean() * 100 if not allocations.empty else 0):.1f}%", money(plan["total_cost"])]}), hide_index=True, use_container_width=True)
    with impact_right:
        st.markdown("#### OPTIMIZATION SUMMARY")
        st.dataframe(pd.DataFrame({"Metric": ["Shipments considered", "Feasible / recovered", "Escalated", "Vehicles used", "Transfers"], "Value": [len(shipments_view), f"{recovered_count} / {len(shipments_view)}", len(plan["escalated"]), allocations.vehicle_id.nunique() if not allocations.empty else 0, int(allocations.strategy.eq("ONE-HUB PIGGYBACK").sum()) if not allocations.empty else 0]}), hide_index=True, use_container_width=True)

elif page == "Shipment Priority":
    st.markdown("### Priority queue")
    filters = st.columns(4)
    with filters[0]: priority_filter = st.multiselect("Priority", sorted(shipments_view.priority.unique()))
    with filters[1]: risk_filter = st.multiselect("Risk", sorted(shipments_view.deadline_risk_label.unique()))
    with filters[2]: destination_filter = st.multiselect("Destination", sorted(shipments_view.destination.unique()))
    with filters[3]: min_weight = st.number_input("Minimum weight kg", 0, int(shipments_view.weight_kg.max()), 0)
    filtered = shipments_view.copy()
    if priority_filter: filtered = filtered[filtered.priority.isin(priority_filter)]
    if risk_filter: filtered = filtered[filtered.deadline_risk_label.isin(risk_filter)]
    if destination_filter: filtered = filtered[filtered.destination.isin(destination_filter)]
    filtered = filtered[filtered.weight_kg >= min_weight]
    display = filtered.reset_index(drop=True)
    display.insert(0, "Rank", range(1, len(display) + 1))
    st.dataframe(display[["Rank", "shipment_id", "current_location", "destination", "weight_kg", "priority", "deadline", "remaining_hours", "delay_probability", "deadline_risk_label", "priority_score", "recommended_action"]], use_container_width=True, hide_index=True, column_config={"priority_score": st.column_config.ProgressColumn("Priority score", min_value=0, max_value=1), "delay_probability": st.column_config.ProgressColumn("Delay probability", min_value=0, max_value=1)})
    st.caption("Sorted by a configurable weighted score combining urgency, deadline pressure, business priority, delay probability, value, and route risk.")

elif page == "Recovery Planner":
    selected_id = st.selectbox("Analyze shipment", shipments_view.shipment_id.tolist())
    shipment = shipments_view[shipments_view.shipment_id == selected_id].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Shipment", selected_id); b.metric("Priority", shipment.priority); c.metric("Weight", f"{shipment.weight_kg:.0f} kg"); d.metric("Deadline risk", shipment.deadline_risk_label)
    st.write(f"Current route: **{shipment.current_location} → {shipment.destination}** · Remaining time: **{shipment.remaining_hours:.1f} h** · Delay probability: **{shipment.delay_probability:.0%}**")
    options = []
    rejected_options = []
    from backend.engine import candidates_for_shipment
    for candidate in candidates_for_shipment(shipment, vehicles, hubs, routes):
        candidate["deadline_margin"] = float(shipment.remaining_hours - candidate["eta_hours"])
        candidate["cost"] = (candidate["transport_cost"] * shipment.weight_kg / max(candidate["capacity_available"], shipment.weight_kg)) + candidate["transfer_cost"]
        candidate["score"] = (.35 * min(max(candidate["deadline_margin"], 0) / max(shipment.remaining_hours, 1), 1) + .20 * shipment.priority_score + .40 * (1 - min(candidate["cost"] / 8000, 1)) + .05 * min(shipment.weight_kg / max(candidate["capacity_available"], shipment.weight_kg), 1))
        candidate["feasible"] = candidate["capacity_available"] >= shipment.weight_kg and candidate["deadline_margin"] >= 0
        if candidate["feasible"]:
            options.append(candidate)
        else:
            reasons = []
            if candidate["capacity_available"] < shipment.weight_kg:
                reasons.append(f"capacity {candidate['capacity_available']:.0f}kg < {shipment.weight_kg:.0f}kg required")
            if candidate["deadline_margin"] < 0:
                reasons.append(f"ETA misses deadline by {abs(candidate['deadline_margin']):.1f}h")
            rejected_options.append({"vehicle_id": candidate["vehicle_id"], "strategy": candidate["strategy"], "eta_hours": round(candidate["eta_hours"], 1), "capacity_available": round(candidate["capacity_available"], 1), "reason": "; ".join(reasons)})
    if options:
        options_df = pd.DataFrame(sorted(options, key=lambda x: x.get("score", 0), reverse=True))
        st.dataframe(options_df[["vehicle_id", "strategy", "route", "cost", "eta_hours", "capacity_available", "transfer_cost"]], use_container_width=True, hide_index=True)
        best = options_df.iloc[0]
        st.success(f"Recommended: {best.vehicle_id} · {best.strategy} · {money(best.cost)} estimated · {best.eta_hours:.1f}h ETA")
        st.markdown("**Why?** Compatible route, hard capacity feasibility, deadline-safe ETA, then weighted priority/cost/utilization scoring.")
    else:
        st.markdown('<div class="planner-state">No feasible piggyback option found for this shipment.</div>', unsafe_allow_html=True)
        if rejected_options:
            st.markdown(f'<div class="planner-note"><strong>Why this happened:</strong> every candidate fails a hard constraint. This shipment has {shipment.remaining_hours:.1f} hours remaining; recovery requires at least one route with enough capacity and an ETA before the deadline.</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rejected_options), use_container_width=True, hide_index=True)
        else:
            st.markdown('<div class="planner-note"><strong>Why this happened:</strong> no compatible vehicle or one-hub route was found from the shipment\'s current location.</div>', unsafe_allow_html=True)
        st.info("Recommended alternatives: dedicated vehicle, next available route, alternate hub, or manual logistics intervention.")

elif page == "Global Recovery Plan":
    st.markdown("### Fleet-wide allocation")
    a, b, c, d = st.columns(4)
    a.metric("Recovered", recovered_count); b.metric("Escalated", len(plan["escalated"])); c.metric("Estimated cost", money(plan["total_cost"])); d.metric("Savings", money(plan["savings"]))
    if not allocations.empty:
        allocation_view = allocations[["shipment_id", "vehicle_id", "strategy", "route", "weight_kg", "eta_hours", "deadline_margin", "cost", "priority"]].copy()
        allocation_view.columns = ["Shipment", "Vehicle(s)", "Strategy", "Route", "Weight kg", "ETA hrs", "Margin hrs", "Est. cost", "Priority"]
        st.dataframe(allocation_view, use_container_width=True, hide_index=True, column_config={"Est. cost": st.column_config.NumberColumn(format="₹%,.0f"), "ETA hrs": st.column_config.NumberColumn(format="%.1f"), "Margin hrs": st.column_config.NumberColumn(format="%.1f")})
        with st.expander("View decision details"):
            st.dataframe(allocations[["shipment_id", "vehicle_id", "strategy", "route", "cost", "eta_hours", "deadline_margin", "score", "reason"]], use_container_width=True, hide_index=True)
    if not plan["escalated"].empty:
        st.markdown("#### Escalations")
        st.dataframe(plan["escalated"][["shipment_id", "destination", "weight_kg", "priority", "reason"]], use_container_width=True, hide_index=True)
    if not plan["rejected"].empty:
        with st.expander("Rejected candidate reasons"):
            st.dataframe(plan["rejected"], use_container_width=True, hide_index=True)

elif page == "Decision Audit":
    st.markdown("### What did the system decide, and why?")
    st.caption("This audit is generated from deterministic optimizer results. The AI explanation layer cannot modify these values.")
    if allocations.empty:
        st.info("No selected decisions are available for audit.")
    else:
        audit_rows = []
        for row in allocations.itertuples():
            audit_rows.append({"Shipment": row.shipment_id, "Selected vehicle(s)": row.vehicle_id, "Strategy": row.strategy, "Cost": money(row.cost), "ETA": f"{row.eta_hours:.1f} hrs", "Deadline margin": f"{row.deadline_margin:.1f} hrs", "Priority score": f"{row.priority_score:.2f}", "Status": "RECOVERED"})
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)
        for row in allocations.itertuples():
            with st.expander(f"{row.shipment_id} · {row.vehicle_id} · {row.strategy}"):
                st.markdown('<div class="audit-card">', unsafe_allow_html=True)
                st.markdown(f"**WHY SELECTED?** {row.reason}")
                factors = pd.DataFrame({"Factor": ["Route compatibility", "Deadline compliance", "Capacity availability", "Cost efficiency", "Priority satisfaction"], "Assessment": ["100%", f"{max(0, min(row.deadline_margin / max(row.eta_hours, 1), 1) * 100):.0f}%", "FEASIBLE", f"{max(0, (1 - row.cost / 8000) * 100):.0f}%", f"{row.priority_score * 100:.0f}%"]})
                st.dataframe(factors, hide_index=True, use_container_width=True)
                rejected = plan["rejected"]
                if not rejected.empty:
                    rejected_for_shipment = rejected[rejected.shipment_id == row.shipment_id]
                    if not rejected_for_shipment.empty:
                        st.markdown("**REJECTED ALTERNATIVES**")
                        st.dataframe(rejected_for_shipment, hide_index=True, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

elif page == "Network Intelligence":
    st.markdown("### Live route network")
    st.caption("Locations are shown as labeled nodes. Line thickness indicates the number of vehicles assigned to that corridor.")
    edge_counts = routes.merge(vehicles.groupby(["route_origin", "route_destination"], as_index=False).size().rename(columns={"size": "vehicles"}), left_on=["origin", "destination"], right_on=["route_origin", "route_destination"], how="left").fillna({"vehicles": 0})
    fig = go.Figure()
    positions = {city: (i % 4, -(i // 4)) for i, city in enumerate(sorted(set(routes.origin) | set(routes.destination)))}
    for row in edge_counts.itertuples():
        x0, y0 = positions[row.origin]; x1, y1 = positions[row.destination]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line={"width": 1 + row.vehicles, "color": "#8aa4aa"}, hovertext=f"{row.origin} → {row.destination} · {int(row.vehicles)} vehicles", showlegend=False))
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2, arrowcolor="#14566a")
    fig.add_trace(go.Scatter(x=[positions[c][0] for c in positions], y=[positions[c][1] for c in positions], mode="markers", hovertext=list(positions), hovertemplate="%{hovertext}<extra></extra>", marker={"size": 24, "color": "#d47a3f", "line": {"width": 3, "color": "#ffffff"}}, showlegend=False))
    for city, (x_position, y_position) in positions.items():
        fig.add_annotation(x=x_position, y=y_position, text=f"<b>{city}</b>", showarrow=False, yshift=25, font={"family": "DM Sans, sans-serif", "size": 13, "color": "#17313d"}, bgcolor="#ffffff", bordercolor="#d47a3f", borderwidth=1, borderpad=4, opacity=0.98)
    fig.update_layout(height=600, xaxis={"visible": False, "range": [-0.7, 3.7]}, yaxis={"visible": False, "range": [-3.2, 0.9]}, plot_bgcolor="#e9f0f2", paper_bgcolor="#f3f6f8", margin={"l": 20, "r": 20, "t": 30, "b": 20}, hoverlabel={"bgcolor": "#092b3d", "font": {"color": "#ffffff"}})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(edge_counts[["route_id", "origin", "destination", "distance_km", "estimated_hours", "delay_risk", "vehicles"]], use_container_width=True, hide_index=True)

elif page == "What-If Simulator":
    st.markdown("### Rerun the plan against changed conditions")
    vehicle_change = st.selectbox("Vehicle becomes unavailable", ["None"] + vehicles.vehicle_id.tolist())
    capacity_loss = st.slider("Reduce all vehicle capacity (%)", 0, 70, 0, 5)
    urgent = st.checkbox("Add a new urgent 80kg shipment to Bengaluru → Chennai")
    shorter_deadlines = st.slider("Shorten deadlines (hours)", 0, 12, 0)
    if st.button("RUN SIMULATION", type="primary"):
        simulated_vehicles = vehicles.copy()
        if vehicle_change != "None": simulated_vehicles.loc[simulated_vehicles.vehicle_id == vehicle_change, "vehicle_status"] = "Unavailable"
        simulated_vehicles["capacity_kg"] = (simulated_vehicles.capacity_kg * (1 - capacity_loss / 100)).round().astype(int)
        simulated_shipments = shipments.copy()
        if shorter_deadlines: simulated_shipments["deadline"] = pd.to_datetime(simulated_shipments.deadline) - pd.to_timedelta(shorter_deadlines, unit="h")
        if urgent:
            row = simulated_shipments.iloc[0].copy(); row["shipment_id"] = "URGENT-NEW"; row["weight_kg"] = 80; row["priority"] = "Critical"; row["deadline"] = pd.Timestamp.now() + pd.Timedelta(hours=5); simulated_shipments = pd.concat([simulated_shipments, pd.DataFrame([row])], ignore_index=True)
        simulated = allocate_plan(simulated_shipments, simulated_vehicles, hubs, routes, weights)
        before_recovery = len(plan["allocations"]); after_recovery = len(simulated["allocations"])
        x, y, z = st.columns(3); x.metric("Recovered", after_recovery, after_recovery - before_recovery); y.metric("Escalated", len(simulated["escalated"]), len(simulated["escalated"]) - len(plan["escalated"])); z.metric("Estimated cost", money(simulated["total_cost"]), money(simulated["total_cost"] - plan["total_cost"]))
        st.dataframe(simulated["allocations"][["vehicle_id", "shipment_id", "strategy", "cost", "deadline_margin"]], use_container_width=True, hide_index=True)
        st.info("The plan was recomputed globally. Changes can cascade because capacity is shared across shipments.")

else:
    st.markdown("### AI logistics analyzer")
    st.write("Ask one question about the complete shipments, vehicles, hubs, routes, allocation, costs, risks, or escalations loaded by the app.")
    st.caption("Only the latest prompt and answer are shown. The agent uses RAG facts, the deterministic decision engine, and an optional LLM.")
    question = st.chat_input("Type your logistics question...")
    if question:
        st.session_state.ai_question = question
        st.session_state.ai_answer, st.session_state.ai_source = answer_question(question, plan, vehicles, hubs, routes)
    if st.session_state.get("ai_question"):
        st.markdown(f"**Your question**\n\n{st.session_state.ai_question}")
        st.success(st.session_state.ai_source)
        st.markdown(st.session_state.ai_answer)
        if st.button("CLEAR PREVIOUS ANSWER"):
            st.session_state.pop("ai_question", None)
            st.session_state.pop("ai_answer", None)
            st.session_state.pop("ai_source", None)
            st.rerun()
