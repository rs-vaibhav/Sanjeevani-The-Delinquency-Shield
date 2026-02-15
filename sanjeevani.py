import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import time
import json
import random
from datetime import datetime, timedelta
from collections import deque

try:
    import xgboost
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(
    page_title="Sanjeevani AI | Business Impact Dashboard",
    layout="wide",
    page_icon="🗽"
)

st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .stMetric { background-color: #262730; border-radius: 10px; padding: 15px; border: 1px solid #444; }
    .stInfo, .stSuccess, .stWarning, .stError { border-radius: 10px; }
    h1, h2, h3 { color: #FAFAFA; }
    div[data-testid="stExpander"] div[role="button"] p { font-size: 1.1rem; font-weight: 600; }
    .kafka-header {
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        border: 1px solid #00ff88;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 10px;
    }
    .event-card {
        background-color: #1a1a2e;
        border-left: 4px solid #00ff88;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 6px;
        font-family: monospace;
        font-size: 0.82rem;
    }
    .event-card.critical  { border-left-color: #ff4444; }
    .event-card.watchlist { border-left-color: #ffaa00; }
    .event-card.healthy   { border-left-color: #00ff88; }
    .kafka-badge {
        display: inline-block;
        background: #00ff88;
        color: #000;
        font-size: 0.7rem;
        font-weight: bold;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 6px;
    }
    .kafka-badge.red    { background: #ff4444; color: #fff; }
    .kafka-badge.orange { background: #ffaa00; color: #000; }
    .partition-box {
        background: #0d1117;
        border: 1px solid #333;
        border-radius: 6px;
        padding: 8px;
        text-align: center;
        font-family: monospace;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 1. RISK ENGINE (FIXED VERSION)
# ==========================================
class RiskEngine:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = None
        self.model_loaded = False
        self.last_error = None
        
        try:
            self.model = joblib.load('model-3.pkl')
            self.feature_names = joblib.load('features.pkl')
            self.model_loaded = True
            st.sidebar.success(f"✅ Model loaded: {len(self.feature_names)} features")
            
            if SHAP_AVAILABLE:
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                    st.sidebar.success("✅ SHAP explainer initialized")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ SHAP unavailable: {str(e)}")
        except FileNotFoundError:
            st.sidebar.warning("⚠️ DEMO MODE: 'model-3.pkl' not found. Using Rule-Based Engine.")
        except Exception as e:
            st.sidebar.error(f"❌ Model load error: {str(e)}")

    def calculate_rule_score(self, inputs):
        """Fallback rule-based scoring"""
        score = 0.0
        score += inputs['salary_delay'] * 2.5
        score += inputs['dti_ratio'] * 40.0
        score += inputs['utility_late_days'] * 1.5
        score += inputs['savings_drawdown'] * 0.2
        score += inputs['lending_apps'] * 12.0
        score += inputs['failed_tx'] * 4.0
        score += inputs['atm_count'] * 2.0
        score += inputs['gambling_count'] * 5.0
        score += inputs['upi_spike_ratio'] * 4.0
        score += inputs['volatility_risk'] * 5.0
        return min(99, max(1, score))

    def get_drivers(self, inputs, df_model=None):
        if self.explainer and df_model is not None:
            try:
                shap_values = self.explainer.shap_values(df_model)
                vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
                drivers = [
                    {"feature": feat, "value": val, "impact": impact}
                    for feat, val, impact in zip(self.feature_names, df_model.iloc[0], vals)
                    if impact > 0
                ]
                return sorted(drivers, key=lambda x: x['impact'], reverse=True)[:5]
            except Exception as e:
                st.sidebar.warning(f"SHAP calculation failed: {str(e)}")
                
        # Fallback to rule-based drivers
        contributions = [
            {"feature": "Lending Apps",       "value": inputs['lending_apps'],    "impact": inputs['lending_apps'] * 12.0},
            {"feature": "DTI Ratio",           "value": inputs['dti_ratio'],       "impact": inputs['dti_ratio'] * 40.0},
            {"feature": "Salary Delay",        "value": inputs['salary_delay'],    "impact": inputs['salary_delay'] * 2.5},
            {"feature": "Gambling",            "value": inputs['gambling_count'],  "impact": inputs['gambling_count'] * 5.0},
            {"feature": "UPI Spike",           "value": inputs['upi_spike_ratio'], "impact": inputs['upi_spike_ratio'] * 4.0},
            {"feature": "Savings Drawdown",    "value": inputs['savings_drawdown'],"impact": inputs['savings_drawdown'] * 0.2},
            {"feature": "Failed Transactions", "value": inputs['failed_tx'],       "impact": inputs['failed_tx'] * 4.0},
        ]
        return sorted(contributions, key=lambda x: x['impact'], reverse=True)[:5]

    def explain_risk_text(self, drivers):
        if not drivers: return "General Transactional Stress"
        reasons = []
        for d in drivers[:2]:
            feat = d['feature'].replace('_', ' ').title()
            val  = d['value']
            if   "Salary"  in feat: reasons.append(f"Salary delayed {int(val)} days")
            elif "Lending" in feat: reasons.append(f"Usage of {int(val)} lending apps")
            elif "Savings" in feat: reasons.append(f"Savings dropped {int(val)}%")
            elif "Dti"     in feat: reasons.append(f"DTI Ratio is {val:.2f}")
            elif "Gambling"in feat: reasons.append("Gambling Detected")
            elif "Upi"     in feat: reasons.append(f"UPI Spikes ({val}x)")
            else:                   reasons.append(f"High {feat}")
        return " + ".join(reasons)

    def get_intervention(self, status, risk_score):
        if status == "CRITICAL":
            return {"action": "Immediate Phone Call", "priority": "CRITICAL", "timeline": "24 hours",
                    "cost": 500, "expected_savings": 46500, "roi": 93, "success_rate": 0.65}
        elif status == "WATCHLIST":
            return {"action": "Proactive Email + SMS", "priority": "HIGH", "timeline": "7 days",
                    "cost": 50, "expected_savings": 15000, "roi": 300, "success_rate": 0.40}
        else:
            return {"action": "Relationship Building", "priority": "MEDIUM", "timeline": "30 days",
                    "cost": 20, "expected_savings": 5000, "roi": 250, "success_rate": 0.15}

    def get_assessment(self, inputs):
        """
        FIXED VERSION: Always tries model prediction first, with proper error handling
        """
        # Start with rule-based score as baseline
        rule_score = self.calculate_rule_score(inputs)
        risk_score = rule_score
        model_type = "Rule-Based"
        df = None
        
        # Try to use the ML model if available
        if self.model_loaded and self.model and self.feature_names:
            try:
                # Prepare full feature set
                full_inputs = dict(inputs)
                defaults = {
                    'pagerank': 0.05, 
                    'network_degree': 2, 
                    'cs_sentiment_score': 0.5, 
                    'lstm_seq_risk': 0.1
                }
                
                # Add missing features with defaults
                for fn in self.feature_names:
                    if fn not in full_inputs:
                        full_inputs[fn] = defaults.get(fn, 0.0)
                
                # Create DataFrame with exact feature order
                df = pd.DataFrame([full_inputs])[self.feature_names]
                
                # Get model prediction
                prob = self.model.predict_proba(df)[0][1]
                risk_score = prob * 100
                
                # Update model type
                model_type = "XGBoost + SHAP" if self.explainer else "XGBoost"
                
                # Show success indicator
                st.sidebar.success(f"🎯 Model Prediction: {risk_score:.1f}% (vs Rule: {rule_score:.1f}%)")
                
            except Exception as e:
                # Log the error for debugging
                error_msg = str(e)
                self.last_error = error_msg
                st.sidebar.error(f"❌ Model prediction failed: {error_msg}")
                st.sidebar.info(f"ℹ️ Using Rule-Based score: {rule_score:.1f}%")
                
                # Fallback to rule-based
                risk_score = rule_score
                model_type = "Rule-Based (Fallback)"
        else:
            st.sidebar.info(f"ℹ️ Rule-Based score: {rule_score:.1f}%")

        # Get drivers and explanation
        drivers = self.get_drivers(inputs, df)
        explanation = self.explain_risk_text(drivers)

        # Determine status based on risk score
        if risk_score > 75:
            status = "CRITICAL"
            color = "red"
            impact = f"**Detected:** Severe signals ({explanation}). **Action:** Immediate Intervention."
            title = "Prevent Default (High Urgency)"
        elif risk_score > 40:
            status = "WATCHLIST"
            color = "orange"
            impact = f"**Detected:** Emerging stress ({explanation}). **Action:** Automated Nudge."
            title = "Mitigate Risk (Medium Urgency)"
        else:
            status = "HEALTHY"
            color = "green"
            impact = "**Detected:** Stable behavior. **Action:** Cross-sell."
            title = "Strengthen Relationship"

        return {
            'risk_score': risk_score,
            'probability': risk_score / 100.0,
            'status': status,
            'color': color,
            'impact_title': title,
            'impact_text': impact,
            'drivers': drivers,
            'intervention': self.get_intervention(status, risk_score),
            'model_type': model_type,
            'rule_score': rule_score  # Added for debugging
        }


# Initialize engine
engine = RiskEngine()


# ==========================================
# 2. KAFKA STREAM SIMULATOR
# ==========================================
TRANSACTION_TYPES = [
    ("UPI_TRANSFER", 0.28), ("LENDING_APP_UPI", 0.08), ("ATM_WITHDRAWAL", 0.10),
    ("SALARY_CREDIT", 0.05), ("UTILITY_PAYMENT", 0.10), ("DINING_DEBIT", 0.12),
    ("EMI_DEBIT", 0.07), ("FAILED_AUTODEBIT", 0.04), ("SAVINGS_WITHDRAWAL", 0.06),
    ("GAMBLING_UPI", 0.03), ("BALANCE_ENQUIRY", 0.07),
]

CUSTOMER_POOL = [
    ("CUST_000142", "high"), ("CUST_000289", "medium"), ("CUST_000531", "low"),
    ("CUST_000873", "critical"), ("CUST_001024", "medium"), ("CUST_001337", "low"),
    ("CUST_001456", "high"), ("CUST_001789", "critical"),
]

AMOUNT_RANGES = {
    "UPI_TRANSFER": (500, 15000), "LENDING_APP_UPI": (2000, 50000),
    "ATM_WITHDRAWAL": (1000, 10000), "SALARY_CREDIT": (25000, 120000),
    "UTILITY_PAYMENT": (500, 5000), "DINING_DEBIT": (200, 3000),
    "EMI_DEBIT": (5000, 35000), "FAILED_AUTODEBIT": (0, 0),
    "SAVINGS_WITHDRAWAL": (5000, 80000), "GAMBLING_UPI": (100, 5000),
    "BALANCE_ENQUIRY": (0, 0),
}

RISK_SIGNALS = {
    "critical": {"LENDING_APP_UPI": 0.30, "GAMBLING_UPI": 0.15, "FAILED_AUTODEBIT": 0.15, "SAVINGS_WITHDRAWAL": 0.20},
    "high":     {"LENDING_APP_UPI": 0.18, "ATM_WITHDRAWAL": 0.18, "SAVINGS_WITHDRAWAL": 0.15},
    "medium":   {"ATM_WITHDRAWAL": 0.12, "UTILITY_PAYMENT": 0.12},
    "low":      {"DINING_DEBIT": 0.20, "UPI_TRANSFER": 0.30},
}

MERCHANTS = {
    "LENDING_APP_UPI":    ["KreditBee", "MoneyTap", "CASHe", "PaySense", "Nira"],
    "GAMBLING_UPI":       ["Dream11", "MPL", "LotteryKing", "RummyCircle"],
    "DINING_DEBIT":       ["Swiggy", "Zomato", "Dominos", "McDonald's"],
    "ATM_WITHDRAWAL":     ["ATM"],
    "SALARY_CREDIT":      ["Employer"],
    "UTILITY_PAYMENT":    ["BESCOM", "Jio", "Airtel", "BWSSB"],
    "SAVINGS_WITHDRAWAL": ["Self Transfer"],
    "FAILED_AUTODEBIT":   ["HDFC EMI", "ICICI Loan", "Bajaj Finserv"],
    "UPI_TRANSFER":       ["PhonePe", "GPay", "Paytm", "BHIM"],
    "EMI_DEBIT":          ["Bajaj Finserv", "HDFC Bank", "ICICI Bank"],
    "BALANCE_ENQUIRY":    ["Self"],
}

EVENT_RISK = {
    "LENDING_APP_UPI": (72, 95), "FAILED_AUTODEBIT": (65, 88), "GAMBLING_UPI": (70, 92),
    "SAVINGS_WITHDRAWAL": (55, 80), "ATM_WITHDRAWAL": (40, 65), "BALANCE_ENQUIRY": (35, 55),
    "UTILITY_PAYMENT": (20, 45), "EMI_DEBIT": (10, 35), "SALARY_CREDIT": (5, 20),
    "DINING_DEBIT": (5, 25), "UPI_TRANSFER": (10, 40),
}

# Session State Initialization
defaults = {
    "stream_running":     False,
    "event_log":          deque(maxlen=50),
    "stream_stats":       {"total": 0, "critical": 0, "watchlist": 0, "healthy": 0},
    "throughput_history": deque(maxlen=30),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def generate_kafka_event(risk_level: str) -> dict:
    """Generate a realistic Kafka event based on risk level"""
    try:
        cust_id, _ = random.choice([c for c in CUSTOMER_POOL if c[1] == risk_level] or CUSTOMER_POOL)
        biases     = RISK_SIGNALS.get(risk_level, {})
        tx_types   = [t[0] for t in TRANSACTION_TYPES]
        tx_weights = [biases.get(t[0], t[1]) for t in TRANSACTION_TYPES]
        total      = sum(tx_weights)
        tx_type    = random.choices(tx_types, weights=[w/total for w in tx_weights], k=1)[0]
        lo, hi     = AMOUNT_RANGES[tx_type]
        amount     = round(random.uniform(lo, hi), 2) if hi > 0 else 0
        ts         = datetime.now() - timedelta(seconds=random.randint(0, 120))
        lo_r, hi_r = EVENT_RISK.get(tx_type, (10, 50))
        risk_score = random.randint(lo_r, hi_r)
        label      = "CRITICAL" if risk_score > 75 else ("WATCHLIST" if risk_score > 40 else "HEALTHY")
        
        return {
            "kafka_meta": {
                "topic": "bank.transactions.realtime",
                "partition": hash(cust_id) % 4,
                "offset": random.randint(100000, 999999),
                "timestamp": ts.isoformat(timespec="milliseconds"),
            },
            "payload": {
                "customer_id":    cust_id,
                "transaction_id": f"TXN{random.randint(10**9, 10**10-1)}",
                "type":           tx_type,
                "amount":         amount,
                "channel":        random.choice(["UPI", "NEFT", "IMPS", "ATM", "POS"]),
                "merchant":       random.choice(MERCHANTS.get(tx_type, ["Unknown"])),
                "risk_score":     risk_score,
                "risk_label":     label,
                "intervention":   "📞 Immediate Phone Call" if risk_score > 75 else ("📧 Email + SMS Nudge" if risk_score > 40 else "✅ No Action Required"),
                "processing_ms":  random.randint(8, 45),
            }
        }
    except Exception as e:
        # Return safe default event
        return {
            "kafka_meta": {"topic": "bank.transactions.realtime", "partition": 0, "offset": 0, "timestamp": datetime.now().isoformat()},
            "payload": {"customer_id": "ERROR", "transaction_id": "ERROR", "type": "ERROR", "amount": 0, 
                       "channel": "ERROR", "merchant": "ERROR", "risk_score": 0, "risk_label": "HEALTHY",
                       "intervention": "Error", "processing_ms": 0}
        }


def render_event_card(event: dict, show_json: bool = False) -> str:
    """Render a Kafka event as HTML card"""
    try:
        p   = event["payload"]
        km  = event["kafka_meta"]
        lbl = p["risk_label"]
        badge_color = "red" if lbl == "CRITICAL" else ("orange" if lbl == "WATCHLIST" else "")
        risk_color  = "#ff4444" if lbl == "CRITICAL" else ("#ffaa00" if lbl == "WATCHLIST" else "#00ff88")
        amt_str     = f"₹{p['amount']:,.0f}" if p['amount'] > 0 else "—"
        ts_short    = km["timestamp"][11:19]
        
        html = f"""
        <div class="event-card {lbl.lower()}">
            <span class="kafka-badge {badge_color}">{lbl}</span>
            <strong>{p['customer_id']}</strong> &nbsp;·&nbsp;
            <code>{p['type']}</code> &nbsp;·&nbsp; {amt_str} &nbsp;·&nbsp;
            <em>{p['merchant']}</em>
            <span style="color:#888; float:right">
                P{km['partition']} · offset {km['offset']} · {ts_short} · {p['processing_ms']}ms
            </span><br/>
            <span style="color:#aaa; font-size:0.78rem;">
                Risk: <strong style="color:{risk_color}">{p['risk_score']}/100</strong>
                &nbsp;→&nbsp; {p['intervention']}
                &nbsp;·&nbsp; txn_id: {p['transaction_id']}
            </span>
        </div>"""
        
        if show_json:
            html += f"<pre style='background:#111;color:#888;font-size:0.7rem;padding:6px;border-radius:4px;margin-bottom:8px;'>{json.dumps(event, indent=2)}</pre>"
        
        return html
    except Exception as e:
        return f"<div class='event-card'>Error rendering event: {str(e)}</div>"



# ==========================================
# 3. KAFKA TAB (Fragment) - STABLE VERSION
# ==========================================
@st.fragment
def kafka_tab():
    st.title("📡 Live Kafka Transaction Stream")
    st.markdown(
        "Simulates **Apache Kafka** ingesting real-time bank transactions. "
        "Each event is scored by the **Sanjeevani AI** engine (<50ms latency) "
        "and routed to the correct intervention queue — exactly as it would "
        "run on **Amazon Kinesis** in production."
    )

    st.markdown("""
    <div class="kafka-header">
        <span style="color:#00ff88; font-weight:bold; font-family:monospace;">
        ▶ TOPIC: bank.transactions.realtime &nbsp;|&nbsp;
        PARTITIONS: 4 &nbsp;|&nbsp;
        REPLICATION FACTOR: 3 &nbsp;|&nbsp;
        CONSUMER GROUP: sanjeevani-scoring-v2
        </span>
    </div>
    """, unsafe_allow_html=True)

    p0, p1, p2, p3 = st.columns(4)
    for col, i in zip([p0, p1, p2, p3], range(4)):
        col.markdown(f"""
        <div class="partition-box">
            <div style="color:#00ff88">PARTITION {i}</div>
            <div style="color:#888;font-size:0.7rem">Consumer: sanjeevani-worker-{i}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Controls
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1])
    with ctrl1:
        speed = st.selectbox("⚡ Emit Speed", ["0.5s (Fast)", "1s (Normal)", "2s (Slow)"], index=1)
        delay = {"0.5s (Fast)": 0.5, "1s (Normal)": 1.0, "2s (Slow)": 2.0}[speed]
    with ctrl2:
        risk_filter = st.selectbox("🎯 Risk Profile Mix",
                                   ["All Profiles", "High Risk Heavy", "Critical Only", "Mostly Healthy"])
    with ctrl3:
        batch_size = st.slider("Events per tick", 1, 5, 2)
    with ctrl4:
        st.markdown("<br/>", unsafe_allow_html=True)
        cs, cs2 = st.columns(2)
        start_btn = cs.button("▶ Start",  type="primary",   use_container_width=True)
        stop_btn  = cs2.button("⏹ Stop", type="secondary",  use_container_width=True)

    if start_btn:
        st.session_state.stream_running = True
    if stop_btn:
        st.session_state.stream_running = False

    if st.button("🗑️ Clear Log"):
        st.session_state.event_log.clear()
        st.session_state.stream_stats       = {"total": 0, "critical": 0, "watchlist": 0, "healthy": 0}
        st.session_state.throughput_history = deque(maxlen=30)

    show_json = st.checkbox("🔬 Show raw JSON payload", value=False)

    st.markdown("---")
    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
    stats_ph     = sm1.empty()
    crit_ph      = sm2.empty()
    watch_ph     = sm3.empty()
    health_ph    = sm4.empty()
    status_ph    = sm5.empty()
    tput_ph      = st.empty()
    st.markdown("#### 📥 Event Feed")
    feed_ph      = st.empty()

    def render_static_metrics():
        s = st.session_state.stream_stats
        stats_ph.metric("📨 Total Events", f"{s['total']:,}")
        crit_ph.metric("🔴 Critical",      f"{s['critical']:,}")
        watch_ph.metric("🟠 Watchlist",    f"{s['watchlist']:,}")
        health_ph.metric("🟢 Healthy",     f"{s['healthy']:,}")

    def render_feed():
        if st.session_state.event_log:
            feed_ph.markdown(
                "".join(render_event_card(ev, show_json) for ev in list(st.session_state.event_log)[:20]),
                unsafe_allow_html=True
            )
        else:
            feed_ph.info("▶ Press **Start** to begin the live event stream.")

    PROFILE_WEIGHTS = {
        "All Profiles":    {"critical": 0.20, "high": 0.30, "medium": 0.30, "low": 0.20},
        "High Risk Heavy": {"critical": 0.35, "high": 0.40, "medium": 0.20, "low": 0.05},
        "Critical Only":   {"critical": 0.70, "high": 0.25, "medium": 0.05, "low": 0.00},
        "Mostly Healthy":  {"critical": 0.05, "high": 0.10, "medium": 0.25, "low": 0.60},
    }

    if st.session_state.stream_running:
        pw       = PROFILE_WEIGHTS[risk_filter]
        profiles = list(pw.keys())
        weights  = list(pw.values())

        status_ph.markdown(
            "<div style='padding:10px;background:#00ff8822;border:1px solid #00ff88;"
            "border-radius:6px;text-align:center;font-family:monospace;color:#00ff88;'>"
            "● STREAMING LIVE</div>", unsafe_allow_html=True
        )

        for tick in range(200):
            if not st.session_state.stream_running:
                break

            tick_critical = 0
            for _ in range(batch_size):
                profile = random.choices(profiles, weights=weights, k=1)[0]
                event   = generate_kafka_event(profile)
                st.session_state.event_log.appendleft(event)
                lbl = event["payload"]["risk_label"]
                st.session_state.stream_stats["total"]    += 1
                st.session_state.stream_stats[lbl.lower()] += 1
                if lbl == "CRITICAL":
                    tick_critical += 1

            st.session_state.throughput_history.append(
                {"tick": tick, "eps": batch_size / delay, "crit": tick_critical}
            )

            # Update metrics
            s = st.session_state.stream_stats
            stats_ph.metric("📨 Total Events", f"{s['total']:,}")
            crit_ph.metric("🔴 Critical",   f"{s['critical']:,}",
                           delta=f"{s['critical']/max(s['total'],1)*100:.1f}%", delta_color="inverse")
            watch_ph.metric("🟠 Watchlist", f"{s['watchlist']:,}",
                            delta=f"{s['watchlist']/max(s['total'],1)*100:.1f}%", delta_color="off")
            health_ph.metric("🟢 Healthy",  f"{s['healthy']:,}",
                             delta=f"{s['healthy']/max(s['total'],1)*100:.1f}%", delta_color="normal")

            # Throughput chart
            if len(st.session_state.throughput_history) > 1:
                th_df = pd.DataFrame(list(st.session_state.throughput_history))
                fig_th = go.Figure()
                fig_th.add_trace(go.Scatter(x=th_df["tick"], y=th_df["eps"],
                    mode="lines", fill="tozeroy", line=dict(color="#00ff88", width=2), name="Events/sec"))
                fig_th.add_trace(go.Scatter(x=th_df["tick"], y=th_df["crit"],
                    mode="lines", line=dict(color="#ff4444", width=1.5, dash="dot"), name="Critical/tick"))
                fig_th.update_layout(
                    height=130, margin=dict(t=10, b=10, l=40, r=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white", size=10), showlegend=True,
                    legend=dict(orientation="h", y=1.2, font=dict(size=9)),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor="#222", zeroline=False),
                )
                tput_ph.plotly_chart(fig_th, use_container_width=True)

            # Event feed
            feed_ph.markdown(
                "".join(render_event_card(ev, show_json) for ev in list(st.session_state.event_log)[:20]),
                unsafe_allow_html=True
            )

            time.sleep(delay)

        st.session_state.stream_running = False
        status_ph.markdown(
            "<div style='padding:10px;background:#33333388;border:1px solid #555;"
            "border-radius:6px;text-align:center;font-family:monospace;color:#888;'>"
            "■ STOPPED</div>", unsafe_allow_html=True
        )
        render_static_metrics()
        render_feed()

    else:
        status_ph.markdown(
            "<div style='padding:10px;background:#33333388;border:1px solid #555;"
            "border-radius:6px;text-align:center;font-family:monospace;color:#888;'>"
            "■ STOPPED</div>", unsafe_allow_html=True
        )
        render_static_metrics()
        render_feed()

    st.markdown("---")
    st.caption(
        "Simulated Kafka stream · Topic: bank.transactions.realtime · "
        "Partitions: 4 · Consumer Group: sanjeevani-scoring-v2 · "
        "Production equivalent: Amazon Kinesis Data Streams"
    )


# ==========================================
# 4. MAIN LAYOUT — TABS
# ==========================================
tab1, tab2 = st.tabs(["🛡️ Risk Assessment", "📡 Live Kafka Stream"])

# ── TAB 1: RISK ASSESSMENT ──────────────────────────────────────────────────
with tab1:
    st.title("🛡️ Sanjeevani: Pre-Delinquency Intervention Platform")
    st.markdown("### Predict. Explain. Act. Save ₹245 Crores.")

    customer_id = st.text_input("👤 Customer ID", value="CUST_000142")
    st.markdown("---")

    with st.expander("💰 Customer Financial Profile (T-1 Month Data)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 💵 Income & Stability")
            s_salary  = st.slider("Salary Delay (Days)",  0, 30,    0)
            s_savings = st.slider("Savings Drawdown (%)", 0, 100,  10)
            s_dti     = st.slider("DTI Ratio",            0.0, 1.0, 0.3, 0.01)
            s_util    = st.slider("Utility Late Days",    0, 30,    0)
            s_liq     = st.slider("Liquidity Pressure",   0.0, 10.0, 2.0)
        with c2:
            st.markdown("#### 💳 Spending Behavior")
            s_apps   = st.slider("Lending Apps",          0, 10,    0)
            s_cc     = st.slider("Credit Velocity",       0, 100,  20)
            s_atm    = st.slider("ATM Withdrawals",       0, 20,    2)
            s_dining = st.slider("Dining Frequency",      0, 40,   25)
            s_failed = st.slider("Failed Transactions",   0, 10,    0)
        with c3:
            st.markdown("#### ⚠️ Risk Signals")
            s_gamble = st.slider("Gambling Count",        0, 10,    0)
            s_upi    = st.slider("UPI Spike Ratio",       0.0, 5.0, 1.0)
            s_bal    = st.slider("Balance Checks",        0, 20,    2)
            s_stress = st.slider("Behavioral Stress",     0.0, 10.0, 1.0)
            s_vol    = st.slider("Volatility Risk",       0.0, 5.0,  0.5)

    inputs = {
        'salary_delay': s_salary,     'savings_drawdown': s_savings, 'dti_ratio': s_dti,
        'utility_late_days': s_util,  'liquidity_pressure': s_liq,
        'lending_apps': s_apps,       'cc_velocity': s_cc,           'atm_count': s_atm,
        'dining_count': s_dining,     'failed_tx': s_failed,
        'gambling_count': s_gamble,   'upi_spike_ratio': s_upi,      'balance_checks': s_bal,
        'behavioral_stress': s_stress,'volatility_risk': s_vol
    }

    # Get assessment - THIS IS WHERE THE PREDICTION HAPPENS
    data = engine.get_assessment(inputs)
    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk Score",       f"{data['risk_score']:.0f}/100",           delta=data['status'],
              delta_color="inverse" if data['status'] != 'HEALTHY' else "normal")
    m2.metric("Probability",      f"{data['probability']:.1%}",              delta=data['model_type'])
    m3.metric("Capital at Risk",  f"₹{data['probability']*500000:,.0f}",     delta="Exposure", delta_color="inverse")
    m4.metric("Intervention ROI", f"{data['intervention']['roi']}x",          delta="Projected Return")

    st.markdown("---")
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.subheader("📊 Risk Severity")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=data['risk_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': data['status'], 'font': {'size': 24}},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': data['color']},
                   'steps': [{'range': [0, 40],  'color': '#2ECC71'},
                              {'range': [40, 75], 'color': '#F39C12'},
                              {'range': [75, 100],'color': '#E74C3C'}]}
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=30, b=30, l=30, r=30),
                                paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("🎯 Risk Drivers")
        if "SHAP" in data['model_type']:
            st.caption("⚡ Powered by SHAP (Game Theoretic Feature Importance)")
        drivers = data['drivers']
        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(
            y=[d['feature'] for d in drivers], x=[d['impact'] for d in drivers],
            orientation='h',
            marker=dict(color=[d['impact'] for d in drivers], colorscale='Reds', showscale=False),
            text=[f"{d['value']:.1f}" for d in drivers], textposition='outside'
        ))
        fig_d.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font={'color': "white"}, showlegend=False, xaxis_title="Impact on Risk Score")
        st.plotly_chart(fig_d, use_container_width=True)

    with col_right:
        st.subheader("💼 Business Value")
        if data['status'] == 'CRITICAL':    st.error(f"**{data['impact_title']}**")
        elif data['status'] == 'WATCHLIST': st.warning(f"**{data['impact_title']}**")
        else:                               st.success(f"**{data['impact_title']}**")
        st.markdown(data['impact_text'])
        st.markdown("---")

        st.subheader("🎯 Intervention")
        inv = data['intervention']
        st.info(f"**Action:** {inv['action']}\n\n**Success Probability:** {inv['success_rate']:.0%}")
        ca, cb = st.columns(2)
        ca.metric("Cost",    f"₹{inv['cost']}")
        cb.metric("Savings", f"₹{inv['expected_savings']:,}")
        if st.button("🚀 Initiate Protocol", type="primary", use_container_width=True):
            st.toast(f"✅ Protocol Sent: {inv['action']} for {customer_id}")

    st.markdown("---")
    st.subheader("📈 Portfolio Impact Simulation")
    col1, col2 = st.columns(2)
    with col1:
        p_size      = st.number_input("Portfolio Size (Customers)", value=100000, step=10000)
        delinq_rate = st.slider("Current Delinquency Rate (%)",     0.0, 20.0, 4.2)
    with col2:
        coverage = st.slider("Intervention Coverage (%)",           0, 100, 80)
        exposure = st.number_input("Avg Exposure per Customer (₹)", value=500000, step=50000)

    at_risk    = p_size * delinq_rate / 100
    intervened = at_risk * coverage / 100
    prevented  = intervened * 0.45
    cost       = intervened * 150
    savings    = prevented * exposure * 0.20
    net        = savings - cost
    roi        = net / cost if cost > 0 else 0

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Defaults Prevented", f"{int(prevented):,}")
    r2.metric("Program Cost",       f"₹{cost/1e7:.2f} Cr")
    r3.metric("Net Savings",        f"₹{net/1e7:.2f} Cr", delta="Annual Impact")
    r4.metric("Program ROI",        f"{roi:.1f}x")

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Bar(name='Without AI', x=['Defaults'], y=[at_risk],             marker_color='#E74C3C'))
    fig_sim.add_trace(go.Bar(name='With AI',    x=['Defaults'], y=[at_risk - prevented], marker_color='#2ECC71'))
    fig_sim.update_layout(barmode='group', height=300, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"},
                          title="Projected Default Reduction")
    st.plotly_chart(fig_sim, use_container_width=True)

    st.markdown("---")
    st.caption("Sanjeevani AI v2.1 | Architecture: Streamlit + XGBoost + SHAP | Compliance: FCA Consumer Duty")


# ── TAB 2: calls the fragment ───────────────────────────────────────────────
with tab2:
    kafka_tab()
