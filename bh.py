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

st.set_page_config(page_title="Sanjeevani AI | Business Impact Dashboard", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .stMetric { background-color: #262730; border-radius: 10px; padding: 15px; border: 1px solid #444; }
    .stInfo, .stSuccess, .stWarning, .stError { border-radius: 10px; }
    h1, h2, h3 { color: #FAFAFA; }
    div[data-testid="stExpander"] div[role="button"] p { font-size: 1.1rem; font-weight: 600; }

    /* Kafka Stream Styles */
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
    .event-card.critical { border-left-color: #ff4444; }
    .event-card.watchlist { border-left-color: #ffaa00; }
    .event-card.healthy { border-left-color: #00ff88; }
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
    .kafka-badge.red { background: #ff4444; color: #fff; }
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
# 1. INTELLIGENT RISK ENGINE
# ==========================================
class RiskEngine:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = None
        
        try:
            self.model = joblib.load('model-3.pkl') 
            self.feature_names = joblib.load('features.pkl')
            if SHAP_AVAILABLE:
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                except Exception:
                    pass
        except FileNotFoundError:
            st.sidebar.warning("DEMO MODE: 'model-3.pkl' not found. Using Logic Engine.")
        except Exception as e:
            st.sidebar.warning(f"DEMO MODE: {e}")

    def calculate_rule_score(self, inputs):
        """Fallback Logic Engine"""
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
        drivers = []
        if self.explainer and df_model is not None:
            try:
                shap_values = self.explainer.shap_values(df_model)
                if isinstance(shap_values, list): vals = shap_values[1][0]
                else: vals = shap_values[0]
                for feat, val, impact in zip(self.feature_names, df_model.iloc[0], vals):
                    if impact > 0:
                        drivers.append({"feature": feat, "value": val, "impact": impact})
                drivers.sort(key=lambda x: x['impact'], reverse=True)
                return drivers[:5]
            except Exception:
                pass

        contributions = [
            {"feature": "Lending Apps",      "value": inputs['lending_apps'],    "impact": inputs['lending_apps'] * 12.0},
            {"feature": "DTI Ratio",          "value": inputs['dti_ratio'],       "impact": inputs['dti_ratio'] * 40.0},
            {"feature": "Salary Delay",       "value": inputs['salary_delay'],    "impact": inputs['salary_delay'] * 2.5},
            {"feature": "Gambling",           "value": inputs['gambling_count'],  "impact": inputs['gambling_count'] * 5.0},
            {"feature": "UPI Spike",          "value": inputs['upi_spike_ratio'], "impact": inputs['upi_spike_ratio'] * 4.0},
            {"feature": "Savings Drawdown",   "value": inputs['savings_drawdown'],"impact": inputs['savings_drawdown'] * 0.2},
            {"feature": "Failed Transactions","value": inputs['failed_tx'],       "impact": inputs['failed_tx'] * 4.0},
        ]
        contributions.sort(key=lambda x: x['impact'], reverse=True)
        return contributions[:5]

    def explain_risk_text(self, drivers):
        if not drivers: return "General Transactional Stress"
        top = drivers[:2]
        reasons = []
        for d in top:
            feat = d['feature'].replace('_', ' ').title()
            val = d['value']
            if "Salary" in feat:  reasons.append(f"Salary delayed {int(val)} days")
            elif "Lending" in feat: reasons.append(f"Usage of {int(val)} lending apps")
            elif "Savings" in feat: reasons.append(f"Savings dropped {int(val)}%")
            elif "Dti" in feat:   reasons.append(f"DTI Ratio is {val:.2f}")
            elif "Gambling" in feat: reasons.append("Gambling Detected")
            elif "Upi" in feat:   reasons.append(f"UPI Spikes ({val}x)")
            else: reasons.append(f"High {feat}")
        return " + ".join(reasons)

    def get_intervention(self, status, risk_score):
        if status == "CRITICAL":
            return {"action": "Immediate Phone Call", "priority": "CRITICAL", "timeline": "24 hours",
                    "offers": ["30-Day Payment Holiday", "EMI Restructure", "Emergency Freeze"],
                    "cost": 500, "expected_savings": 46500, "roi": 93, "success_rate": 0.65}
        elif status == "WATCHLIST":
            return {"action": "Proactive Email + SMS", "priority": "HIGH", "timeline": "7 days",
                    "offers": ["Spending Insights", "Early Payment Reminder", "Auto-Save Plan"],
                    "cost": 50, "expected_savings": 15000, "roi": 300, "success_rate": 0.40}
        else:
            return {"action": "Relationship Building", "priority": "MEDIUM", "timeline": "30 days",
                    "offers": ["Limit Increase", "Premium Upgrade"],
                    "cost": 20, "expected_savings": 5000, "roi": 250, "success_rate": 0.15}

    def get_assessment(self, inputs):
        rule_score = self.calculate_rule_score(inputs)
        risk_score = rule_score
        model_type = "Rule-Based"
        df = None

        if self.model and self.feature_names:
            try:
                full_inputs = dict(inputs)
                defaults = {'pagerank': 0.05, 'network_degree': 2, 'cs_sentiment_score': 0.5, 'lstm_seq_risk': 0.1}
                for fn in self.feature_names:
                    if fn not in full_inputs: full_inputs[fn] = defaults.get(fn, 0.0)
                df = pd.DataFrame([full_inputs])[self.feature_names]
                prob = self.model.predict_proba(df)[0][1]
                risk_score = prob * 100
                model_type = "XGBoost + SHAP" if self.explainer else "XGBoost"
            except Exception:
                pass

        drivers = self.get_drivers(inputs, df)
        explanation = self.explain_risk_text(drivers)

        if risk_score > 75:
            status = "CRITICAL"; color = "red"
            impact = f"**Detected:** Severe signals ({explanation}). **Action:** Immediate Intervention."
            title = "Prevent Default (High Urgency)"
        elif risk_score > 40:
            status = "WATCHLIST"; color = "orange"
            impact = f"**Detected:** Emerging stress ({explanation}). **Action:** Automated Nudge."
            title = " Mitigate Risk (Medium Urgency)"
        else:
            status = "HEALTHY"; color = "green"
            impact = "**Detected:** Stable behavior. **Action:** Cross-sell."
            title = "Strengthen Relationship"

        return {
            'risk_score': risk_score, 'probability': risk_score / 100.0,
            'status': status, 'color': color,
            'impact_title': title, 'impact_text': impact,
            'drivers': drivers, 'intervention': self.get_intervention(status, risk_score),
            'model_type': model_type
        }


# ==========================================
# 2. KAFKA STREAM SIMULATOR
# ==========================================

# Realistic transaction types with weights (higher weight = more common)
TRANSACTION_TYPES = [
    ("UPI_TRANSFER",       0.28),
    ("LENDING_APP_UPI",    0.08),
    ("ATM_WITHDRAWAL",     0.10),
    ("SALARY_CREDIT",      0.05),
    ("UTILITY_PAYMENT",    0.10),
    ("DINING_DEBIT",       0.12),
    ("EMI_DEBIT",          0.07),
    ("FAILED_AUTODEBIT",   0.04),
    ("SAVINGS_WITHDRAWAL", 0.06),
    ("GAMBLING_UPI",       0.03),
    ("BALANCE_ENQUIRY",    0.07),
]

CUSTOMER_POOL = [
    # (cust_id, base_risk_level)  risk_level drives how "bad" the generated data is
    ("CUST_000142", "high"),
    ("CUST_000289", "medium"),
    ("CUST_000531", "low"),
    ("CUST_000873", "critical"),
    ("CUST_001024", "medium"),
    ("CUST_001337", "low"),
    ("CUST_001456", "high"),
    ("CUST_001789", "critical"),
]

AMOUNT_RANGES = {
    "UPI_TRANSFER":       (500,   15000),
    "LENDING_APP_UPI":    (2000,  50000),
    "ATM_WITHDRAWAL":     (1000,  10000),
    "SALARY_CREDIT":      (25000, 120000),
    "UTILITY_PAYMENT":    (500,   5000),
    "DINING_DEBIT":       (200,   3000),
    "EMI_DEBIT":          (5000,  35000),
    "FAILED_AUTODEBIT":   (0,     0),
    "SAVINGS_WITHDRAWAL": (5000,  80000),
    "GAMBLING_UPI":       (100,   5000),
    "BALANCE_ENQUIRY":    (0,     0),
}

RISK_SIGNALS = {
    "critical": {"LENDING_APP_UPI": 0.30, "GAMBLING_UPI": 0.15, "FAILED_AUTODEBIT": 0.15, "SAVINGS_WITHDRAWAL": 0.20},
    "high":     {"LENDING_APP_UPI": 0.18, "ATM_WITHDRAWAL": 0.18, "SAVINGS_WITHDRAWAL": 0.15},
    "medium":   {"ATM_WITHDRAWAL": 0.12, "UTILITY_PAYMENT": 0.12},
    "low":      {"DINING_DEBIT": 0.20, "UPI_TRANSFER": 0.30},
}

def generate_kafka_event(risk_level: str) -> dict:
    """Generates one realistic transaction event for a given risk profile."""
    cust_id, _ = random.choice([c for c in CUSTOMER_POOL if c[1] == risk_level] or CUSTOMER_POOL)

    # Bias transaction type toward risk signals for this profile
    biases = RISK_SIGNALS.get(risk_level, {})
    tx_types  = [t[0] for t in TRANSACTION_TYPES]
    tx_weights = []
    for t in TRANSACTION_TYPES:
        w = t[1]
        if t[0] in biases:
            w = biases[t[0]]
        tx_weights.append(w)
    # Normalise
    total = sum(tx_weights)
    tx_weights = [w / total for w in tx_weights]

    tx_type = random.choices(tx_types, weights=tx_weights, k=1)[0]
    lo, hi  = AMOUNT_RANGES[tx_type]
    amount  = round(random.uniform(lo, hi), 2) if hi > 0 else 0

    # Timestamp: between 0-120 seconds ago (simulates near real-time lag)
    ts = datetime.now() - timedelta(seconds=random.randint(0, 120))

    # Derive a quick risk score from the event type alone (displayed in stream)
    event_risk_map = {
        "LENDING_APP_UPI":    random.randint(72, 95),
        "FAILED_AUTODEBIT":   random.randint(65, 88),
        "GAMBLING_UPI":       random.randint(70, 92),
        "SAVINGS_WITHDRAWAL": random.randint(55, 80),
        "ATM_WITHDRAWAL":     random.randint(40, 65),
        "BALANCE_ENQUIRY":    random.randint(35, 55),
        "UTILITY_PAYMENT":    random.randint(20, 45),
        "EMI_DEBIT":          random.randint(10, 35),
        "SALARY_CREDIT":      random.randint(5,  20),
        "DINING_DEBIT":       random.randint(5,  25),
        "UPI_TRANSFER":       random.randint(10, 40),
    }
    risk_score = event_risk_map.get(tx_type, random.randint(10, 50))

    partition = hash(cust_id) % 4   # simulate 4 Kafka partitions
    offset    = random.randint(100000, 999999)

    return {
        "kafka_meta": {
            "topic":     "bank.transactions.realtime",
            "partition": partition,
            "offset":    offset,
            "timestamp": ts.isoformat(timespec="milliseconds"),
        },
        "payload": {
            "customer_id":    cust_id,
            "transaction_id": f"TXN{random.randint(10**9, 10**10 - 1)}",
            "type":           tx_type,
            "amount":         amount,
            "channel":        random.choice(["UPI", "NEFT", "IMPS", "ATM", "POS"]),
            "merchant":       _merchant_for(tx_type),
            "risk_score":     risk_score,
            "risk_label":     "CRITICAL" if risk_score > 75 else ("WATCHLIST" if risk_score > 40 else "HEALTHY"),
            "intervention":   _intervention_for(risk_score),
            "processing_ms":  random.randint(8, 45),
        }
    }

def _merchant_for(tx_type: str) -> str:
    merchants = {
        "LENDING_APP_UPI":    random.choice(["KreditBee", "MoneyTap", "CASHe", "PaySense", "Nira"]),
        "GAMBLING_UPI":       random.choice(["Dream11", "MPL", "LotteryKing", "RummyCircle"]),
        "DINING_DEBIT":       random.choice(["Swiggy", "Zomato", "Dominos", "McDonald's"]),
        "ATM_WITHDRAWAL":     "ATM",
        "SALARY_CREDIT":      "Employer",
        "UTILITY_PAYMENT":    random.choice(["BESCOM", "Jio", "Airtel", "BWSSB"]),
        "SAVINGS_WITHDRAWAL": "Self Transfer",
        "FAILED_AUTODEBIT":   random.choice(["HDFC EMI", "ICICI Loan", "Bajaj Finserv"]),
        "UPI_TRANSFER":       random.choice(["PhonePe", "GPay", "Paytm", "BHIM"]),
        "EMI_DEBIT":          random.choice(["Bajaj Finserv", "HDFC Bank", "ICICI Bank"]),
        "BALANCE_ENQUIRY":    "Self",
    }
    return merchants.get(tx_type, "Unknown")

def _intervention_for(risk_score: int) -> str:
    if risk_score > 75: return "📞 Immediate Phone Call"
    if risk_score > 40: return "📧 Email + SMS Nudge"
    return "✅ No Action Required"


def render_event_card(event: dict) -> str:
    """Returns HTML for one event card."""
    p   = event["payload"]
    km  = event["kafka_meta"]
    lbl = p["risk_label"]
    css_class = lbl.lower()
    badge_color = "red" if lbl == "CRITICAL" else ("orange" if lbl == "WATCHLIST" else "")
    badge_html = f'<span class="kafka-badge {badge_color}">{lbl}</span>'

    amt_str = f"₹{p['amount']:,.0f}" if p['amount'] > 0 else "—"
    ts_short = km["timestamp"][11:19]  # HH:MM:SS

    return f"""
    <div class="event-card {css_class}">
        {badge_html}
        <strong>{p['customer_id']}</strong> &nbsp;·&nbsp;
        <code>{p['type']}</code> &nbsp;·&nbsp; {amt_str} &nbsp;·&nbsp;
        <em>{p['merchant']}</em>
        &nbsp;&nbsp;
        <span style="color:#888; float:right">
            P{km['partition']} · offset {km['offset']} · {ts_short} · {p['processing_ms']}ms
        </span><br/>
        <span style="color:#aaa; font-size:0.78rem;">
            Risk: <strong style="color:{'#ff4444' if lbl=='CRITICAL' else '#ffaa00' if lbl=='WATCHLIST' else '#00ff88'}">{p['risk_score']}/100</strong>
            &nbsp;→&nbsp; {p['intervention']}
            &nbsp;·&nbsp; txn_id: {p['transaction_id']}
        </span>
    </div>
    """


# ==========================================
# 3. SESSION STATE INIT
# ==========================================
if "stream_running" not in st.session_state:
    st.session_state.stream_running = False
if "event_log" not in st.session_state:
    st.session_state.event_log = deque(maxlen=50)   # keep last 50 events
if "stream_stats" not in st.session_state:
    st.session_state.stream_stats = {"total": 0, "critical": 0, "watchlist": 0, "healthy": 0}
if "throughput_history" not in st.session_state:
    st.session_state.throughput_history = deque(maxlen=30)  # 30 ticks for chart


# ==========================================
# 4. LOAD MODEL
# ==========================================
with st.spinner(" Loading AI Model..."):
    engine = RiskEngine()


# ==========================================
# 5. TABS
# ==========================================
tab1, tab2 = st.tabs([" Risk Assessment", " Live Kafka Stream"])


# ==========================================
# TAB 1: ORIGINAL DASHBOARD (unchanged)
# ==========================================
with tab1:
    st.title(" Sanjeevani: Pre-Delinquency Intervention Platform")
    st.markdown("### Predict. Explain. Act. Save ₹245 Crores.")

    customer_id = st.text_input(" Customer ID", value="CUST_000142")
    st.markdown("---")

    with st.expander(" Customer Financial Profile (T-1 Month Data)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("####  Income & Stability")
            s_salary  = st.slider("Salary Delay (Days)",    0, 30,   0)
            s_savings = st.slider("Savings Drawdown (%)",   0, 100, 10)
            s_dti     = st.slider("DTI Ratio",              0.0, 1.0, 0.3, 0.01)
            s_util    = st.slider("Utility Late Days",      0, 30,   0)
            s_liq     = st.slider("Liquidity Pressure",     0.0, 10.0, 2.0)
        with c2:
            st.markdown("####  Spending Behavior")
            s_apps   = st.slider("Lending Apps",            0, 10,   0)
            s_cc     = st.slider("Credit Velocity",         0, 100, 20)
            s_atm    = st.slider("ATM Withdrawals",         0, 20,   2)
            s_dining = st.slider("Dining Frequency",        0, 40,  25)
            s_failed = st.slider("Failed Transactions",     0, 10,   0)
        with c3:
            st.markdown("####  Risk Signals")
            s_gamble = st.slider("Gambling Count",          0, 10,   0)
            s_upi    = st.slider("UPI Spike Ratio",         0.0, 5.0, 1.0)
            s_bal    = st.slider("Balance Checks",          0, 20,   2)
            s_stress = st.slider("Behavioral Stress",       0.0, 10.0, 1.0)
            s_vol    = st.slider("Volatility Risk",         0.0, 5.0, 0.5)

    inputs = {
        'salary_delay': s_salary,    'savings_drawdown': s_savings, 'dti_ratio': s_dti,
        'utility_late_days': s_util, 'liquidity_pressure': s_liq,
        'lending_apps': s_apps,      'cc_velocity': s_cc,           'atm_count': s_atm,
        'dining_count': s_dining,    'failed_tx': s_failed,
        'gambling_count': s_gamble,  'upi_spike_ratio': s_upi,      'balance_checks': s_bal,
        'behavioral_stress': s_stress, 'volatility_risk': s_vol
    }

    data = engine.get_assessment(inputs)
    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk Score",       f"{data['risk_score']:.0f}/100",          delta=data['status'],       delta_color="inverse" if data['status'] != 'HEALTHY' else "normal")
    m2.metric("Probability",      f"{data['probability']:.1%}",             delta=data['model_type'])
    m3.metric("Capital at Risk",  f"₹{(data['probability'] * 500000):,.0f}", delta="Exposure",           delta_color="inverse")
    m4.metric("Intervention ROI", f"{data['intervention']['roi']}x",         delta="Projected Return")

    st.markdown("---")
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.subheader(" Risk Severity")
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

        st.subheader(" Risk Drivers")
        if "SHAP" in data['model_type']: st.caption("⚡ Powered by SHAP (Game Theoretic Feature Importance)")
        drivers = data['drivers']
        fig_drivers = go.Figure()
        fig_drivers.add_trace(go.Bar(
            y=[d['feature'] for d in drivers], x=[d['impact'] for d in drivers],
            orientation='h',
            marker=dict(color=[d['impact'] for d in drivers], colorscale='Reds', showscale=False),
            text=[f"{d['value']:.1f}" for d in drivers], textposition='outside'
        ))
        fig_drivers.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"},
                                  showlegend=False, xaxis_title="Impact on Risk Score")
        st.plotly_chart(fig_drivers, use_container_width=True)

    with col_right:
        st.subheader(" Business Value")
        if data['status'] == 'CRITICAL':   st.error(f"**{data['impact_title']}**")
        elif data['status'] == 'WATCHLIST': st.warning(f"**{data['impact_title']}**")
        else:                              st.success(f"**{data['impact_title']}**")
        st.markdown(data['impact_text'])
        st.markdown("---")

        st.subheader(" Intervention")
        inv = data['intervention']
        st.info(f"**Action:** {inv['action']}\n\n**Success Probability:** {inv['success_rate']:.0%}")
        c_a, c_b = st.columns(2)
        c_a.metric("Cost",    f"₹{inv['cost']}")
        c_b.metric("Savings", f"₹{inv['expected_savings']:,}")
        if st.button("Initiate Protocol", type="primary", use_container_width=True):
            st.toast(f" Protocol Sent: {inv['action']} for {customer_id}")

    st.markdown("---")
    st.subheader("Portfolio Impact Simulation")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            p_size      = st.number_input("Portfolio Size (Customers)",      value=100000, step=10000)
            delinq_rate = st.slider("Current Delinquency Rate (%)",           0.0, 20.0, 4.2)
        with col2:
            coverage = st.slider("Intervention Coverage (%)",                 0, 100, 80)
            exposure = st.number_input("Avg Exposure per Customer (₹)",       value=500000, step=50000)

        at_risk_count   = p_size * (delinq_rate / 100.0)
        intervened_count = at_risk_count * (coverage / 100.0)
        prevented       = intervened_count * 0.45
        cost_total      = intervened_count * 150
        savings_total   = prevented * exposure * 0.20
        net_savings     = savings_total - cost_total
        roi_calc        = net_savings / cost_total if cost_total > 0 else 0

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Defaults Prevented", f"{int(prevented):,}",              help="Customers saved from default")
        r2.metric("Program Cost",       f"₹{cost_total/1e7:.2f} Cr")
        r3.metric("Net Savings",        f"₹{net_savings/1e7:.2f} Cr",      delta="Annual Impact")
        r4.metric("Program ROI",        f"{roi_calc:.1f}x")

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(name='Without AI', x=['Defaults'], y=[at_risk_count],            marker_color='#E74C3C'))
        fig_sim.add_trace(go.Bar(name='With AI',    x=['Defaults'], y=[at_risk_count - prevented], marker_color='#2ECC71'))
        fig_sim.update_layout(barmode='group', height=300,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={'color': "white"}, title="Projected Default Reduction")
        st.plotly_chart(fig_sim, use_container_width=True)

    st.markdown("---")
    st.caption("Sanjeevani AI v2.1 | Architecture: Streamlit + XGBoost + SHAP | Compliance: FCA Consumer Duty")


# ==========================================
# TAB 2: KAFKA STREAM SIMULATOR
# ==========================================
with tab2:
    st.title("Live Kafka Transaction Stream")
    st.markdown(
        "Simulates **Apache Kafka** ingesting real-time bank transactions. "
        "Each event is scored by the RiskPulse AI engine (<50ms latency) "
        "and routed to the correct intervention queue — exactly as it would "
        "run on **Amazon Kinesis** in production."
    )

    # ---- Architecture callout ----
    st.markdown("""
    <div class="kafka-header">
        <span style="color:#00ff88; font-weight:bold; font-family:monospace;">
        ▶ TOPIC: bank.transactions.realtime &nbsp;|&nbsp; 
        PARTITIONS: 4 &nbsp;|&nbsp; 
        REPLICATION FACTOR: 3 &nbsp;|&nbsp;
        CONSUMER GROUP: riskpulse-scoring-v2
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ---- Partition boxes ----
    p0, p1, p2, p3 = st.columns(4)
    for col, i in zip([p0, p1, p2, p3], range(4)):
        col.markdown(f"""
        <div class="partition-box">
            <div style="color:#00ff88">PARTITION {i}</div>
            <div style="color:#888; font-size:0.7rem">Consumer: riskpulse-worker-{i}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ---- Controls ----
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
        col_start, col_stop = st.columns(2)
        with col_start:
            start_btn = st.button("▶ Start", type="primary",  use_container_width=True)
        with col_stop:
            stop_btn  = st.button("⏹ Stop",  type="secondary", use_container_width=True)

    if start_btn:
        st.session_state.stream_running = True
    if stop_btn:
        st.session_state.stream_running = False

    if st.button("🗑️ Clear Log", use_container_width=False):
        st.session_state.event_log.clear()
        st.session_state.stream_stats = {"total": 0, "critical": 0, "watchlist": 0, "healthy": 0}
        st.session_state.throughput_history.clear()

    # ---- Live metrics row ----
    st.markdown("---")
    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
    stats_placeholder     = sm1.empty()
    critical_placeholder  = sm2.empty()
    watchlist_placeholder = sm3.empty()
    healthy_placeholder   = sm4.empty()
    status_placeholder    = sm5.empty()

    # ---- Throughput sparkline ----
    throughput_chart = st.empty()

    # ---- JSON inspector toggle ----
    show_json = st.checkbox("🔬 Show raw JSON payload", value=False)

    # ---- Event feed ----
    st.markdown("####  Event Feed")
    feed_placeholder = st.empty()

    # ---- Stream loop ----
    if st.session_state.stream_running:
        # Map risk_filter to profile weights
        profile_weights = {
            "All Profiles":      {"critical": 0.20, "high": 0.30, "medium": 0.30, "low": 0.20},
            "High Risk Heavy":   {"critical": 0.35, "high": 0.40, "medium": 0.20, "low": 0.05},
            "Critical Only":     {"critical": 0.70, "high": 0.25, "medium": 0.05, "low": 0.00},
            "Mostly Healthy":    {"critical": 0.05, "high": 0.10, "medium": 0.25, "low": 0.60},
        }[risk_filter]

        profiles = list(profile_weights.keys())
        weights  = list(profile_weights.values())

        for tick in range(200):   # max 200 ticks before auto-stop to avoid runaway
            if not st.session_state.stream_running:
                break

            # Generate a batch of events
            tick_critical = 0
            for _ in range(batch_size):
                chosen_profile = random.choices(profiles, weights=weights, k=1)[0]
                event = generate_kafka_event(chosen_profile)
                st.session_state.event_log.appendleft(event)   # newest first

                lbl = event["payload"]["risk_label"]
                st.session_state.stream_stats["total"]    += 1
                st.session_state.stream_stats[lbl.lower()] += 1
                if lbl == "CRITICAL":
                    tick_critical += 1

            st.session_state.throughput_history.append(
                {"tick": tick, "events_per_sec": batch_size / delay, "critical": tick_critical}
            )

            # --- Update live metrics ---
            s = st.session_state.stream_stats
            stats_placeholder.metric(" Total Events",  f"{s['total']:,}")
            critical_placeholder.metric("🔴 Critical",   f"{s['critical']:,}",
                                        delta=f"{s['critical']/max(s['total'],1)*100:.1f}%",
                                        delta_color="inverse")
            watchlist_placeholder.metric("🟠 Watchlist", f"{s['watchlist']:,}",
                                         delta=f"{s['watchlist']/max(s['total'],1)*100:.1f}%",
                                         delta_color="off")
            healthy_placeholder.metric("🟢 Healthy",     f"{s['healthy']:,}",
                                       delta=f"{s['healthy']/max(s['total'],1)*100:.1f}%",
                                       delta_color="normal")
            status_placeholder.markdown(
                "<div style='padding:10px; background:#00ff8822; border:1px solid #00ff88; "
                "border-radius:6px; text-align:center; font-family:monospace; color:#00ff88;'>"
                "● STREAMING LIVE</div>", unsafe_allow_html=True
            )

            # --- Throughput chart ---
            if len(st.session_state.throughput_history) > 1:
                th_df = pd.DataFrame(list(st.session_state.throughput_history))
                fig_th = go.Figure()
                fig_th.add_trace(go.Scatter(
                    x=th_df["tick"], y=th_df["events_per_sec"],
                    mode="lines", fill="tozeroy",
                    line=dict(color="#00ff88", width=2),
                    name="Events/sec"
                ))
                fig_th.add_trace(go.Scatter(
                    x=th_df["tick"], y=th_df["critical"],
                    mode="lines",
                    line=dict(color="#ff4444", width=1.5, dash="dot"),
                    name="Critical/tick"
                ))
                fig_th.update_layout(
                    height=130, margin=dict(t=10, b=10, l=40, r=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white", size=10),
                    showlegend=True,
                    legend=dict(orientation="h", y=1.2, font=dict(size=9)),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor="#222", zeroline=False),
                )
                throughput_chart.plotly_chart(fig_th, use_container_width=True)

            # --- Event feed ---
            cards_html = ""
            for ev in list(st.session_state.event_log)[:20]:   # show latest 20
                cards_html += render_event_card(ev)
                if show_json:
                    cards_html += (
                        f"<pre style='background:#111; color:#888; font-size:0.7rem; "
                        f"padding:6px; border-radius:4px; margin-bottom:8px;'>"
                        f"{json.dumps(ev, indent=2)}</pre>"
                    )
            feed_placeholder.markdown(cards_html, unsafe_allow_html=True)

            time.sleep(delay)

        # Auto-stop after 200 ticks
        st.session_state.stream_running = False
        status_placeholder.markdown(
            "<div style='padding:10px; background:#33333388; border:1px solid #555; "
            "border-radius:6px; text-align:center; font-family:monospace; color:#888;'>"
            "■ STOPPED</div>", unsafe_allow_html=True
        )

    else:
        # Show last known state when paused
        s = st.session_state.stream_stats
        stats_placeholder.metric(" Total Events",  f"{s['total']:,}")
        critical_placeholder.metric("🔴 Critical",   f"{s['critical']:,}")
        watchlist_placeholder.metric("🟠 Watchlist", f"{s['watchlist']:,}")
        healthy_placeholder.metric("🟢 Healthy",     f"{s['healthy']:,}")
        status_placeholder.markdown(
            "<div style='padding:10px; background:#33333388; border:1px solid #555; "
            "border-radius:6px; text-align:center; font-family:monospace; color:#888;'>"
            "■ STOPPED</div>", unsafe_allow_html=True
        )

        if st.session_state.event_log:
            cards_html = ""
            for ev in list(st.session_state.event_log)[:20]:
                cards_html += render_event_card(ev)
                if show_json:
                    cards_html += (
                        f"<pre style='background:#111; color:#888; font-size:0.7rem; "
                        f"padding:6px; border-radius:4px; margin-bottom:8px;'>"
                        f"{json.dumps(ev, indent=2)}</pre>"
                    )
            feed_placeholder.markdown(cards_html, unsafe_allow_html=True)
        else:
            feed_placeholder.info("▶ Press **Start** to begin the live event stream.")

    st.markdown("---")
    st.caption("Simulated Kafka stream · Topic: bank.transactions.realtime · "
               "Partitions: 4 · Consumer Group: riskpulse-scoring-v2 · "
               "Production equivalent: Amazon Kinesis Data Streams")
