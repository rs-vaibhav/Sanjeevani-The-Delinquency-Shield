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
    {"id": "CUST_101", "name": "Rahul Sharma", "base_risk": 35},
    {"id": "CUST_102", "name": "Priya Patel", "base_risk": 68},
    {"id": "CUST_103", "name": "Amit Kumar", "base_risk": 22},
    {"id": "CUST_104", "name": "Sneha Reddy", "base_risk": 82},
    {"id": "CUST_105", "name": "Vikram Singh", "base_risk": 45},
]

if 'event_log' not in st.session_state:
    st.session_state.event_log = deque(maxlen=100)
if 'stream_running' not in st.session_state:
    st.session_state.stream_running = False
if 'tick_count' not in st.session_state:
    st.session_state.tick_count = 0
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'total_processed': 0, 'critical_count': 0, 'watchlist_count': 0,
        'healthy_count': 0, 'throughput_history': deque([0]*30, maxlen=30)
    }


def generate_event():
    """Generate one realistic banking event"""
    tx_type, _ = random.choices(TRANSACTION_TYPES, weights=[w for _, w in TRANSACTION_TYPES])[0]
    customer = random.choice(CUSTOMER_POOL)
    
    features = {
        'salary_delay': random.randint(0, 30) if random.random() < 0.3 else 0,
        'savings_drawdown': random.randint(0, 50),
        'dti_ratio': round(random.uniform(0.1, 0.8), 2),
        'utility_late_days': random.randint(0, 10),
        'liquidity_pressure': round(random.uniform(0, 5), 1),
        'lending_apps': random.randint(0, 8) if 'LENDING' in tx_type else random.randint(0, 2),
        'cc_velocity': random.randint(10, 60),
        'atm_count': random.randint(1, 15),
        'dining_count': random.randint(5, 30),
        'failed_tx': random.randint(0, 5) if 'FAILED' in tx_type else 0,
        'gambling_count': random.randint(1, 8) if 'GAMBLING' in tx_type else 0,
        'upi_spike_ratio': round(random.uniform(0.5, 3.0), 1),
        'balance_checks': random.randint(1, 15),
        'behavioral_stress': round(random.uniform(0, 5), 1),
        'volatility_risk': round(random.uniform(0, 3), 1),
    }
    
    # Spike risk for certain transaction types
    if tx_type in ['LENDING_APP_UPI', 'GAMBLING_UPI', 'FAILED_AUTODEBIT']:
        features['lending_apps'] = min(8, features['lending_apps'] + 3)
        features['dti_ratio'] = min(0.9, features['dti_ratio'] + 0.2)
    
    assessment = engine.get_assessment(features)
    
    return {
        'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3],
        'customer_id': customer['id'],
        'customer_name': customer['name'],
        'transaction_type': tx_type,
        'risk_score': assessment['risk_score'],
        'status': assessment['status'],
        'features': features,
        'partition': random.randint(0, 3)
    }


def render_event_card(event, show_json=False):
    """Render a single event as HTML"""
    status_class = event['status'].lower()
    badge_class = 'red' if status_class == 'critical' else ('orange' if status_class == 'watchlist' else '')
    
    json_block = ""
    if show_json:
        json_block = f"<pre style='margin-top:8px;font-size:0.7rem;'>{json.dumps(event['features'], indent=2)}</pre>"
    
    return f"""
    <div class="event-card {status_class}">
        <span class="kafka-badge {badge_class}">{event['status']}</span>
        <strong>{event['customer_id']}</strong> | {event['customer_name']} | {event['transaction_type']}<br>
        <span style="color:#888;">Risk: {event['risk_score']:.0f}/100 | {event['timestamp']} | Partition-{event['partition']}</span>
        {json_block}
    </div>
    """


# ==========================================
# 3. KAFKA TAB (Fragment)
# ==========================================
@st.fragment
def kafka_tab():
    st.title("📡 Live Kafka Event Stream")
    st.markdown("Real-time transaction scoring from **bank.transactions.realtime**")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("<div class='kafka-header'>🔌 <b>KAFKA CLUSTER:</b> sanjeevani-prod-1.aws.cluster | "
                   "<b>TOPIC:</b> bank.transactions.realtime | <b>PARTITIONS:</b> 4</div>", 
                   unsafe_allow_html=True)
    
    with col2:
        if st.button("▶️ START STREAM" if not st.session_state.stream_running else "⏸️ STOP STREAM",
                    type="primary", use_container_width=True):
            st.session_state.stream_running = not st.session_state.stream_running
            st.rerun()
    
    st.markdown("---")
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("🔴 Live Event Feed")
        show_json = st.checkbox("Show JSON Payloads", value=False)
        feed_ph = st.empty()
    
    with c2:
        st.subheader("📊 Stream Metrics")
        status_ph = st.empty()
        m1, m2, m3 = st.columns(3)
        critical_ph = m1.empty()
        watch_ph = m2.empty()
        healthy_ph = m3.empty()
        total_ph = st.empty()
        tput_ph = st.empty()
    
    def render_static_metrics():
        m = st.session_state.metrics
        critical_ph.metric("🔴 Critical", m['critical_count'])
        watch_ph.metric("🟠 Watchlist", m['watchlist_count'])
        healthy_ph.metric("🟢 Healthy", m['healthy_count'])
        total_ph.metric("Total Processed", f"{m['total_processed']:,}")
    
    def render_feed():
        feed_ph.markdown(
            "".join(render_event_card(ev, show_json) for ev in list(st.session_state.event_log)[:20]),
            unsafe_allow_html=True
        )
    
    # Main streaming loop
    if st.session_state.stream_running:
        tick_num = st.session_state.tick_count
        st.session_state.tick_count += 1
        delay = 0.8
        
        status_ph.markdown(
            "<div style='padding:10px;background:#00ff8822;border:1px solid #00ff88;"
            "border-radius:6px;text-align:center;font-family:monospace;'>"
            "● STREAMING</div>", unsafe_allow_html=True
        )
        
        # Generate event
        event = generate_event()
        st.session_state.event_log.appendleft(event)
        
        # Update metrics
        m = st.session_state.metrics
        m['total_processed'] += 1
        if event['status'] == 'CRITICAL':
            m['critical_count'] += 1
        elif event['status'] == 'WATCHLIST':
            m['watchlist_count'] += 1
        else:
            m['healthy_count'] += 1
        
        m['throughput_history'].append(m['total_processed'])
        
        # Render metrics
        render_static_metrics()
        
        # Throughput chart
        if tick_num % 5 == 0:
            fig_th = go.Figure()
            fig_th.add_trace(go.Scatter(
                y=list(m['throughput_history']),
                mode='lines',
                line=dict(color='#00ff88', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,255,136,0.1)'
            ))
            fig_th.update_layout(
                height=180,
                margin=dict(t=20, b=20, l=30, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white"},
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=True, gridcolor='#333'),
                title="Throughput"
            )
            tput_ph.plotly_chart(fig_th, use_container_width=True)
        
        # Event feed
        render_feed()
        
        # Auto-stop after 200 ticks
        if tick_num >= 200:
            st.session_state.stream_running = False
        else:
            time.sleep(delay)
            st.rerun(scope="fragment")
    
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
