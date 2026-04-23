"""
Virtual ICU - Streamlit Dashboard
Real-time patient monitoring with AI-driven early warning system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './')))

from synthetic_data_generator.clinical_scorer import NEWS2Calculator, qSOFACalculator, CARTCalculator, ClinicalRecommendations

# Page configuration
st.set_page_config(
    page_title="Virtual ICU - AI Patient Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .alert-high {
        background-color: #ff4444;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-medium {
        background-color: #ffaa00;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-low {
        background-color: #44aa44;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'current_row' not in st.session_state:
    st.session_state.current_row = 0

# Sidebar
st.sidebar.title("🏥 Virtual ICU Monitor")
st.sidebar.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dashboard", "📈 Vital Signs", "🔢 Clinical Scores", "🎮 Invigilator Panel"]
)

# ============ TAB 1: DASHBOARD ============
with tab1:
    st.title("Virtual ICU - Real-Time Patient Monitoring")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📁 Upload Patient Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(st.session_state.data)} patient records")
    
    with col2:
        if st.session_state.data is not None:
            st.metric("Total Records", len(st.session_state.data))
            st.metric("Columns", len(st.session_state.data.columns))
    
    st.markdown("---")
    
    # Display data info
    if st.session_state.data is not None:
        st.subheader("📋 Data Preview")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Heart Rate (bpm)",
                f"{st.session_state.data['heart_rate'].mean():.0f}",
                f"Range: {st.session_state.data['heart_rate'].min():.0f}-{st.session_state.data['heart_rate'].max():.0f}"
            )
        
        with col2:
            st.metric(
                "SpO2 (%)",
                f"{st.session_state.data['spo2'].mean():.1f}",
                f"Min: {st.session_state.data['spo2'].min():.1f}%"
            )
        
        with col3:
            st.metric(
                "Systolic BP (mmHg)",
                f"{st.session_state.data['systolic_bp'].mean():.0f}",
                f"Range: {st.session_state.data['systolic_bp'].min():.0f}-{st.session_state.data['systolic_bp'].max():.0f}"
            )

# ============ TAB 2: VITAL SIGNS ============
with tab2:
    st.title("📈 Vital Signs Monitoring")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload patient data first")
    else:
        # Prepare data
        data = st.session_state.data.copy()
        data['index'] = range(len(data))
        
        # Create figure with subplots
        fig = go.Figure()
        
        # Heart Rate
        fig.add_trace(go.Scatter(
            x=data['index'],
            y=data['heart_rate'],
            name='Heart Rate (bpm)',
            line=dict(color='red', width=2),
            yaxis='y1'
        ))
        
        # SpO2
        fig.add_trace(go.Scatter(
            x=data['index'],
            y=data['spo2'],
            name='SpO2 (%)',
            line=dict(color='blue', width=2),
            yaxis='y2'
        ))
        
        # Systolic BP
        fig.add_trace(go.Scatter(
            x=data['index'],
            y=data['systolic_bp'],
            name='Systolic BP (mmHg)',
            line=dict(color='green', width=2),
            yaxis='y3'
        ))
        
        # Respiratory Rate
        fig.add_trace(go.Scatter(
            x=data['index'],
            y=data['respiratory_rate'],
            name='RR (breaths/min)',
            line=dict(color='orange', width=2),
            yaxis='y4'
        ))
        
        # Update layout with multiple y-axes
        fig.update_layout(
            title="Multi-Parameter Vital Signs Trend",
            xaxis=dict(title="Time (samples)"),
            yaxis=dict(title="HR (bpm)", side="left"),
            yaxis2=dict(title="SpO2 (%)", overlaying="y", side="right"),
            yaxis3=dict(title="SBP (mmHg)", overlaying="y", side="left", anchor="free", position=0.0),
            yaxis4=dict(title="RR (breaths/min)", overlaying="y", side="right", anchor="free", position=1.0),
            hovermode='x unified',
            height=600,
            width=1200
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual vital signs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("❤️ Heart Rate Trend")
            fig_hr = px.line(data, x='index', y='heart_rate', title='Heart Rate Over Time')
            fig_hr.add_hline(y=60, line_dash="dash", line_color="green", annotation_text="Normal: 60-100")
            fig_hr.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Tachycardia: >100")
            st.plotly_chart(fig_hr, use_container_width=True)
        
        with col2:
            st.subheader("🫁 Oxygen Saturation Trend")
            fig_spo2 = px.line(data, x='index', y='spo2', title='SpO2 Over Time')
            fig_spo2.add_hline(y=95, line_dash="dash", line_color="green", annotation_text="Normal: >95%")
            fig_spo2.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="Critical: <90%")
            st.plotly_chart(fig_spo2, use_container_width=True)

# ============ TAB 3: CLINICAL SCORES ============
with tab3:
    st.title("🔢 Clinical Risk Scores")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload patient data first")
    else:
        data = st.session_state.data.copy()
        
        # Select record to analyze
        st.subheader("Select Record to Score")
        record_idx = st.slider("Record number", 0, len(data) - 1, 0)
        
        vitals = data.iloc[record_idx].to_dict()
        vitals['alert_status'] = 'Alert'
        vitals['supplemental_oxygen'] = False
        vitals['age'] = 65
        
        # Calculate scores
        news2 = NEWS2Calculator.calculate(vitals)
        qsofa = qSOFACalculator.calculate(vitals)
        cart = CARTCalculator.calculate(vitals)
        
        # Display current vitals
        st.subheader("Current Vital Signs")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("HR", f"{vitals['heart_rate']:.0f} bpm")
        with col2:
            st.metric("SpO2", f"{vitals['spo2']:.1f}%")
        with col3:
            st.metric("SBP", f"{vitals['systolic_bp']:.0f} mmHg")
        with col4:
            st.metric("RR", f"{vitals['respiratory_rate']:.0f} /min")
        
        st.markdown("---")
        
        # NEWS2 Score
        st.subheader("📊 NEWS2 Score")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Score", f"{news2['total']}/{news2['max_possible']}")
        with col2:
            risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            st.metric("Risk Level", f"{risk_color.get(news2['risk_level'], '❓')} {news2['risk_level']}")
        with col3:
            st.write("**Components:**")
            for component, value in news2['components'].items():
                st.write(f"  • {component}: {value}")
        
        st.markdown("---")
        
        # qSOFA Score
        st.subheader("🦠 qSOFA Score (Sepsis Screening)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Score", f"{qsofa['total']}/{qsofa['max_possible']}")
        with col2:
            st.metric("Sepsis Risk", qsofa['risk_level'])
        with col3:
            st.write("**Components:**")
            for component, value in qsofa['components'].items():
                status = "✅" if value == 0 else "⚠️"
                st.write(f"  {status} {component}: {value}")
        
        st.markdown("---")
        
        # CART Score
        st.subheader("⚡ CART Score (Cardiac Arrest Risk)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Score", f"{cart['total']}/{cart['max_possible']}")
        with col2:
            risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "Highest": "🔴🔴"}
            st.metric("Risk Category", f"{risk_colors.get(cart['risk_category'], '❓')} {cart['risk_category']}")
        with col3:
            st.metric("Risk Percentile", f"{cart['percentile']:.0f}%")
        
        st.markdown("---")
        
        # Recommendations
        st.subheader("💊 Clinical Recommendations")
        recommendations = ClinicalRecommendations.get_recommendations(news2, qsofa, cart)
        
        urgency_colors = {"Low": "alert-low", "Medium": "alert-medium", "High": "alert-high"}
        
        st.markdown(f'<div class="alert-{recommendations["urgency"].lower()}"><strong>Urgency: {recommendations["urgency"]}</strong></div>', unsafe_allow_html=True)
        
        st.write("**Recommended Actions:**")
        for rec in recommendations['recommendations']:
            st.write(f"  • {rec}")

# ============ TAB 4: INVIGILATOR PANEL ============
with tab4:
    st.title("🎮 Invigilator Control Panel")
    st.info("🔬 Educational demonstration mode - Manipulate vital signs to observe system response")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload patient data first")
    else:
        st.subheader("Real-Time Parameter Adjustment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hr_override = st.slider("Heart Rate (bpm)", 40, 200, 80)
            spo2_override = st.slider("SpO2 (%)", 60, 100, 95)
            sbp_override = st.slider("Systolic BP (mmHg)", 30, 200, 120)
        
        with col2:
            rr_override = st.slider("Respiratory Rate (/min)", 8, 40, 16)
            temp_override = st.slider("Temperature (°C)", 35.0, 41.0, 37.0, 0.1)
            alert_status = st.selectbox("Alert Status", ["Alert", "Confused", "Lethargic", "Unresponsive"])
        
        # Create overridden vitals
        demo_vitals = {
            'heart_rate': hr_override,
            'spo2': spo2_override,
            'systolic_bp': sbp_override,
            'respiratory_rate': rr_override,
            'temperature': temp_override,
            'alert_status': alert_status,
            'supplemental_oxygen': st.checkbox("Supplemental Oxygen"),
            'age': 65
        }
        
        # Calculate scores with overridden values
        news2_demo = NEWS2Calculator.calculate(demo_vitals)
        qsofa_demo = qSOFACalculator.calculate(demo_vitals)
        cart_demo = CARTCalculator.calculate(demo_vitals)
        
        st.markdown("---")
        st.subheader("📊 Real-Time Score Response")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("NEWS2", f"{news2_demo['total']}/20")
            st.write(f"Risk: {news2_demo['risk_level']}")
        
        with col2:
            st.metric("qSOFA", f"{qsofa_demo['total']}/3")
            st.write(f"Sepsis: {'⚠️ YES' if qsofa_demo['needs_investigation'] else '✅ NO'}")
        
        with col3:
            st.metric("CART", f"{cart_demo['total']}/20")
            st.write(f"Risk: {cart_demo['risk_category']}")
        
        # Dynamic recommendations
        st.markdown("---")
        st.subheader("🚨 Dynamic Recommendations")
        
        recommendations_demo = ClinicalRecommendations.get_recommendations(news2_demo, qsofa_demo, cart_demo)
        
        urgency_colors = {"Low": "alert-low", "Medium": "alert-medium", "High": "alert-high"}
        st.markdown(f'<div class="alert-{recommendations_demo["urgency"].lower()}"><strong>Urgency Level: {recommendations_demo["urgency"]}</strong></div>', unsafe_allow_html=True)
        
        st.write("**System Recommendations:**")
        for rec in recommendations_demo['recommendations']:
            st.write(f"  • {rec}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    Virtual ICU - AI-Driven Real-Time Early Warning System<br>
    Phase 3: Streamlit Dashboard | Educational & Research Use Only
</div>
""", unsafe_allow_html=True)
