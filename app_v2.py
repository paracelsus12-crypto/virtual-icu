"""
Virtual ICU v2 - Complete Streamlit Application
Переписана з нуля: models + clinical_scorer_v2 + data_loader
ВСІ 15 сценаріїв ГОТОВІ (без # ... more scenarios ...)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
from typing import Dict, Optional
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════
# IMPORTS - Models and Scorers
# ════════════════════════════════════════════════════════════════════

# Import from v2 modules (будуть в тому ж директорії)
try:
    from clinical_scorer_v2 import (
        NEWS2CalculatorV2, 
        qSOFACalculatorV2, 
        CARTCalculatorV2,
        ClinicalRecommendationsEngineV2
    )
except ImportError:
    st.error("❌ Error: clinical_scorer_v2.py not found")
    st.stop()

# Import advanced features (v2.0)
try:
    from lstm_predictor import LSTMPredictor, DeteriortationDetector
    from pdf_generator import PDFReportGenerator
    from streamlit_integration import StreamlitIntegration
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False
    logger.warning("Advanced features (LSTM, PDF, Integration) not available")

# ════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Virtual ICU v2",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .alert-high {
        background-color: #ff4444;
        color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #cc0000;
    }
    .alert-medium {
        background-color: #ffaa00;
        color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ff8800;
    }
    .alert-low {
        background-color: #44aa44;
        color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #228822;
    }
    .metric-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
    }
    .recommendation-item {
        padding: 10px;
        margin: 8px 0;
        border-left: 4px solid #1f77b4;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ════════════════════════════════════════════════════════════════════

if 'data' not in st.session_state:
    st.session_state.data = None
if 'current_record' not in st.session_state:
    st.session_state.current_record = 0
if 'data_source' not in st.session_state:
    st.session_state.data_source = "None"
if 'last_updated' not in st.session_state:
    st.session_state.last_updated = None
if 'scenario_history' not in st.session_state:
    st.session_state.scenario_history = []

# ════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATORS (simplified for v2)
# ════════════════════════════════════════════════════════════════════

class SyntheticDataGenerator:
    """Генератор синтетичних даних для різних сценаріїв"""
    
    @staticmethod
    def generate_sepsis(duration_hours=6, variant="sepsis") -> pd.DataFrame:
        """Sepsis scenario - Sepsis-3 compliant"""
        samples = (duration_hours * 60) // 5
        data = []
        
        for i in range(samples):
            progress = i / max(1, samples - 1)
            
            # Sepsis progression: Normal → Infection → Sepsis
            if variant == "septic_shock":
                hr = 70 + progress * 100  # 70 → 170 bpm
                sbp = 120 - progress * 60  # 120 → 60 mmHg
                rr = 14 + progress * 20  # 14 → 34
                temp = 37 + progress * 2  # 37 → 39°C
                spo2 = 98 - progress * 8  # 98 → 90%
            else:  # regular sepsis
                hr = 70 + progress * 60  # 70 → 130 bpm
                sbp = 120 - progress * 25  # 120 → 95 mmHg
                rr = 14 + progress * 15  # 14 → 29
                temp = 37 + progress * 1.5  # 37 → 38.5°C
                spo2 = 98 - progress * 5  # 98 → 93%
            
            # Add noise
            data.append({
                'heart_rate': max(40, hr + np.random.normal(0, 2)),
                'systolic_bp': max(30, sbp + np.random.normal(0, 3)),
                'diastolic_bp': max(20, (sbp * 0.6) + np.random.normal(0, 2)),
                'respiratory_rate': max(8, rr + np.random.normal(0, 1)),
                'spo2': np.clip(spo2 + np.random.normal(0, 1), 60, 100),
                'temperature': temp + np.random.normal(0, 0.2),
                'alert_status': 'Alert' if progress < 0.5 else ('Confused' if progress < 0.8 else 'Lethargic'),
                'supplemental_oxygen': True if progress > 0.3 else False,
                'age': 65,
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_cardiac_arrest(variant="with_rosc") -> pd.DataFrame:
        """Cardiac Arrest scenario - 10 minutes"""
        samples = 120  # 2 hours / 5 min intervals
        data = []
        
        for i in range(samples):
            progress = i / (samples - 1)
            
            if progress < 0.1:  # Normal
                hr = 80 + np.random.normal(0, 2)
                sbp = 120 + np.random.normal(0, 3)
                spo2 = 98 + np.random.normal(0, 1)
            elif progress < 0.2:  # Collapse
                hr = 40 + np.random.normal(0, 5)
                sbp = 60 + np.random.normal(0, 5)
                spo2 = 85 + np.random.normal(0, 3)
            elif progress < 0.5:  # Cardiac arrest (VF/VT)
                hr = 0 if progress > 0.25 else np.random.uniform(30, 200)
                sbp = 20 + np.random.normal(0, 5)
                spo2 = 60 + np.random.normal(0, 10)
            elif variant == "with_rosc" and progress > 0.5:  # ROSC
                hr = 60 + (progress - 0.5) * 80  # 60 → 100 bpm
                sbp = 40 + (progress - 0.5) * 160  # 40 → 120 mmHg
                spo2 = 70 + (progress - 0.5) * 60  # 70 → 98%
            else:  # No ROSC
                hr = 0
                sbp = 30 + np.random.normal(0, 5)
                spo2 = 50 + np.random.normal(0, 10)
            
            data.append({
                'heart_rate': max(0, hr),
                'systolic_bp': max(20, sbp),
                'diastolic_bp': max(10, sbp * 0.5),
                'respiratory_rate': max(0, 30 if hr == 0 else 16 + np.random.normal(0, 2)),
                'spo2': np.clip(spo2, 0, 100),
                'temperature': 36.5 + np.random.normal(0, 0.5),
                'alert_status': 'Unresponsive',
                'supplemental_oxygen': True,
                'age': 65,
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_respiratory_failure(variant="type_i") -> pd.DataFrame:
        """Respiratory Failure - Type I (Hypoxemic) or Type II (Hypercapnic)"""
        samples = (6 * 60) // 5  # 6 hours
        data = []
        
        for i in range(samples):
            progress = i / (samples - 1)
            
            if variant == "type_ii":  # Hypercapnic
                spo2 = 96 - progress * 20  # 96 → 76%
                rr = 15 - progress * 5  # 15 → 10 (decreased)
                ph = 7.39 - progress * 0.15  # 7.39 → 7.24 (acidosis)
            else:  # Type I - Hypoxemic
                spo2 = 97 - progress * 25  # 97 → 72%
                rr = 16 + progress * 20  # 16 → 36 (increased)
                ph = 7.40 - progress * 0.10  # 7.40 → 7.30
            
            data.append({
                'heart_rate': max(40, 70 + progress * 60 + np.random.normal(0, 2)),
                'systolic_bp': max(80, 120 - progress * 30 + np.random.normal(0, 3)),
                'diastolic_bp': max(40, 80 - progress * 20 + np.random.normal(0, 3)),
                'respiratory_rate': max(8, rr + np.random.normal(0, 1)),
                'spo2': np.clip(spo2 + np.random.normal(0, 1), 40, 100),
                'temperature': 37 + np.random.normal(0, 0.3),
                'alert_status': 'Alert' if progress < 0.5 else 'Confused',
                'supplemental_oxygen': True if progress > 0.2 else False,
                'age': 65,
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_hypotension(variant="progressive") -> pd.DataFrame:
        """Hypotensive shock - Progressive or Sudden"""
        samples = (6 * 60) // 5
        data = []
        
        for i in range(samples):
            progress = i / (samples - 1)
            
            if variant == "sudden":
                # Sudden drop at 20%
                if progress < 0.2:
                    sbp = 130 + np.random.normal(0, 3)
                    hr = 75 + np.random.normal(0, 2)
                else:
                    sbp = 50 + (progress - 0.2) * 40 + np.random.normal(0, 5)
                    hr = 150 - (progress - 0.2) * 40 + np.random.normal(0, 3)
            else:  # Progressive
                sbp = 114 - progress * 64 + np.random.normal(0, 3)  # 114 → 50
                hr = 79 + progress * 80 + np.random.normal(0, 2)  # 79 → 159
            
            data.append({
                'heart_rate': max(40, hr),
                'systolic_bp': max(30, sbp),
                'diastolic_bp': max(15, sbp * 0.5),
                'respiratory_rate': max(8, 16 + progress * 15 + np.random.normal(0, 1)),
                'spo2': np.clip(98 - progress * 15 + np.random.normal(0, 1), 75, 100),
                'temperature': 37 - progress * 1 + np.random.normal(0, 0.2),
                'alert_status': 'Alert' if progress < 0.5 else 'Confused',
                'supplemental_oxygen': True if progress > 0.3 else False,
                'age': 65,
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_hypoxemia(variant="acute") -> pd.DataFrame:
        """Acute/Gradual Hypoxemia"""
        samples = (4 * 60) // 5
        data = []
        
        for i in range(samples):
            progress = i / (samples - 1)
            
            if variant == "acute":
                spo2 = 98.3 - progress * 28.2 + np.random.normal(0, 1)  # 98.3 → 70.1%
                hr = 75 + progress * 44 + np.random.normal(0, 2)  # 75 → 119
            else:  # gradual
                spo2 = 99 - progress * 28.3 + np.random.normal(0, 1)  # 99 → 70.7%
                hr = 71 + progress * 43 + np.random.normal(0, 2)  # 71 → 114
            
            data.append({
                'heart_rate': max(40, hr),
                'systolic_bp': max(80, 115 - progress * 30 + np.random.normal(0, 3)),
                'diastolic_bp': max(40, 75 - progress * 20 + np.random.normal(0, 3)),
                'respiratory_rate': max(10, 18 + progress * 18 + np.random.normal(0, 1)),
                'spo2': np.clip(spo2, 50, 100),
                'temperature': 37 + np.random.normal(0, 0.3),
                'alert_status': 'Alert' if progress < 0.3 else ('Confused' if progress < 0.7 else 'Lethargic'),
                'supplemental_oxygen': True if progress > 0.2 else False,
                'age': 65,
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_arrhythmia(variant="afib") -> pd.DataFrame:
        """
        Cardiac Arrhythmias: AFib, VT, SVT, Bradycardia
        
        CLINICAL NOTE:
        - AFib/SVT: High HR (compensatory), normal BP/SpO2/RR → NO sepsis indicators
        - VT: Hemodynamically unstable → may show some deterioration
        - Bradycardia: Low HR, relatively stable BP
        """
        samples = (3 * 60) // 5
        data = []
        
        for i in range(samples):
            progress = i / (samples - 1)
            
            if variant == "afib":
                # AFib: Tachycardia is PRIMARY, not secondary to sepsis
                # Keep BP, SpO2, RR NORMAL to avoid false sepsis alert
                hr = 120 + progress * 40 + np.random.uniform(-10, 10)  # 120 → 160
                sbp = 125 - progress * 8 + np.random.normal(0, 3)     # 125 → 117 (stable!)
                rr = 14 + np.random.normal(0, 1)                       # NORMAL (not 25!)
                spo2 = 97 + np.random.normal(0, 1)                     # NORMAL (not 92!)
                
            elif variant == "vt":
                # VT: More unstable, may deteriorate
                hr = 140 + progress * 40 + np.random.uniform(-5, 5)   # 140 → 180
                sbp = 110 - progress * 20 + np.random.normal(0, 3)    # 110 → 90 (deteriorates)
                rr = 16 + progress * 8 + np.random.normal(0, 1)       # 16 → 24 (worsens)
                spo2 = np.clip(98 - progress * 5 + np.random.normal(0, 1), 90, 100)  # Gradual decline
                
            elif variant == "svt":
                # SVT: Similar to AFib but more rapid onset, usually stable
                hr = 160 + progress * 20 + np.random.uniform(-5, 5)   # 160 → 180
                sbp = 120 - progress * 5 + np.random.normal(0, 3)     # 120 → 115 (minimal drop)
                rr = 15 + np.random.normal(0, 1)                      # NORMAL
                spo2 = 97 + np.random.normal(0, 1)                    # NORMAL
                
            else:  # bradycardia
                # Bradycardia: Low HR, usually hemodynamically stable if mild
                hr = 50 - progress * 15 + np.random.uniform(-2, 2)   # 50 → 35
                sbp = 118 - progress * 8 + np.random.normal(0, 2)    # 118 → 110 (relatively stable)
                rr = 14 + np.random.normal(0, 1)                      # NORMAL
                spo2 = 97 + np.random.normal(0, 1)                    # NORMAL
            
            # Temperature: Never fever in simple arrhythmias (unless it's VT with shock)
            if variant == "vt":
                temp = 37 + np.random.normal(0, 0.3)  # May be normal or slightly elevated
            else:
                temp = 37 + np.random.normal(0, 0.1)  # NORMAL temperature
            
            # Alert status worsens if VT (most unstable), minimal change otherwise
            if variant == "vt":
                alert = 'Alert' if progress < 0.6 else 'Confused'
            else:
                alert = 'Alert'  # AFib/SVT/Bradycardia patients usually remain alert
            
            data.append({
                'heart_rate': np.clip(hr, 30, 200),
                'systolic_bp': max(80, sbp),
                'diastolic_bp': max(40, sbp * 0.67),
                'respiratory_rate': np.clip(rr, 6, 40),
                'spo2': np.clip(spo2, 70, 100),
                'temperature': temp,
                'alert_status': alert,
                'supplemental_oxygen': False,
                'age': 65,
            })
        
        return pd.DataFrame(data)

# ════════════════════════════════════════════════════════════════════
# SIDEBAR - DATA LOADING
# ════════════════════════════════════════════════════════════════════

st.sidebar.title("🏥 Virtual ICU Monitor v2")
st.sidebar.markdown("---")

# Demo Scenarios
st.sidebar.subheader("📊 Demo Scenarios (15)")

demo_scenarios = {
    "None": None,
    "Sepsis (Sepsis-3)": ("sepsis", "sepsis"),
    "Septic Shock": ("sepsis", "septic_shock"),
    "Cardiac Arrest with ROSC": ("cardiac_arrest", "with_rosc"),
    "Cardiac Arrest without ROSC": ("cardiac_arrest", "without_rosc"),
    "Respiratory Failure Type I": ("respiratory_failure", "type_i"),
    "Respiratory Failure Type II": ("respiratory_failure", "type_ii"),
    "Hypotension Progressive": ("hypotension", "progressive"),
    "Hypotension Sudden": ("hypotension", "sudden"),
    "Hypoxemia Acute": ("hypoxemia", "acute"),
    "Hypoxemia Gradual": ("hypoxemia", "gradual"),
    "Arrhythmia: AFib": ("arrhythmia", "afib"),
    "Arrhythmia: Ventricular Tachycardia": ("arrhythmia", "vt"),
    "Arrhythmia: Supraventricular Tachycardia": ("arrhythmia", "svt"),
    "Arrhythmia: Bradycardia": ("arrhythmia", "bradycardia"),
}

demo_option = st.sidebar.selectbox("Select Scenario:", list(demo_scenarios.keys()))

if st.sidebar.button("🔄 Generate Demo Data", key="generate_demo"):
    if demo_option != "None":
        with st.sidebar.status("Generating...", expanded=True) as status:
            try:
                gen_type, variant = demo_scenarios[demo_option]
                
                if gen_type == "sepsis":
                    st.session_state.data = SyntheticDataGenerator.generate_sepsis(variant=variant)
                elif gen_type == "cardiac_arrest":
                    st.session_state.data = SyntheticDataGenerator.generate_cardiac_arrest(variant=variant)
                elif gen_type == "respiratory_failure":
                    st.session_state.data = SyntheticDataGenerator.generate_respiratory_failure(variant=variant)
                elif gen_type == "hypotension":
                    st.session_state.data = SyntheticDataGenerator.generate_hypotension(variant=variant)
                elif gen_type == "hypoxemia":
                    st.session_state.data = SyntheticDataGenerator.generate_hypoxemia(variant=variant)
                elif gen_type == "arrhythmia":
                    st.session_state.data = SyntheticDataGenerator.generate_arrhythmia(variant=variant)
                
                st.session_state.data_source = demo_option
                st.session_state.current_record = 0
                st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.scenario_history.append(demo_option)
                
                status.update(label="✅ Data generated!", state="complete")
                st.rerun()
            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(f"Error: {str(e)}")
                logger.error(f"Generation error: {e}")

# CSV Upload
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Or Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.session_state.data_source = uploaded_file.name
        st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading CSV: {str(e)}")

# ════════════════════════════════════════════════════════════════════
# MAIN CONTENT - TABS
# ════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6, tab_euro, tab_ahf = st.tabs([
    "📊 Dashboard",
    "📈 Vital Signs",
    "🔢 Clinical Scores",
    "🎮 Invigilator Panel",
    "🔮 Forecast (NEW)",
    "📋 Report (NEW)",
    "🫀 EUROScore II"
])

# ════════════════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ════════════════════════════════════════════════════════════════════

with tab1:
    st.title("🏥 Virtual ICU - Real-Time Patient Monitoring")
    
    if st.session_state.data is None:
        st.info("👈 Select a demo scenario from sidebar or upload a CSV file to get started!")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Records", len(st.session_state.data))
        with col2:
            st.metric("📋 Columns", len(st.session_state.data.columns))
        with col3:
            st.metric("📅 Source", st.session_state.data_source.replace("(Sepsis-3)", "").strip()[:30])
        with col4:
            st.metric("🕐 Updated", st.session_state.last_updated or "N/A")
        
        st.markdown("---")
        st.subheader("📋 Data Preview (First 10 records)")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 Vital Signs Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "❤️ Heart Rate (bpm)",
                f"{st.session_state.data['heart_rate'].mean():.0f}",
                f"Range: {st.session_state.data['heart_rate'].min():.0f}-{st.session_state.data['heart_rate'].max():.0f}"
            )
        
        with col2:
            st.metric(
                "🫁 SpO2 (%)",
                f"{st.session_state.data['spo2'].mean():.1f}",
                f"Min: {st.session_state.data['spo2'].min():.1f}%"
            )
        
        with col3:
            st.metric(
                "📉 Systolic BP (mmHg)",
                f"{st.session_state.data['systolic_bp'].mean():.0f}",
                f"Range: {st.session_state.data['systolic_bp'].min():.0f}-{st.session_state.data['systolic_bp'].max():.0f}"
            )

# ════════════════════════════════════════════════════════════════════
# TAB 2: VITAL SIGNS
# ════════════════════════════════════════════════════════════════════

with tab2:
    st.title("📈 Vital Signs Monitoring")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please select a demo scenario or upload patient data first")
    else:
        data = st.session_state.data.copy()
        data['time_index'] = range(len(data))
        
        # Multi-parameter graph
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['time_index'],
            y=data['heart_rate'],
            name='Heart Rate (bpm)',
            line=dict(color='red', width=2),
            yaxis='y1'
        ))
        
        fig.add_trace(go.Scatter(
            x=data['time_index'],
            y=data['spo2'],
            name='SpO2 (%)',
            line=dict(color='blue', width=2),
            yaxis='y2'
        ))
        
        fig.add_trace(go.Scatter(
            x=data['time_index'],
            y=data['systolic_bp'],
            name='Systolic BP (mmHg)',
            line=dict(color='green', width=2),
            yaxis='y3'
        ))
        
        fig.add_trace(go.Scatter(
            x=data['time_index'],
            y=data['respiratory_rate'],
            name='RR (breaths/min)',
            line=dict(color='orange', width=2),
            yaxis='y4'
        ))
        
        fig.update_layout(
            title="Multi-Parameter Vital Signs (Real-Time Trend)",
            xaxis=dict(title="Time (5-minute intervals)"),
            yaxis=dict(title="HR (bpm)", side="left"),
            yaxis2=dict(title="SpO2 (%)", overlaying="y", side="right"),
            yaxis3=dict(title="SBP (mmHg)", overlaying="y", side="left", anchor="free", position=0.0),
            yaxis4=dict(title="RR (breaths/min)", overlaying="y", side="right", anchor="free", position=1.0),
            hovermode='x unified',
            height=600,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual parameter trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("❤️ Heart Rate Trend")
            fig_hr = px.line(
                data,
                x='time_index',
                y='heart_rate',
                title='Heart Rate Over Time',
                labels={'heart_rate': 'HR (bpm)', 'time_index': 'Time Index'}
            )
            st.plotly_chart(fig_hr, use_container_width=True)
        
        with col2:
            st.subheader("🫁 Oxygen Saturation Trend")
            fig_spo2 = px.line(
                data,
                x='time_index',
                y='spo2',
                title='SpO2 Over Time',
                labels={'spo2': 'SpO2 (%)', 'time_index': 'Time Index'}
            )
            st.plotly_chart(fig_spo2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════

def render_oxygenation_module(data, record_idx):
    import streamlit as st
    import plotly.graph_objects as go
    vitals = data.iloc[record_idx].to_dict()
    pao2 = vitals.get('pao2')
    fio2 = vitals.get('fio2')
    peep = vitals.get('peep')
    oi_end = vitals.get('oxygenation_index_end_op')
    oi_ext = vitals.get('oxygenation_index_pre_extubation')
    ards = vitals.get('ards_class', 'N/A')
    def safe(v):
        try: return float(v) if v is not None and str(v) not in ['nan','None',''] else None
        except: return None
    pao2=safe(pao2); fio2=safe(fio2); peep=safe(peep)
    oi_end=safe(oi_end); oi_ext=safe(oi_ext)
    oi = round(pao2/fio2) if pao2 and fio2 else None
    def classify(v):
        if v is None: return 'No data', 'gray'
        if v > 300: return 'Normal (>300)', 'green'
        elif v > 200: return 'Mild ARDS (200-300)', 'yellow'
        elif v > 100: return 'Moderate ARDS (100-200)', 'orange'
        else: return 'Severe ARDS (<100)', 'red'
    st.subheader('Oxygenation Index (Berlin Criteria)')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('paO2 (mmHg)', f'{pao2:.0f}' if pao2 else 'nan')
        st.metric('FiO2', f'{fio2:.2f}' if fio2 else 'nan')
        st.metric('PEEP (cmH2O)', f'{peep:.0f}' if peep else 'nan')
    with col2:
        display = oi or oi_end
        if display:
            label, _ = classify(display)
            st.metric('OI (paO2/FiO2)', f'{display:.0f}')
            st.info(f'Status: {label}')
        if oi_end: st.metric('OI end of surgery', f'{oi_end:.0f}')
        if oi_ext: st.metric('OI pre-extubation', f'{oi_ext:.0f}')
    with col3:
        st.metric('Clinical ARDS', str(ards))
        display_oi = oi or oi_end
        if display_oi:
            fig = go.Figure(go.Indicator(
                mode='gauge+number', value=display_oi,
                title={'text': 'Oxygenation Index'},
                gauge={'axis':{'range':[0,500]},
                    'bar':{'color':'darkblue'},
                    'steps':[
                        {'range':[0,100],'color':'red'},
                        {'range':[100,200],'color':'orange'},
                        {'range':[200,300],'color':'yellow'},
                        {'range':[300,500],'color':'lightgreen'}],
                    'threshold':{'line':{'color':'black','width':4},'thickness':0.75,'value':300}}))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

# TAB 3: CLINICAL SCORES
# ════════════════════════════════════════════════════════════════════

with tab3:
    st.title("🔢 Clinical Risk Scores")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please select a demo scenario or upload patient data")
    else:
        data = st.session_state.data.copy()
        
        st.subheader("Select Record to Score")
        record_idx = st.slider("Record number", 0, len(data)-1, 0, key="score_slider")
        
        # Get vitals from data
        vitals = data.iloc[record_idx].to_dict()
        import math
        vitals = {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in vitals.items()}
        
        # Ensure age is set
        if 'age' not in vitals or pd.isna(vitals['age']):
            vitals['age'] = 65
        
        # Sanitize vitals - replace all NaN with None before scoring
        def sanitize(v):
            if v is None: return None
            try:
                f = float(v)
                return None if math.isnan(f) else f
            except (TypeError, ValueError):
                return v
        vitals = {k: sanitize(v) for k, v in vitals.items()}
        # Calculate scores using v2 calculators
        news2 = NEWS2CalculatorV2.calculate(vitals)
        qsofa = qSOFACalculatorV2.calculate(vitals)
        cart = CARTCalculatorV2.calculate(vitals)
        recommendations = ClinicalRecommendationsEngineV2.generate(news2, qsofa, cart)
        # Oxygenation Index Module
        if any(c in data.columns for c in ['pao2','oxygenation_index_end_op']):
            st.divider()
            render_oxygenation_module(data, record_idx)

        
        # Display current vitals
        st.subheader("📊 Current Vital Signs")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("❤️ HR", f"{vitals['heart_rate']:.0f} bpm" if vitals.get('heart_rate') is not None else 'nan bpm')
        with col2:
            st.metric("🫁 SpO2", f"{vitals['spo2']:.1f}%" if vitals.get('spo2') is not None else 'nan%')
        with col3:
            st.metric("📉 SBP", f"{vitals['systolic_bp']:.0f} mmHg" if vitals.get('systolic_bp') is not None else "nan mmHg")
        with col4:
            st.metric("🌬️ RR", f"{vitals['respiratory_rate']:.0f} /min" if vitals.get('respiratory_rate') is not None else 'nan /min')
        with col5:
            st.metric("🌡️ Temp", f"{vitals['temperature']:.1f}°C" if vitals.get('temperature') is not None else 'nan°C')
        
        st.markdown("---")
        

        # NEWS2 Score
        st.subheader("📊 NEWS2 Score (National Early Warning Score 2)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Score", f"{news2['total']}/{news2['max_possible']}")
        with col2:
            risk_color_map = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            risk_icon = risk_color_map.get(news2['risk_level'], "❓")
            st.metric("Risk Level", f"{risk_icon} {news2['risk_level']}")
        with col3:
            st.metric("Recommendation", news2['recommendation'])
        
        with st.expander("📋 NEWS2 Components"):
            for component, value in news2['components'].items():
                st.write(f"**{component}**: {value} points")
        
        st.markdown("---")
        
        # qSOFA Score
        st.subheader("🦠 qSOFA Score (Sepsis-3)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Score", f"{qsofa['total']}/{qsofa['max_possible']}")
        with col2:
            sepsis_icon = "🔴 YES" if qsofa['needs_investigation'] else "✅ NO"
            st.metric("Sepsis Risk", sepsis_icon)
        with col3:
            st.metric("Recommendation", qsofa['recommendation'])
        
        with st.expander("📋 qSOFA Components"):
            for component, value in qsofa['components'].items():
                status_icon = "❌" if value == 0 else "⚠️"
                st.write(f"{status_icon} **{component}**: {value} point(s)")
        
        st.markdown("---")
        
        # CART Score
        st.subheader("⚡ CART Score (Cardiac Arrest Risk Triage)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Score", f"{cart['total']}/{cart['max_possible']}")
        with col2:
            risk_map = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "Highest": "🔴🔴"}
            risk_icon = risk_map.get(cart['risk_category'], "❓")
            st.metric("Risk Category", f"{risk_icon} {cart['risk_category']}")
        with col3:
            st.metric("Percentile", f"{cart['percentile']:.0f}%")
        
        st.markdown("---")
        
        # Clinical Recommendations
        st.subheader("💊 Clinical Recommendations")
        
        urgency_colors = {
            "Low": "alert-low",
            "Medium": "alert-medium",
            "High": "alert-high"
        }
        
        urgency_class = urgency_colors.get(recommendations['urgency'], "alert-low")
        st.markdown(
            f'<div class="{urgency_class}"><strong>⚠️ Urgency Level: {recommendations["urgency"]}</strong></div>',
            unsafe_allow_html=True
        )
        
        st.write(f"**Total Actions: {recommendations['total_actions']}**")
        
        for rec in recommendations['recommendations']:
            priority_icon = ["🔴", "🟠", "🟡", "🟢", "⚪"][min(4, rec['priority'] - 1)]
            st.markdown(
                f'<div class="recommendation-item">'
                f'{priority_icon} <strong>{rec["action"]}</strong><br/>'
                f'<small><em>{rec["rationale"]}</em></small><br/>'
                f'<small>Protocol: {rec["protocol"]}</small>'
                f'</div>',
                unsafe_allow_html=True
            )

# ════════════════════════════════════════════════════════════════════
# TAB 4: INVIGILATOR PANEL
# ════════════════════════════════════════════════════════════════════

with tab4:
    st.title("🎮 Invigilator Control Panel")
    st.info("🔬 Educational demonstration mode - Real-time parameter adjustment and scoring")
    
    st.subheader("Real-Time Parameter Adjustment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hr_invig = st.slider("Heart Rate (bpm)", 30, 200, 80, key="hr_slider")
        spo2_invig = st.slider("SpO2 (%)", 50, 100, 95, key="spo2_slider")
        sbp_invig = st.slider("Systolic BP (mmHg)", 30, 200, 120, key="sbp_slider")
        rr_invig = st.slider("Respiratory Rate (/min)", 6, 40, 16, key="rr_slider")
    
    with col2:
        temp_invig = st.slider("Temperature (°C)", 35.0, 42.0, 37.0, step=0.1, key="temp_slider")
        alert_invig = st.selectbox(
            "Alert Status",
            ["Alert", "Confused", "Lethargic", "Unresponsive"],
            key="alert_invig"
        )
        o2_invig = st.checkbox("Supplemental Oxygen", value=False, key="o2_invig")
        age_invig = st.slider("Age (years)", 18, 100, 65, key="age_invig")
    
    st.markdown("---")
    
    # Create vitals dict for invigilator
    invig_vitals = {
        'heart_rate': hr_invig,
        'systolic_bp': sbp_invig,
        'diastolic_bp': int(sbp_invig * 0.67),
        'respiratory_rate': rr_invig,
        'spo2': spo2_invig,
        'temperature': temp_invig,
        'alert_status': alert_invig,
        'supplemental_oxygen': o2_invig,
        'age': age_invig,
    }
    
    # Calculate scores in real-time
    news2_invig = NEWS2CalculatorV2.calculate(invig_vitals)
    qsofa_invig = qSOFACalculatorV2.calculate(invig_vitals)
    cart_invig = CARTCalculatorV2.calculate(invig_vitals)
    rec_invig = ClinicalRecommendationsEngineV2.generate(news2_invig, qsofa_invig, cart_invig)
    
    st.subheader("📊 Real-Time Score Response")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("NEWS2", f"{news2_invig['total']}/20")
        st.write(f"**Risk:** {news2_invig['risk_level']}")
    
    with col2:
        st.metric("qSOFA", f"{qsofa_invig['total']}/3")
        st.write(f"**Sepsis:** {'🔴 YES' if qsofa_invig['needs_investigation'] else '✅ NO'}")
    
    with col3:
        st.metric("CART", f"{cart_invig['total']}/20")
        st.write(f"**Risk:** {cart_invig['risk_category']}")
    
    st.markdown("---")
    st.subheader("🚨 Dynamic Recommendations")
    
    # Define urgency colors for this tab
    urgency_colors_invig = {
        "Low": "alert-low",
        "Medium": "alert-medium",
        "High": "alert-high"
    }
    
    urgency_class = urgency_colors_invig.get(rec_invig['urgency'], "alert-low")
    st.markdown(
        f'<div class="{urgency_class}"><strong>⚠️ Urgency: {rec_invig["urgency"]} (Level {rec_invig["urgency_level"]})</strong></div>',
        unsafe_allow_html=True
    )
    
    st.write(f"**{rec_invig['total_actions']} actions recommended:**")
    
    for rec in rec_invig['recommendations']:
        priority_icon = ["🔴", "🟠", "🟡", "🟢", "⚪"][min(4, rec['priority'] - 1)]
        st.markdown(
            f'<div class="recommendation-item">'
            f'{priority_icon} {rec["action"]}<br/>'
            f'<small><em>{rec["rationale"]}</em></small>'
            f'</div>',
            unsafe_allow_html=True
        )

# ════════════════════════════════════════════════════════════════════
# TAB 5: PREDICTIVE FORECAST (NEW - v2.0)
# ════════════════════════════════════════════════════════════════════

with tab5:
    if ADVANCED_FEATURES_AVAILABLE:
        if st.session_state.data is not None and len(st.session_state.data) > 0:
            # Initialize integration
            if 'integration' not in st.session_state:
                st.session_state.integration = StreamlitIntegration()
            
            integration = st.session_state.integration
            
            # Get current scores for context
            last_vitals = st.session_state.data.iloc[-1].to_dict()
            scores = {
                'news2': NEWS2CalculatorV2.calculate(last_vitals),
                'qsofa': qSOFACalculatorV2.calculate(last_vitals),
                'cart': CARTCalculatorV2.calculate(last_vitals)
            }
            
            # Render forecast tab
            integration.render_forecast_tab(st.session_state.data, scores)
        else:
            st.info("📊 Generate demo data first to see 24-hour forecast")
    else:
        st.error("⚠️ Advanced features not available. Ensure lstm_predictor.py is in the same directory.")

# ════════════════════════════════════════════════════════════════════
# TAB 6: CLINICAL REPORT (NEW - v2.0)
# ════════════════════════════════════════════════════════════════════

with tab6:
    if ADVANCED_FEATURES_AVAILABLE:
        if st.session_state.data is not None and len(st.session_state.data) > 0:
            # Initialize integration if not already done
            if 'integration' not in st.session_state:
                st.session_state.integration = StreamlitIntegration()
            
            integration = st.session_state.integration
            
            # Get patient data
            patient_data = {
                'patient_id': st.session_state.data_source if st.session_state.data_source else 'P001',
                'scenario': st.session_state.data_source.split('_')[0] if st.session_state.data_source else 'Unknown',
                'age': 65
            }
            
            # Get current scores
            last_vitals = st.session_state.data.iloc[-1].to_dict()
            scores = {
                'news2': NEWS2CalculatorV2.calculate(last_vitals),
                'qsofa': qSOFACalculatorV2.calculate(last_vitals),
                'cart': CARTCalculatorV2.calculate(last_vitals)
            }
            
            # Get recommendations
            recommendations = ClinicalRecommendationsEngineV2.generate(
                scores['news2'],
                scores['qsofa'],
                scores['cart']
            ).get('recommendations', [])
            
            # Render report tab
            integration.render_report_tab(
                st.session_state.data,
                scores,
                recommendations,
                patient_data
            )
        else:
            st.info("📋 Generate demo data first to generate clinical report")
    else:
        st.error("⚠️ Advanced features not available. Ensure pdf_generator.py is in the same directory.")

# ════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "Virtual ICU v2.0 - AI-Driven Real-Time Early Warning System | "
    f"Data Source: {st.session_state.data_source if st.session_state.data is not None else 'None'} | "
    f"Version: {'v2.0 (Advanced Features)' if ADVANCED_FEATURES_AVAILABLE else 'v1.2'}"
    "</div>",
    unsafe_allow_html=True
)

# ============================================================
# TAB: EUROSCORE II CALCULATOR
# ============================================================
with tab_euro:
    st.title("EUROScore II Calculator")
    st.markdown("*Nashef et al. Eur Heart J 2012 - Predicted operative mortality for cardiac surgery*")
    
    import math
    
    def calc_euroscore2(p):
        lo = -5.324537
        age = p.get('age', 60)
        if age > 60: lo += 0.0285181 * (age - 60)
        if p.get('female'): lo += 0.2196434
        cc = p.get('creatinine_clearance', 85)
        if cc < 50: lo += 0.987207
        elif cc < 85: lo += 0.303553
        if p.get('extracardiac_arteriopathy'): lo += 0.5360268
        if p.get('poor_mobility'): lo += 0.2296615
        if p.get('previous_cardiac_surgery'): lo += 1.118599
        if p.get('chronic_lung_disease'): lo += 0.1886564
        if p.get('active_endocarditis'): lo += 0.6194522
        if p.get('critical_preop'): lo += 1.086517
        if p.get('diabetes_insulin'): lo += 0.3542749
        nyha = p.get('nyha', 1)
        if nyha == 2: lo += 0.1070545
        elif nyha == 3: lo += 0.2958358
        elif nyha == 4: lo += 0.5597929
        if p.get('ccs4_angina'): lo += 0.2226147
        ef = p.get('ef', 50)
        if ef < 21: lo += 0.9346919
        elif ef < 31: lo += 0.8084096
        elif ef < 51: lo += 0.3150652
        if p.get('recent_mi'): lo += 0.1528943
        pasp = p.get('pasp', 25)
        if pasp >= 55: lo += 0.5181810
        elif pasp >= 31: lo += 0.1788899
        urg = p.get('urgency', 'elective')
        if urg == 'urgent': lo += 0.3174673
        elif urg == 'emergency': lo += 0.7039121
        elif urg == 'salvage': lo += 1.362947
        proc = p.get('procedure', 'isolated_cabg')
        if proc == 'single_non_cabg': lo += 0.0062118
        elif proc == 'two_procedures': lo += 0.5521478
        elif proc == 'three_procedures': lo += 0.9724533
        if p.get('thoracic_aorta'): lo += 0.6527205
        return round(math.exp(lo) / (1 + math.exp(lo)) * 100, 2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Characteristics")
        age = st.number_input("Age (years)", 18, 100, 65)
        female = st.checkbox("Female")
        cc = st.number_input("Creatinine clearance (ml/min)", 1, 200, 85)
        ef = st.slider("LV Ejection Fraction (%)", 1, 80, 50)
        nyha = st.selectbox("NYHA Class", [1, 2, 3, 4], index=1)
        pasp = st.number_input("Pulmonary artery systolic pressure (mmHg)", 10, 120, 25)
        
        st.subheader("Risk Factors")
        extracardiac = st.checkbox("Extracardiac arteriopathy")
        poor_mobility = st.checkbox("Poor mobility")
        prev_surgery = st.checkbox("Previous cardiac surgery")
        lung_disease = st.checkbox("Chronic lung disease")
        endocarditis = st.checkbox("Active endocarditis")
        critical = st.checkbox("Critical preoperative state")
        diabetes = st.checkbox("Diabetes on insulin")
        ccs4 = st.checkbox("CCS Class 4 angina")
        recent_mi = st.checkbox("Recent MI (<90 days)")
        thoracic = st.checkbox("Surgery on thoracic aorta")
    
    with col2:
        st.subheader("Procedure")
        urgency = st.selectbox("Urgency", 
            ["elective", "urgent", "emergency", "salvage"],
            format_func=lambda x: {"elective":"Elective","urgent":"Urgent","emergency":"Emergency","salvage":"Salvage"}[x])
        procedure = st.selectbox("Weight of procedure",
            ["isolated_cabg", "single_non_cabg", "two_procedures", "three_procedures"],
            format_func=lambda x: {"isolated_cabg":"Isolated CABG","single_non_cabg":"Single non-CABG","two_procedures":"Two procedures","three_procedures":"Three or more procedures"}[x])
        
        params = {
            'age': age, 'female': female, 'creatinine_clearance': cc,
            'ef': ef, 'nyha': nyha, 'pasp': pasp,
            'extracardiac_arteriopathy': extracardiac, 'poor_mobility': poor_mobility,
            'previous_cardiac_surgery': prev_surgery, 'chronic_lung_disease': lung_disease,
            'active_endocarditis': endocarditis, 'critical_preop': critical,
            'diabetes_insulin': diabetes, 'ccs4_angina': ccs4, 'recent_mi': recent_mi,
            'thoracic_aorta': thoracic, 'urgency': urgency, 'procedure': procedure
        }
        
        mortality = calc_euroscore2(params)
        
        st.subheader("Result")
        if mortality < 2:
            risk_label = "Low Risk"
            color = "success"
        elif mortality < 5:
            risk_label = "Intermediate Risk"
            color = "warning"
        else:
            risk_label = "High Risk"
            color = "error"
        
        st.metric("Predicted Mortality", f"{mortality}%")
        getattr(st, color)(f"{risk_label}")
        
        import plotly.graph_objects as go
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mortality,
            number={"suffix": "%"},
            title={"text": "EUROScore II", "AHF Profiles"},
            gauge={
                "axis": {"range": [0, 30]},
                "bar": {"color": "darkred"},
                "steps": [
                    {"range": [0, 2], "color": "lightgreen"},
                    {"range": [2, 5], "color": "yellow"},
                    {"range": [5, 30], "color": "salmon"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 5}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("EUROScore II: Nashef et al. Eur Heart J 2012;33:2427-2435")

# ============================================================
# TAB: AHF PROFILES (ESC 2021)
# ============================================================
with tab_ahf:
    st.title("Acute Heart Failure - ESC Profiles")
    st.markdown("*ESC Guidelines 2021 - Haemodynamic profiles and treatment algorithm*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Haemodynamic Profile")
        st.markdown("Select the patient profile based on perfusion and congestion status:")
        
        profile = st.radio(
            "Patient Profile",
            ["Warm-Wet (Decompensated HF)", 
             "Cold-Wet (Cardiogenic Shock)",
             "Cold-Dry (Hypovolaemia)",
             "Warm-Dry (Compensated)"],
            index=0
        )
        
        st.divider()
        st.subheader("Clinical Assessment")
        sbp = st.number_input("Systolic BP (mmHg)", 40, 250, 110)
        hr = st.number_input("Heart Rate (bpm)", 20, 200, 90)
        spo2 = st.number_input("SpO2 (%)", 50, 100, 94)
        lactate = st.number_input("Lactate (mmol/L)", 0.0, 20.0, 2.0, step=0.1)
        urine = st.number_input("Urine output (ml/kg/h)", 0.0, 5.0, 0.5, step=0.1)
        
    with col2:
        st.subheader("Profile Details")
        
        profiles = {
            "Warm-Wet (Decompensated HF)": {
                "signs": ["Dyspnoea at rest", "Oedema", "SpO2 <94%", "PCWP >18 mmHg", "Normal or high BP"],
                "treatment": ["IV Furosemide", "Nitrates (if SBP >110)", "Monitor diuresis", "CPAP if SpO2<90%"],
                "mortality": "10-20%",
                "color": "warning"
            },
            "Cold-Wet (Cardiogenic Shock)": {
                "signs": ["SBP <90 mmHg", "Cold extremities", "Oliguria <0.5 ml/kg/h", "Lactate >2 mmol/L", "PCWP >18 mmHg"],
                "treatment": ["Dobutamine + Noradrenaline", "IABP or VA-ECMO", "Careful diuretics", "CO/CI monitoring"],
                "mortality": "40-60%",
                "color": "error"
            },
            "Cold-Dry (Hypovolaemia)": {
                "signs": ["Low BP", "Tachycardia", "CVP <5 mmHg", "PCWP <12 mmHg", "Dry mucous membranes"],
                "treatment": ["Fluid challenge 250ml bolus", "Assess fluid responsiveness", "Exclude tamponade", "Stop diuretics"],
                "mortality": "15-25%",
                "color": "warning"
            },
            "Warm-Dry (Compensated)": {
                "signs": ["Normal BP", "Good perfusion", "Normal diuresis", "SpO2 >95%", "No congestion"],
                "treatment": ["Optimise oral therapy", "Prepare for transfer", "Monitor LVEF", "Rehabilitation"],
                "mortality": "<5%",
                "color": "success"
            }
        }
        
        p = profiles[profile]
        
        if p["color"] == "error":
            st.error(f"Mortality risk: {p['mortality']}")
        elif p["color"] == "warning":
            st.warning(f"Mortality risk: {p['mortality']}")
        else:
            st.success(f"Mortality risk: {p['mortality']}")
        
        st.markdown("**Clinical signs:**")
        for sign in p["signs"]:
            st.markdown(f"▸ {sign}")
        
        st.markdown("**Treatment:**")
        for t in p["treatment"]:
            st.markdown(f"✓ {t}")
        
        st.divider()
        st.subheader("Shock Severity")
        
        shock_score = 0
        if sbp < 90: shock_score += 2
        elif sbp < 100: shock_score += 1
        if lactate > 4: shock_score += 2
        elif lactate > 2: shock_score += 1
        if urine < 0.3: shock_score += 2
        elif urine < 0.5: shock_score += 1
        if hr > 120: shock_score += 1
        if spo2 < 90: shock_score += 1
        
        if shock_score >= 5:
            st.error(f"Shock Score: {shock_score}/8 — CRITICAL. Consider IABP/VA-ECMO")
        elif shock_score >= 3:
            st.warning(f"Shock Score: {shock_score}/8 — HIGH. Inotropes required")
        else:
            st.success(f"Shock Score: {shock_score}/8 — LOW-MODERATE")
        
    st.divider()
    st.subheader("Mechanical Circulatory Support Algorithm")
    
    col3, col4, col5 = st.columns(3)
    with col3:
        st.info("**Step 1: IABP**\nMAP <65 + inotropes\nCI <2.2 L/min/m²\nBridge to recovery/LVAD")
    with col4:
        st.warning("**Step 2: Impella**\nRefractory shock\nCI <1.8 L/min/m²\nHigh-risk PCI")
    with col5:
        st.error("**Step 3: VA-ECMO**\nRefractory cardiogenic shock\nCardiac arrest\nBridge to transplant")
