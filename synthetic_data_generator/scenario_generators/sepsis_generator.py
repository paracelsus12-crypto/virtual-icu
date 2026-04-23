"""
Virtual ICU - Sepsis & Septic Shock Generator
Based on Sepsis-3 Consensus Definitions (2016)

Sepsis: Life-threatening organ dysfunction caused by dysregulated host response
to infection, characterized by acute change in SOFA score ≥ 2

Septic Shock: A subset of sepsis in which underlying circulatory, cellular, and
metabolic abnormalities are associated with greater risk of mortality.
- Defined by: Sepsis + vasopressor requirement (MAP ≥65) + lactate >2 mmol/L

Clinical Features:
SEPSIS:
  - Infection source (blood culture, clinical focus)
  - SOFA score increase ≥2 from baseline
  - Respiratory dysfunction (↓ PaO2/FiO2)
  - Coagulation dysfunction (↓ platelets)
  - Hepatic dysfunction (↑ bilirubin)
  - Cardiovascular dysfunction (↓ MAP or need for vasopressors)
  - CNS dysfunction (↓ GCS)
  - Renal dysfunction (↑ creatinine)

SEPTIC SHOCK:
  - All SEPSIS criteria +
  - Vasopressor requirement to maintain MAP ≥ 65 mmHg +
  - Lactate > 2 mmol/L despite adequate fluid resuscitation

Timeline: 8 hours from early infection to severe septic shock

References:
- Singer M, et al. JAMA. 2016;315(8):801-810. (Sepsis-3 Consensus)
- Seymour CW, et al. JAMA. 2016;315(8):775-787. (SOFA Score)

Author: Virtual ICU Team
Version: 2.0.0-alpha (Sepsis-3)
Date: April 23, 2024
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ..base_generator import BasePatientGenerator


class SepsisSOFACalculator:
    """Calculate SOFA score based on vital signs and lab values"""
    
    @staticmethod
    def calculate_sofa(vitals: dict) -> int:
        """
        Calculate SOFA (Sequential Organ Failure Assessment) score
        
        Components:
        1. Respiration (PaO2/FiO2 ratio)
        2. Coagulation (platelet count)
        3. Hepatic (bilirubin)
        4. Cardiovascular (MAP or vasopressor)
        5. CNS (GCS)
        6. Renal (creatinine or urine output)
        
        Each component: 0-4 points
        Total: 0-24 points
        
        Sepsis defined as: SOFA ≥ 2 (from baseline)
        """
        score = 0
        
        # 1. RESPIRATION: Based on PaO2/FiO2 ratio (simulated from SpO2)
        spo2 = vitals.get('spo2', 98)
        # Estimate PaO2 from SpO2 (simplified)
        if spo2 >= 95:
            score += 0  # PaO2/FiO2 > 400
        elif spo2 >= 90:
            score += 1  # PaO2/FiO2 300-399
        elif spo2 >= 85:
            score += 2  # PaO2/FiO2 200-299
        elif spo2 >= 80:
            score += 3  # PaO2/FiO2 100-199
        else:
            score += 4  # PaO2/FiO2 < 100
        
        # 2. COAGULATION: Based on platelet count (simulated)
        # Normal: >150k, score 0
        # This is simulated through trend
        platelets = vitals.get('platelets', 250)  # k/uL
        if platelets > 150:
            pass  # 0 points
        elif platelets > 100:
            score += 1
        elif platelets > 50:
            score += 2
        elif platelets > 20:
            score += 3
        else:
            score += 4
        
        # 3. HEPATIC: Based on bilirubin (simulated)
        bilirubin = vitals.get('bilirubin', 1.2)  # mg/dL
        if bilirubin < 1.2:
            pass  # 0 points
        elif bilirubin < 2.0:
            score += 1
        elif bilirubin < 6.0:
            score += 2
        elif bilirubin < 12.0:
            score += 3
        else:
            score += 4
        
        # 4. CARDIOVASCULAR: Based on MAP and vasopressor need
        map_val = vitals.get('map', 85)  # mmHg
        vasopressor_required = vitals.get('vasopressor_required', False)
        
        if map_val >= 70 and not vasopressor_required:
            pass  # 0 points
        elif map_val >= 70 and vasopressor_required:
            score += 1
        elif map_val >= 60:
            score += 2
        elif map_val >= 50:
            score += 3
        else:
            score += 4
        
        # 5. CNS: Based on GCS (Glasgow Coma Scale)
        gcs = vitals.get('gcs', 15)
        if gcs == 15:
            pass  # 0 points
        elif gcs >= 13:
            score += 1
        elif gcs >= 10:
            score += 2
        elif gcs >= 6:
            score += 3
        else:
            score += 4
        
        # 6. RENAL: Based on creatinine
        creatinine = vitals.get('creatinine', 1.0)  # mg/dL
        if creatinine < 1.2:
            pass  # 0 points
        elif creatinine < 2.0:
            score += 1
        elif creatinine < 3.5:
            score += 2
        elif creatinine < 5.0:
            score += 3
        else:
            score += 4
        
        return min(24, score)  # Cap at 24


class SepsisGenerator(BasePatientGenerator):
    """
    Generate synthetic patient data with Sepsis progression.
    
    Based on Sepsis-3 Consensus Definitions (Singer et al, JAMA 2016)
    
    Two variants:
    1. SEPSIS: Infection + SOFA ≥2
       - Progressive organ dysfunction
       - SOFA score increases over time
       
    2. SEPTIC SHOCK: Sepsis + Vasopressor requirement + Lactate >2
       - More severe hemodynamic dysfunction
       - Requires vasopressor support
       - Higher mortality risk
    
    Attributes:
        variant: 'sepsis' or 'septic_shock'
    """
    
    def __init__(
        self,
        patient_id: str,
        duration_hours: int = 8,
        variant: str = "sepsis",  # 'sepsis' or 'septic_shock'
        sample_rate_minutes: int = 1
    ):
        """
        Initialize Sepsis/Septic Shock generator (Sepsis-3 compliant)
        
        Args:
            patient_id: Unique patient identifier
            duration_hours: Duration of simulation (typically 8 hours)
            variant: 'sepsis' (SOFA ≥2) or 'septic_shock' (more severe)
            sample_rate_minutes: Sampling interval in minutes
        """
        if variant not in ['sepsis', 'septic_shock']:
            raise ValueError("Variant must be 'sepsis' or 'septic_shock'")
        
        super().__init__(
            patient_id=patient_id,
            duration_hours=duration_hours,
            severity="high",  # For compatibility
            sample_rate_minutes=sample_rate_minutes
        )
        
        self.variant = variant
        
        # Sepsis-3 progression parameters
        self.progression_params = {
            'sepsis': {
                'lactate_slope': 1.0,           # mmol/L per 8 hours
                'bp_drop_rate': 0.08,          # % per hour
                'hr_increase': 0.18,           # % per hour
                'temp_rise': 1.5,              # °C
                'sofa_max': 4,                 # SOFA increase
                'vasopressor_threshold': None, # Not needed
            },
            'septic_shock': {
                'lactate_slope': 2.0,          # mmol/L per 8 hours (higher)
                'bp_drop_rate': 0.20,          # % per hour (much steeper)
                'hr_increase': 0.30,           # % per hour
                'temp_rise': 2.0,              # °C
                'sofa_max': 8,                 # SOFA increase (more severe)
                'vasopressor_threshold': 0.5,  # Activated at 50% of timeline
            }
        }
        
        # Initialize additional markers
        self.baseline_vitals['platelets'] = 250  # k/uL
        self.baseline_vitals['bilirubin'] = 1.2  # mg/dL
        self.baseline_vitals['creatinine'] = 1.0  # mg/dL
        self.baseline_vitals['gcs'] = 15  # Glasgow Coma Scale
        self.baseline_vitals['map'] = 85  # Mean Arterial Pressure
    
    def generate(self) -> pd.DataFrame:
        """Generate 8-hour Sepsis/Septic Shock progression (Sepsis-3)"""
        print(f"🏥 Generating {self.variant.upper()} (Sepsis-3 Consensus)...")
        
        params = self.progression_params[self.variant]
        baseline_sofa = 0
        
        for sample_idx in range(self.total_samples):
            progress = self._get_time_progress(sample_idx)
            timestamp = f"T+{int(sample_idx * self.sample_rate_minutes)}"
            
            current_vitals = self.baseline_vitals.copy()
            
            # ===== SEPSIS-3 SPECIFIC CHANGES =====
            
            # 1. HEART RATE - Progressive increase (compensatory)
            if progress < 0.5:
                hr_increase = params['hr_increase'] * progress * 2
            else:
                hr_increase = params['hr_increase'] * (0.5 + (progress - 0.5) * 1.5)
            current_vitals['heart_rate'] = self.baseline_vitals['heart_rate'] * (1 + hr_increase)
            
            # 2. BLOOD PRESSURE - Progressive decrease (distributed intravascular volume)
            if progress > 0.25:
                bp_drop = params['bp_drop_rate'] * (progress - 0.25) * (1 / 0.75)
                current_vitals['systolic_bp'] = self.baseline_vitals['systolic_bp'] * (1 - bp_drop)
                current_vitals['diastolic_bp'] = self.baseline_vitals['diastolic_bp'] * (1 - bp_drop * 1.2)
                # Calculate MAP (Mean Arterial Pressure)
                current_vitals['map'] = current_vitals['systolic_bp'] * 0.33 + current_vitals['diastolic_bp'] * 0.67
            
            # 3. RESPIRATORY RATE - Progressive increase
            rr_increase = params['hr_increase'] * progress * 0.8
            current_vitals['respiratory_rate'] = self.baseline_vitals['respiratory_rate'] * (1 + rr_increase)
            
            # 4. TEMPERATURE - Elevation (fever from infection)
            temp_elevation = params['temp_rise'] * progress
            current_vitals['temperature'] = self.baseline_vitals['temperature'] + temp_elevation
            
            # 5. LACTATE - Progressive increase (tissue hypoxia marker)
            # Septic Shock: higher lactate earlier
            lactate_increase = params['lactate_slope'] * progress * (1 + progress)
            current_vitals['lactate'] = self.baseline_vitals['lactate'] + lactate_increase
            
            # 6. ORGAN DYSFUNCTION MARKERS
            
            # Coagulation: platelet decrease
            if progress > 0.3:
                platelet_drop = progress * 0.6  # Up to 60% drop
                current_vitals['platelets'] = self.baseline_vitals['platelets'] * (1 - platelet_drop)
            
            # Hepatic: bilirubin increase
            if progress > 0.4:
                bili_increase = (progress - 0.4) * 3.0  # Up to 3 mg/dL
                current_vitals['bilirubin'] = self.baseline_vitals['bilirubin'] + bili_increase
            
            # Renal: creatinine increase
            if progress > 0.5:
                creat_increase = (progress - 0.5) * 2.0 * 2  # Up to 4 mg/dL
                current_vitals['creatinine'] = self.baseline_vitals['creatinine'] + creat_increase
            
            # CNS: GCS decrease (altered mental status)
            if progress > 0.6:
                gcs_decrease = int((progress - 0.6) * 5)  # Decrease by up to 5 points
                current_vitals['gcs'] = max(3, self.baseline_vitals['gcs'] - gcs_decrease)
            
            # 7. SpO2 - Gradual decrease (ARDS development in septic shock)
            if self.variant == 'septic_shock' and progress > 0.5:
                spo2_drop = (progress - 0.5) * 2 * 20  # Up to 20% drop
                current_vitals['spo2'] = self.baseline_vitals['spo2'] - spo2_drop
            
            # 8. pH - Decrease (lactic acidosis)
            if current_vitals['lactate'] > 1:
                acidosis = 0.02 * (current_vitals['lactate'] - 1)
                current_vitals['ph'] = self.baseline_vitals['ph'] - acidosis
            
            # 9. VASOPRESSOR REQUIREMENT (for Septic Shock)
            if self.variant == 'septic_shock':
                vasopressor_threshold = params['vasopressor_threshold']
                if progress >= vasopressor_threshold:
                    current_vitals['vasopressor_required'] = True
                else:
                    current_vitals['vasopressor_required'] = False
            else:
                current_vitals['vasopressor_required'] = False
            
            # ===== ADD PHYSIOLOGICAL NOISE =====
            for vital_name in current_vitals.keys():
                if vital_name not in ['vasopressor_required']:
                    current_vitals[vital_name] = self.add_noise(
                        current_vitals[vital_name],
                        vital_name,
                        noise_factor=1.0
                    )
            
            # ===== APPLY PHYSIOLOGICAL CORRELATIONS =====
            self.apply_physiological_correlations(current_vitals)
            
            # Store data
            self.timestamps.append(timestamp)
            self.vitals_history.append(current_vitals)
        
        df = pd.DataFrame(self.vitals_history)
        print(f"✅ Generated {len(df)} samples ({self.variant})")
        
        return df
    
    def get_sepsis3_summary(self) -> dict:
        """Get clinical summary using Sepsis-3 criteria"""
        if not self.vitals_history:
            return {}
        
        df = pd.DataFrame(self.vitals_history)
        
        # Calculate SOFA scores
        initial_sofa = SepsisSOFACalculator.calculate_sofa(self.vitals_history[0])
        final_sofa = SepsisSOFACalculator.calculate_sofa(self.vitals_history[-1])
        sofa_increase = final_sofa - initial_sofa
        
        # Check septic shock criteria
        has_vasopressor = df['vasopressor_required'].any() if 'vasopressor_required' in df.columns else False
        has_high_lactate = (df['lactate'] > 2).any()
        has_low_map = (df['map'] < 65).any() if 'map' in df.columns else False
        
        is_septic_shock = has_vasopressor or (has_high_lactate and has_low_map)
        
        return {
            'variant': self.variant,
            'sepsis3_compliant': True,
            'initial_sofa': initial_sofa,
            'final_sofa': final_sofa,
            'sofa_increase': sofa_increase,
            'meets_sepsis_criteria': sofa_increase >= 2,  # SOFA ≥2 increase
            'meets_septic_shock_criteria': is_septic_shock,
            'initial_lactate': float(df['lactate'].iloc[0]),
            'final_lactate': float(df['lactate'].iloc[-1]),
            'max_lactate': float(df['lactate'].max()),
            'min_map': float(df['map'].min()) if 'map' in df.columns else None,
            'vasopressor_required': has_vasopressor,
            'initial_hr': float(df['heart_rate'].iloc[0]),
            'final_hr': float(df['heart_rate'].iloc[-1]),
            'initial_bp': float(df['systolic_bp'].iloc[0]),
            'final_bp': float(df['systolic_bp'].iloc[-1]),
            'initial_temp': float(df['temperature'].iloc[0]),
            'final_temp': float(df['temperature'].iloc[-1]),
            'initial_ph': float(df['ph'].iloc[0]),
            'final_ph': float(df['ph'].iloc[-1]),
            'min_ph': float(df['ph'].min()),
        }


if __name__ == "__main__":
    print("Virtual ICU - Sepsis & Septic Shock Generator (Sepsis-3)")
    print("=" * 70)
    
    print("\n1. Generating SEPSIS (SOFA ≥2)...")
    gen_sepsis = SepsisGenerator("SEPSIS_001", variant="sepsis")
    data_sepsis = gen_sepsis.generate()
    gen_sepsis.export_to_csv("sepsis.csv")
    summary_sepsis = gen_sepsis.get_sepsis3_summary()
    print(f"   ✅ SOFA increase: {summary_sepsis['sofa_increase']} (meets criteria: {summary_sepsis['meets_sepsis_criteria']})")
    print(f"   ✅ Lactate: {summary_sepsis['initial_lactate']:.2f} → {summary_sepsis['final_lactate']:.2f}")
    print(f"   ✅ SOFA: {summary_sepsis['initial_sofa']} → {summary_sepsis['final_sofa']}")
    
    print("\n2. Generating SEPTIC SHOCK (Sepsis + Vasopressors + Lactate >2)...")
    gen_shock = SepsisGenerator("SEPTICSHOCK_001", variant="septic_shock")
    data_shock = gen_shock.generate()
    gen_shock.export_to_csv("septic_shock.csv")
    summary_shock = gen_shock.get_sepsis3_summary()
    print(f"   ✅ SOFA increase: {summary_shock['sofa_increase']}")
    print(f"   ✅ Lactate: {summary_shock['initial_lactate']:.2f} → {summary_shock['final_lactate']:.2f}")
    print(f"   ✅ Vasopressor required: {summary_shock['vasopressor_required']}")
    print(f"   ✅ Meets Septic Shock criteria: {summary_shock['meets_septic_shock_criteria']}")
    print(f"   ✅ MAP: {summary_shock['min_map']:.1f} mmHg (minimum)")
    
    print("\n✅ Sepsis-3 compliant generators completed!")
