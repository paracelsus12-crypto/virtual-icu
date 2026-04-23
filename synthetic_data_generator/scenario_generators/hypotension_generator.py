"""Virtual ICU - Hypotension Generator"""

import numpy as np
import pandas as pd
from ..base_generator import BasePatientGenerator


class HypotensionGenerator(BasePatientGenerator):
    """Generate synthetic patient data with Hypotension progression.
    
    4 Stages:
    Stage 1: SBP 90-100 (Mild)
    Stage 2: SBP 70-90 (Moderate)
    Stage 3: SBP 50-70 (Severe)
    Stage 4: SBP < 50 (Shock - critical)
    """

    def __init__(self, patient_id: str, duration_hours: int = 6, variant: str = "progressive", sample_rate_minutes: int = 5):
        if variant not in ['progressive', 'sudden']:
            raise ValueError("Variant must be 'progressive' or 'sudden'")

        super().__init__(patient_id=patient_id, duration_hours=duration_hours, severity="hypotension", sample_rate_minutes=sample_rate_minutes)
        
        self.variant = variant
        self.baseline_vitals['lactate'] = 1.0
        self.baseline_vitals['urine_output'] = 1.0  # mL/kg/hr

    def generate(self) -> pd.DataFrame:
        """Generate 6-hour Hypotension progression."""
        print(f"📉 Generating HYPOTENSION ({self.variant})...")

        for sample_idx in range(self.total_samples):
            progress = self._get_time_progress(sample_idx)
            current_hour = (sample_idx * self.duration_hours * 60 / self.total_samples) / 60
            timestamp = f"T+{int(current_hour)}h"

            current_vitals = self.baseline_vitals.copy()

            # BP progression
            if self.variant == 'progressive':
                # Gradual drop
                sbp_target = 40
                dbp_target = 25
            else:
                # Sudden drop
                if progress < 0.2:
                    sbp_target = self.baseline_vitals['systolic_bp']
                    dbp_target = self.baseline_vitals['diastolic_bp']
                else:
                    sbp_target = 40
                    dbp_target = 25
                    progress = (progress - 0.2) / 0.8  # Adjust for sudden phase

            current_vitals['systolic_bp'] = self.baseline_vitals['systolic_bp'] - (self.baseline_vitals['systolic_bp'] - sbp_target) * progress
            current_vitals['diastolic_bp'] = self.baseline_vitals['diastolic_bp'] - (self.baseline_vitals['diastolic_bp'] - dbp_target) * progress
            current_vitals['map'] = current_vitals['systolic_bp'] * 0.33 + current_vitals['diastolic_bp'] * 0.67

            # HR - Compensatory tachycardia
            hr_increase = 0.50 * progress * 80
            current_vitals['heart_rate'] = self.baseline_vitals['heart_rate'] + hr_increase

            # RR - Increases with shock
            if progress > 0.4:
                rr_increase = (progress - 0.4) * 2 * 15
                current_vitals['respiratory_rate'] = self.baseline_vitals['respiratory_rate'] + rr_increase

            # SpO2 - Decreases if severe shock
            if progress > 0.6:
                spo2_drop = (progress - 0.6) * 2 * 20
                current_vitals['spo2'] = self.baseline_vitals['spo2'] - spo2_drop

            # Lactate - Increases (tissue hypoperfusion)
            lactate_increase = progress * 8.0
            current_vitals['lactate'] = self.baseline_vitals['lactate'] + lactate_increase

            # pH - Decreases (lactic acidosis)
            if current_vitals['lactate'] > 1:
                acidosis = 0.02 * (current_vitals['lactate'] - 1)
                current_vitals['ph'] = self.baseline_vitals['ph'] - acidosis

            # Urine output - Decreases (renal perfusion)
            if progress > 0.3:
                urine_drop = (progress - 0.3) * 1.5 * 0.8
                current_vitals['urine_output'] = max(0.1, self.baseline_vitals['urine_output'] - urine_drop)

            # Add noise
            for vital_name in current_vitals.keys():
                if vital_name not in ['map', 'urine_output']:
                    current_vitals[vital_name] = self.add_noise(current_vitals[vital_name], vital_name, noise_factor=0.4)

            self.apply_physiological_correlations(current_vitals)
            self.timestamps.append(timestamp)
            self.vitals_history.append(current_vitals)

        df = pd.DataFrame(self.vitals_history)
        print(f"✅ Generated {len(df)} samples ({self.variant})")
        return df

    def get_hypotension_summary(self) -> dict:
        if not self.vitals_history:
            return {}
        df = pd.DataFrame(self.vitals_history)
        return {
            'variant': self.variant,
            'initial_sbp': float(df['systolic_bp'].iloc[0]),
            'final_sbp': float(df['systolic_bp'].iloc[-1]),
            'min_sbp': float(df['systolic_bp'].min()),
            'initial_hr': float(df['heart_rate'].iloc[0]),
            'final_hr': float(df['heart_rate'].iloc[-1]),
            'max_lactate': float(df['lactate'].max()),
            'min_ph': float(df['ph'].min()),
            'cardiogenic_shock': float(df['systolic_bp'].min()) < 50,
        }


if __name__ == "__main__":
    print("Virtual ICU - Hypotension Generator\n" + "=" * 70)

    print("\n1. Generating PROGRESSIVE HYPOTENSION...")
    gen_prog = HypotensionGenerator("HYPO_PROG_001", variant="progressive")
    data_prog = gen_prog.generate()
    gen_prog.export_to_csv("hypotension_progressive.csv")
    summary_prog = gen_prog.get_hypotension_summary()
    print(f"   ✅ SBP: {summary_prog['initial_sbp']:.0f} → {summary_prog['final_sbp']:.0f} mmHg (min: {summary_prog['min_sbp']:.0f})")
    print(f"   ✅ HR: {summary_prog['initial_hr']:.0f} → {summary_prog['final_hr']:.0f} bpm")
    print(f"   ✅ Max lactate: {summary_prog['max_lactate']:.1f} mmol/L")
    print(f"   ✅ Cardiogenic shock: {summary_prog['cardiogenic_shock']}")

    print("\n2. Generating SUDDEN HYPOTENSION...")
    gen_sudden = HypotensionGenerator("HYPO_SUDDEN_001", variant="sudden")
    data_sudden = gen_sudden.generate()
    gen_sudden.export_to_csv("hypotension_sudden.csv")
    summary_sudden = gen_sudden.get_hypotension_summary()
    print(f"   ✅ SBP: {summary_sudden['initial_sbp']:.0f} → {summary_sudden['final_sbp']:.0f} mmHg (min: {summary_sudden['min_sbp']:.0f})")
    print(f"   ✅ HR: {summary_sudden['initial_hr']:.0f} → {summary_sudden['final_hr']:.0f} bpm")
    print(f"   ✅ Max lactate: {summary_sudden['max_lactate']:.1f} mmol/L")
    print(f"   ✅ Cardiogenic shock: {summary_sudden['cardiogenic_shock']}")

    print("\n✅ Hypotension scenarios completed!")
