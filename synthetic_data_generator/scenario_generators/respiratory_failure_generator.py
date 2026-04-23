"""
Virtual ICU - Respiratory Failure Scenario Generator - CORRECTED
"""

import numpy as np
import pandas as pd
from ..base_generator import BasePatientGenerator


class RespiratoryFailureGenerator(BasePatientGenerator):
    """Generate synthetic patient data with Respiratory Failure progression."""

    def __init__(
        self,
        patient_id: str,
        duration_hours: int = 6,
        variant: str = "type_i",
        trigger_factor: str = "pneumonia",
        sample_rate_minutes: int = 5
    ):
        """Initialize Respiratory Failure generator."""
        if variant not in ['type_i', 'type_ii']:
            raise ValueError("Variant must be 'type_i' or 'type_ii'")

        super().__init__(
            patient_id=patient_id,
            duration_hours=duration_hours,
            severity="respiratory_failure",
            sample_rate_minutes=sample_rate_minutes
        )

        self.variant = variant
        self.trigger_factor = trigger_factor

        # Progression parameters
        if variant == 'type_i':
            # Type I: Severe hypoxemia (SpO2 drops significantly)
            self.spo2_target = 75  # Final SpO2
            self.rr_target = 35  # Final RR
            self.etco2_target = 50  # Final ETCO2
            self.ph_target = 7.20  # Final pH
        else:
            # Type II: Moderate hypoxemia, severe CO2 retention
            self.spo2_target = 85
            self.rr_target = 32
            self.etco2_target = 70  # HIGH CO2
            self.ph_target = 7.25

        self.baseline_vitals['etco2'] = 40
        self.baseline_vitals['paco2'] = 40
        self.baseline_vitals['work_of_breathing'] = 1.0

    def generate(self) -> pd.DataFrame:
        """Generate 6-hour Respiratory Failure progression."""
        print(f"🫁 Generating RESPIRATORY FAILURE ({self.variant})...")

        for sample_idx in range(self.total_samples):
            progress = self._get_time_progress(sample_idx)
            current_hour = (sample_idx * self.duration_hours * 60 / self.total_samples) / 60
            timestamp = f"T+{int(current_hour)}h"

            current_vitals = self.baseline_vitals.copy()

            # === LINEAR PROGRESSION TOWARDS TARGET VALUES ===

            # SpO2 - Linear decrease from baseline to target
            current_vitals['spo2'] = self.baseline_vitals['spo2'] - (self.baseline_vitals['spo2'] - self.spo2_target) * progress

            # RR - Linear increase from baseline to target
            current_vitals['respiratory_rate'] = self.baseline_vitals['respiratory_rate'] + (self.rr_target - self.baseline_vitals['respiratory_rate']) * progress

            # ETCO2 - Linear increase from baseline to target
            current_vitals['etco2'] = self.baseline_vitals['etco2'] + (self.etco2_target - self.baseline_vitals['etco2']) * progress
            current_vitals['paco2'] = current_vitals['etco2']

            # pH - Linear decrease from baseline to target
            current_vitals['ph'] = self.baseline_vitals['ph'] - (self.baseline_vitals['ph'] - self.ph_target) * progress

            # WORK OF BREATHING - Increases with distress
            current_vitals['work_of_breathing'] = 1.0 + (progress * 2.5)

            # HEART RATE - Increases due to hypoxemia
            if self.variant == 'type_i':
                hr_increase = 0.35 * progress
            else:
                hr_increase = 0.20 * progress
            current_vitals['heart_rate'] = self.baseline_vitals['heart_rate'] * (1 + hr_increase)

            # BLOOD PRESSURE - May decrease in severe Type I
            if self.variant == 'type_i' and progress > 0.5:
                bp_drop = (progress - 0.5) * 2 * 0.20
                current_vitals['systolic_bp'] = self.baseline_vitals['systolic_bp'] * (1 - bp_drop)
                current_vitals['diastolic_bp'] = self.baseline_vitals['diastolic_bp'] * (1 - bp_drop)

            # LACTATE - Increases if severe hypoxemia (Type I)
            if self.variant == 'type_i' and progress > 0.3:
                lactate_increase = (progress - 0.3) * 4.0
                current_vitals['lactate'] = self.baseline_vitals['lactate'] + lactate_increase

            # TEMPERATURE - Increases if infection
            if self.trigger_factor in ['pneumonia', 'ards']:
                temp_increase = progress * 1.8
                current_vitals['temperature'] = self.baseline_vitals['temperature'] + temp_increase

            # ADD NOISE
            for vital_name in current_vitals.keys():
                if vital_name not in ['work_of_breathing']:
                    current_vitals[vital_name] = self.add_noise(
                        current_vitals[vital_name],
                        vital_name,
                        noise_factor=0.5
                    )

            # APPLY CORRELATIONS
            self.apply_physiological_correlations(current_vitals)

            self.timestamps.append(timestamp)
            self.vitals_history.append(current_vitals)

        df = pd.DataFrame(self.vitals_history)
        print(f"✅ Generated {len(df)} samples ({self.variant})")

        return df

    def get_respiratory_summary(self) -> dict:
        """Get clinical summary of respiratory failure."""
        if not self.vitals_history:
            return {}

        df = pd.DataFrame(self.vitals_history)

        return {
            'variant': self.variant,
            'trigger_factor': self.trigger_factor,
            'duration_hours': self.duration_hours,
            'initial_spo2': float(df['spo2'].iloc[0]),
            'final_spo2': float(df['spo2'].iloc[-1]),
            'min_spo2': float(df['spo2'].min()),
            'initial_rr': float(df['respiratory_rate'].iloc[0]),
            'final_rr': float(df['respiratory_rate'].iloc[-1]),
            'max_rr': float(df['respiratory_rate'].max()),
            'initial_ph': float(df['ph'].iloc[0]),
            'final_ph': float(df['ph'].iloc[-1]),
            'min_ph': float(df['ph'].min()),
            'initial_etco2': float(df['etco2'].iloc[0]),
            'final_etco2': float(df['etco2'].iloc[-1]),
            'max_etco2': float(df['etco2'].max()),
            'max_lactate': float(df['lactate'].max()) if 'lactate' in df.columns else None,
            'initial_hr': float(df['heart_rate'].iloc[0]),
            'final_hr': float(df['heart_rate'].iloc[-1]),
            'respiratory_acidosis_present': float(df['ph'].iloc[-1]) < 7.35,
            'severe_hypoxemia': float(df['spo2'].min()) < 80,
        }


if __name__ == "__main__":
    print("Virtual ICU - Respiratory Failure Generator")
    print("=" * 70)

    print("\n1. Generating TYPE I RESPIRATORY FAILURE (Hypoxemic - Pneumonia)...")
    gen_type1 = RespiratoryFailureGenerator("RESPFAIL_TYPE1_001", variant="type_i", trigger_factor="pneumonia")
    data_type1 = gen_type1.generate()
    gen_type1.export_to_csv("respiratory_failure_type_i.csv")
    summary_type1 = gen_type1.get_respiratory_summary()
    print(f"   ✅ SpO2: {summary_type1['initial_spo2']:.1f}% → {summary_type1['final_spo2']:.1f}% (min: {summary_type1['min_spo2']:.1f}%)")
    print(f"   ✅ RR: {summary_type1['initial_rr']:.0f} → {summary_type1['final_rr']:.0f} breaths/min (max: {summary_type1['max_rr']:.0f})")
    print(f"   ✅ pH: {summary_type1['initial_ph']:.2f} → {summary_type1['final_ph']:.2f}")
    print(f"   ✅ ETCO2: {summary_type1['initial_etco2']:.0f} → {summary_type1['final_etco2']:.0f} mmHg")
    print(f"   ✅ Severe hypoxemia: {summary_type1['severe_hypoxemia']}")

    print("\n2. Generating TYPE II RESPIRATORY FAILURE (Hypercapnic - COPD)...")
    gen_type2 = RespiratoryFailureGenerator("RESPFAIL_TYPE2_001", variant="type_ii", trigger_factor="copd_exacerbation")
    data_type2 = gen_type2.generate()
    gen_type2.export_to_csv("respiratory_failure_type_ii.csv")
    summary_type2 = gen_type2.get_respiratory_summary()
    print(f"   ✅ SpO2: {summary_type2['initial_spo2']:.1f}% → {summary_type2['final_spo2']:.1f}% (min: {summary_type2['min_spo2']:.1f}%)")
    print(f"   ✅ RR: {summary_type2['initial_rr']:.0f} → {summary_type2['final_rr']:.0f} breaths/min (max: {summary_type2['max_rr']:.0f})")
    print(f"   ✅ pH: {summary_type2['initial_ph']:.2f} → {summary_type2['final_ph']:.2f}")
    print(f"   ✅ ETCO2: {summary_type2['initial_etco2']:.0f} → {summary_type2['final_etco2']:.0f} mmHg")
    print(f"   ✅ Respiratory acidosis: {summary_type2['respiratory_acidosis_present']}")

    print("\n✅ Respiratory Failure scenarios completed!")
