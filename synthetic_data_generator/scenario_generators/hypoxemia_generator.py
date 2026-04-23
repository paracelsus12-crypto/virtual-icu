"""Virtual ICU - Hypoxemia Generator"""

import numpy as np
import pandas as pd
from ..base_generator import BasePatientGenerator


class HypoxemiaGenerator(BasePatientGenerator):
    """Generate synthetic patient data with Hypoxemia progression."""

    def __init__(self, patient_id: str, duration_hours: int = 4, variant: str = "acute", sample_rate_minutes: int = 5):
        if variant not in ['acute', 'gradual']:
            raise ValueError("Variant must be 'acute' or 'gradual'")

        super().__init__(patient_id=patient_id, duration_hours=duration_hours, severity="hypoxemia", sample_rate_minutes=sample_rate_minutes)
        
        self.variant = variant
        self.baseline_vitals['confusion'] = 0

    def generate(self) -> pd.DataFrame:
        """Generate 4-hour Hypoxemia progression."""
        print(f"💨 Generating HYPOXEMIA ({self.variant})...")

        for sample_idx in range(self.total_samples):
            progress = self._get_time_progress(sample_idx)
            current_hour = (sample_idx * self.duration_hours * 60 / self.total_samples) / 60
            timestamp = f"T+{int(current_hour)}h"

            current_vitals = self.baseline_vitals.copy()

            # SpO2 - Progressive decrease
            if self.variant == 'acute':
                # Rapid drop
                if progress < 0.2:
                    spo2_target = self.baseline_vitals['spo2']
                else:
                    spo2_target = 70
                    progress_adj = (progress - 0.2) / 0.8
                    current_vitals['spo2'] = self.baseline_vitals['spo2'] - (self.baseline_vitals['spo2'] - spo2_target) * progress_adj
            else:
                # Gradual drop
                spo2_target = 70
                current_vitals['spo2'] = self.baseline_vitals['spo2'] - (self.baseline_vitals['spo2'] - spo2_target) * progress

            if self.variant != 'acute' or progress >= 0.2:
                current_vitals['spo2'] = self.baseline_vitals['spo2'] - (self.baseline_vitals['spo2'] - 70) * (progress if self.variant == 'gradual' else min(1, (progress - 0.2) / 0.8))

            # HR - Increases (compensatory)
            hr_increase = 0.40 * progress * 70
            current_vitals['heart_rate'] = self.baseline_vitals['heart_rate'] + hr_increase

            # RR - Increases
            rr_increase = 0.30 * progress * 20
            current_vitals['respiratory_rate'] = self.baseline_vitals['respiratory_rate'] + rr_increase

            # BP - May drop if severe
            if progress > 0.6:
                bp_drop = (progress - 0.6) * 2 * 0.15
                current_vitals['systolic_bp'] = self.baseline_vitals['systolic_bp'] * (1 - bp_drop)

            # Mental status - Deteriorates with hypoxemia
            if current_vitals['spo2'] < 85:
                current_vitals['confusion'] = min(3, (85 - current_vitals['spo2']) / 5)

            # Cyanosis indicator (visual sign of severe hypoxemia)
            current_vitals['cyanosis_present'] = current_vitals['spo2'] < 80

            # Add noise
            for vital_name in current_vitals.keys():
                if vital_name not in ['confusion', 'cyanosis_present']:
                    current_vitals[vital_name] = self.add_noise(current_vitals[vital_name], vital_name, noise_factor=0.4)

            self.apply_physiological_correlations(current_vitals)
            self.timestamps.append(timestamp)
            self.vitals_history.append(current_vitals)

        df = pd.DataFrame(self.vitals_history)
        print(f"✅ Generated {len(df)} samples ({self.variant})")
        return df

    def get_hypoxemia_summary(self) -> dict:
        if not self.vitals_history:
            return {}
        df = pd.DataFrame(self.vitals_history)
        return {
            'variant': self.variant,
            'initial_spo2': float(df['spo2'].iloc[0]),
            'final_spo2': float(df['spo2'].iloc[-1]),
            'min_spo2': float(df['spo2'].min()),
            'initial_hr': float(df['heart_rate'].iloc[0]),
            'final_hr': float(df['heart_rate'].iloc[-1]),
            'max_confusion': float(df['confusion'].max()),
            'cyanosis_present': bool(df['cyanosis_present'].any()),
            'severe_hypoxemia': float(df['spo2'].min()) < 75,
        }


if __name__ == "__main__":
    print("Virtual ICU - Hypoxemia Generator\n" + "=" * 70)

    print("\n1. Generating ACUTE HYPOXEMIA...")
    gen_acute = HypoxemiaGenerator("HYPOX_ACUTE_001", variant="acute")
    data_acute = gen_acute.generate()
    gen_acute.export_to_csv("hypoxemia_acute.csv")
    summary_acute = gen_acute.get_hypoxemia_summary()
    print(f"   ✅ SpO2: {summary_acute['initial_spo2']:.1f}% → {summary_acute['final_spo2']:.1f}% (min: {summary_acute['min_spo2']:.1f}%)")
    print(f"   ✅ HR: {summary_acute['initial_hr']:.0f} → {summary_acute['final_hr']:.0f} bpm")
    print(f"   ✅ Confusion level: {summary_acute['max_confusion']:.1f}")
    print(f"   ✅ Cyanosis: {summary_acute['cyanosis_present']}")

    print("\n2. Generating GRADUAL HYPOXEMIA...")
    gen_grad = HypoxemiaGenerator("HYPOX_GRAD_001", variant="gradual")
    data_grad = gen_grad.generate()
    gen_grad.export_to_csv("hypoxemia_gradual.csv")
    summary_grad = gen_grad.get_hypoxemia_summary()
    print(f"   ✅ SpO2: {summary_grad['initial_spo2']:.1f}% → {summary_grad['final_spo2']:.1f}% (min: {summary_grad['min_spo2']:.1f}%)")
    print(f"   ✅ HR: {summary_grad['initial_hr']:.0f} → {summary_grad['final_hr']:.0f} bpm")
    print(f"   ✅ Confusion level: {summary_grad['max_confusion']:.1f}")
    print(f"   ✅ Cyanosis: {summary_grad['cyanosis_present']}")

    print("\n✅ Hypoxemia scenarios completed!")
