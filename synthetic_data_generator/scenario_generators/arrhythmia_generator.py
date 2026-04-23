"""Virtual ICU - Arrhythmia Generator"""

import numpy as np
import pandas as pd
from ..base_generator import BasePatientGenerator


class ArrhythmiaGenerator(BasePatientGenerator):
    """Generate synthetic patient data with Arrhythmia progression.
    
    Types:
    - AFib: Atrial Fibrillation (irregular HR 100-160)
    - VT: Ventricular Tachycardia (regular fast 140-200)
    - SVT: Supraventricular Tachycardia (regular fast 150-250)
    - Bradycardia: Slow HR (<60)
    """

    def __init__(self, patient_id: str, duration_hours: int = 3, arrhythmia_type: str = "afib", sample_rate_minutes: int = 2):
        if arrhythmia_type not in ['afib', 'vt', 'svt', 'bradycardia']:
            raise ValueError("Type must be afib, vt, svt, or bradycardia")

        super().__init__(patient_id=patient_id, duration_hours=duration_hours, severity="arrhythmia", sample_rate_minutes=sample_rate_minutes)
        
        self.arrhythmia_type = arrhythmia_type

    def generate(self) -> pd.DataFrame:
        """Generate 3-hour Arrhythmia progression."""
        print(f"⚡ Generating ARRHYTHMIA ({self.arrhythmia_type.upper()})...")

        for sample_idx in range(self.total_samples):
            progress = self._get_time_progress(sample_idx)
            current_min = (sample_idx * self.duration_hours * 60 / self.total_samples)
            timestamp = f"T+{int(current_min)}m"

            current_vitals = self.baseline_vitals.copy()

            # HR - Based on arrhythmia type
            if self.arrhythmia_type == 'afib':
                # Atrial fibrillation - irregular, 100-160 bpm
                base_hr = 100 + progress * 40
                current_vitals['heart_rate'] = base_hr + np.random.uniform(-15, 15)
                current_vitals['rhythm_regular'] = False

            elif self.arrhythmia_type == 'vt':
                # Ventricular tachycardia - regular, 140-200 bpm
                current_vitals['heart_rate'] = 140 + progress * 40
                current_vitals['rhythm_regular'] = True
                current_vitals['hemodynamic_instability'] = progress > 0.5

            elif self.arrhythmia_type == 'svt':
                # Supraventricular tachycardia - regular, 150-250 bpm
                current_vitals['heart_rate'] = 150 + progress * 80
                current_vitals['rhythm_regular'] = True
                current_vitals['palpitations'] = True

            else:  # bradycardia
                # Bradycardia - slow, <60 bpm
                current_vitals['heart_rate'] = 60 - progress * 30
                current_vitals['rhythm_regular'] = True

            # BP - May drop with fast arrhythmias
            if self.arrhythmia_type in ['vt', 'svt'] and progress > 0.4:
                bp_drop = (progress - 0.4) * 2 * 0.20
                current_vitals['systolic_bp'] = self.baseline_vitals['systolic_bp'] * (1 - bp_drop)
                current_vitals['diastolic_bp'] = self.baseline_vitals['diastolic_bp'] * (1 - bp_drop)

            elif self.arrhythmia_type == 'bradycardia' and progress > 0.5:
                bp_drop = (progress - 0.5) * 2 * 0.15
                current_vitals['systolic_bp'] = self.baseline_vitals['systolic_bp'] * (1 - bp_drop)

            # RR - May increase with symptoms
            if self.arrhythmia_type in ['vt', 'svt']:
                rr_increase = progress * 10
                current_vitals['respiratory_rate'] = self.baseline_vitals['respiratory_rate'] + rr_increase

            # SpO2 - May drop if severe/prolonged
            if progress > 0.6 and self.arrhythmia_type in ['vt', 'svt']:
                spo2_drop = (progress - 0.6) * 2 * 15
                current_vitals['spo2'] = self.baseline_vitals['spo2'] - spo2_drop

            # Symptoms
            current_vitals['chest_pain'] = self.arrhythmia_type in ['vt', 'svt'] and progress > 0.3
            current_vitals['dyspnea'] = self.arrhythmia_type in ['vt', 'svt'] and progress > 0.4
            current_vitals['dizziness'] = progress > 0.5

            # Add noise
            for vital_name in current_vitals.keys():
                if vital_name not in ['rhythm_regular', 'hemodynamic_instability', 'palpitations', 'chest_pain', 'dyspnea', 'dizziness']:
                    current_vitals[vital_name] = self.add_noise(current_vitals[vital_name], vital_name, noise_factor=0.3)

            self.apply_physiological_correlations(current_vitals)
            self.timestamps.append(timestamp)
            self.vitals_history.append(current_vitals)

        df = pd.DataFrame(self.vitals_history)
        print(f"✅ Generated {len(df)} samples ({self.arrhythmia_type})")
        return df

    def get_arrhythmia_summary(self) -> dict:
        if not self.vitals_history:
            return {}
        df = pd.DataFrame(self.vitals_history)
        return {
            'type': self.arrhythmia_type,
            'initial_hr': float(df['heart_rate'].iloc[0]),
            'final_hr': float(df['heart_rate'].iloc[-1]),
            'max_hr': float(df['heart_rate'].max()),
            'min_hr': float(df['heart_rate'].min()),
            'mean_hr': float(df['heart_rate'].mean()),
            'min_sbp': float(df['systolic_bp'].min()),
            'hemodynamic_compromise': float(df['systolic_bp'].min()) < 90,
        }


if __name__ == "__main__":
    print("Virtual ICU - Arrhythmia Generator\n" + "=" * 70)

    types = ['afib', 'vt', 'svt', 'bradycardia']
    for i, arr_type in enumerate(types, 1):
        print(f"\n{i}. Generating {arr_type.upper()}...")
        gen = ArrhythmiaGenerator(f"ARR_{arr_type.upper()}_001", arrhythmia_type=arr_type)
        data = gen.generate()
        gen.export_to_csv(f"arrhythmia_{arr_type}.csv")
        summary = gen.get_arrhythmia_summary()
        print(f"   ✅ HR: {summary['initial_hr']:.0f} → {summary['final_hr']:.0f} (max: {summary['max_hr']:.0f}, mean: {summary['mean_hr']:.0f})")
        print(f"   ✅ Min SBP: {summary['min_sbp']:.0f} mmHg")
        print(f"   ✅ Hemodynamic compromise: {summary['hemodynamic_compromise']}")

    print("\n✅ Arrhythmia scenarios completed!")
