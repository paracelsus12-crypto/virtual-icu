"""
Virtual ICU - Cardiac Arrest Scenario Generator
Fixed version with proper ROSC logic
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ..base_generator import BasePatientGenerator


class CardiacArrestGenerator(BasePatientGenerator):
    """Generate synthetic patient data with Cardiac Arrest progression."""

    def __init__(
        self,
        patient_id: str,
        duration_minutes: int = 10,
        variant: str = "with_rosc",
        cpr_quality: str = "adequate",
        sample_rate_seconds: int = 10
    ):
        """Initialize Cardiac Arrest generator."""
        if variant not in ['with_rosc', 'without_rosc']:
            raise ValueError("Variant must be 'with_rosc' or 'without_rosc'")

        duration_hours = duration_minutes / 60

        super().__init__(
            patient_id=patient_id,
            duration_hours=duration_hours,
            severity="cardiac_arrest",
            sample_rate_minutes=sample_rate_seconds / 60
        )

        self.variant = variant
        self.duration_minutes = duration_minutes
        self.cpr_quality = cpr_quality
        self.total_samples = int((duration_minutes * 60) / sample_rate_seconds)

        self.arrest_params = {
            'with_rosc': {
                'rosc_time_min': 8,
                'rosc_time_max': 10,
                'cpr_effectiveness_factor': 0.3 if cpr_quality == 'poor' else (
                    0.6 if cpr_quality == 'adequate' else 0.9
                ),
            },
            'without_rosc': {
                'progressive_deterioration': True,
                'cpr_effectiveness_factor': 0.1,
            }
        }

        self.baseline_vitals['lactate'] = 1.0
        self.baseline_vitals['ph'] = 7.40
        self.baseline_vitals['etco2'] = 40

    def generate(self) -> pd.DataFrame:
        """Generate 10-minute Cardiac Arrest progression."""
        print(f"🏥 Generating CARDIAC ARREST ({self.variant})...")

        params = self.arrest_params[self.variant]
        rosc_time_samples = None

        if self.variant == 'with_rosc':
            rosc_time_min = params['rosc_time_min']
            rosc_time_max = params['rosc_time_max']
            rosc_time_seconds = np.random.uniform(
                rosc_time_min * 60, rosc_time_max * 60
            )
            rosc_time_samples = int(rosc_time_seconds / (self.duration_minutes * 60 / self.total_samples))
            print(f"   ROSC scheduled at sample {rosc_time_samples} of {self.total_samples}")

        for sample_idx in range(self.total_samples):
            progress = self._get_time_progress(sample_idx)
            current_minute = (sample_idx * self.duration_minutes * 60 / self.total_samples) / 60
            timestamp = f"T+{int(current_minute)}m"

            current_vitals = self.baseline_vitals.copy()

            # PHASE 1: COLLAPSE (0-1 minute) - VF/VT
            if progress < 0.1:
                current_vitals['heart_rate'] = np.random.uniform(100, 300)
                current_vitals['systolic_bp'] = self.baseline_vitals['systolic_bp'] * (1 - progress * 5)
                current_vitals['diastolic_bp'] = self.baseline_vitals['diastolic_bp'] * (1 - progress * 5)
                current_vitals['respiratory_rate'] = 0
                current_vitals['spo2'] = 98 - (progress * 50)

            # PHASE 2: EARLY ARREST (1-3 minutes) - HR=0, BP=0
            elif progress < 0.3:
                current_vitals['heart_rate'] = 0
                current_vitals['systolic_bp'] = 0
                current_vitals['diastolic_bp'] = 0
                current_vitals['respiratory_rate'] = 0
                current_vitals['spo2'] = max(20, 98 - (progress * 100))

                if params['cpr_effectiveness_factor'] > 0.3:
                    cpr_bp = params['cpr_effectiveness_factor'] * 40
                    current_vitals['systolic_bp'] = cpr_bp
                    current_vitals['diastolic_bp'] = cpr_bp * 0.5

            # PHASE 3: LATE ARREST (3-10 minutes) - Recovery or deterioration
            else:  # progress >= 0.3
                # CHECK ROSC
                if self.variant == 'with_rosc' and rosc_time_samples is not None and sample_idx >= rosc_time_samples:
                    # ROSC PHASE - RECOVERY
                    recovery_progress = (sample_idx - rosc_time_samples) / (self.total_samples - rosc_time_samples)

                    if recovery_progress < 0.2:
                        current_vitals['heart_rate'] = 40 + (recovery_progress * 50)
                    else:
                        current_vitals['heart_rate'] = 90 + (recovery_progress * 40)

                    current_vitals['systolic_bp'] = 60 + (recovery_progress * 60)
                    current_vitals['diastolic_bp'] = 40 + (recovery_progress * 40)
                    current_vitals['respiratory_rate'] = max(4, recovery_progress * 20)
                    current_vitals['spo2'] = 40 + (recovery_progress * 55)

                    lactate_increase = params['cpr_effectiveness_factor'] * progress * 15
                    current_vitals['lactate'] = self.baseline_vitals['lactate'] + (lactate_increase * (1 - recovery_progress * 0.5))

                    acidosis = 0.02 * (current_vitals['lactate'] - 1)
                    current_vitals['ph'] = self.baseline_vitals['ph'] - acidosis + (recovery_progress * 0.2)

                    current_vitals['etco2'] = 10 + (recovery_progress * 30)

                else:
                    # NO ROSC - CONTINUING ARREST
                    current_vitals['heart_rate'] = 0
                    current_vitals['systolic_bp'] = params['cpr_effectiveness_factor'] * 30 if params['cpr_effectiveness_factor'] > 0 else 0
                    current_vitals['diastolic_bp'] = current_vitals['systolic_bp'] * 0.5
                    current_vitals['respiratory_rate'] = 0
                    current_vitals['spo2'] = max(10, 20 - (progress * 10))

                    lactate_increase = progress * 20
                    current_vitals['lactate'] = self.baseline_vitals['lactate'] + lactate_increase

                    acidosis = 0.02 * (current_vitals['lactate'] - 1)
                    current_vitals['ph'] = max(6.8, self.baseline_vitals['ph'] - acidosis)

                    current_vitals['etco2'] = max(0, 40 - (progress * 40))

            # ADD NOISE
            for vital_name in current_vitals.keys():
                if vital_name not in ['heart_rate']:
                    current_vitals[vital_name] = self.add_noise(
                        current_vitals[vital_name],
                        vital_name,
                        noise_factor=0.3
                    )

            # APPLY CORRELATIONS
            self.apply_physiological_correlations(current_vitals)

            self.timestamps.append(timestamp)
            self.vitals_history.append(current_vitals)

        df = pd.DataFrame(self.vitals_history)
        print(f"✅ Generated {len(df)} samples ({self.variant})")

        return df

    def get_arrest_summary(self) -> dict:
        """Get clinical summary of cardiac arrest."""
        if not self.vitals_history:
            return {}

        df = pd.DataFrame(self.vitals_history)

        # Find ROSC point
        rosc_index = None
        for idx in range(1, len(df)):
            if df['heart_rate'].iloc[idx] > 50 and df['heart_rate'].iloc[idx-1] <= 50:
                rosc_index = idx
                break

        return {
            'variant': self.variant,
            'cpr_quality': self.cpr_quality,
            'duration_minutes': self.duration_minutes,
            'initial_hr': float(df['heart_rate'].iloc[0]),
            'min_hr': float(df['heart_rate'].min()),
            'max_lactate': float(df['lactate'].max()),
            'min_ph': float(df['ph'].min()),
            'min_spo2': float(df['spo2'].min()),
            'rosc_achieved': rosc_index is not None,
            'rosc_time_minutes': (rosc_index / len(df) * self.duration_minutes) if rosc_index else None,
            'max_etco2': float(df['etco2'].max()),
            'min_etco2': float(df['etco2'].min()),
        }


if __name__ == "__main__":
    print("Virtual ICU - Cardiac Arrest Generator")
    print("=" * 70)

    print("\n1. Generating CARDIAC ARREST WITH ROSC (successful resuscitation)...")
    gen_rosc = CardiacArrestGenerator("ARREST_ROSC_001", variant="with_rosc", cpr_quality="adequate")
    data_rosc = gen_rosc.generate()
    gen_rosc.export_to_csv("cardiac_arrest_with_rosc.csv")
    summary_rosc = gen_rosc.get_arrest_summary()
    print(f"   ✅ ROSC achieved: {summary_rosc['rosc_achieved']}")
    if summary_rosc['rosc_time_minutes']:
        print(f"   ✅ ROSC time: {summary_rosc['rosc_time_minutes']:.1f} minutes")
    print(f"   ✅ Max lactate: {summary_rosc['max_lactate']:.1f} mmol/L")
    print(f"   ✅ Min pH: {summary_rosc['min_ph']:.2f}")
    print(f"   ✅ Min SpO2: {summary_rosc['min_spo2']:.1f}%")

    print("\n2. Generating CARDIAC ARREST WITHOUT ROSC (unsuccessful resuscitation)...")
    gen_no_rosc = CardiacArrestGenerator("ARREST_NOROSC_001", variant="without_rosc", cpr_quality="poor")
    data_no_rosc = gen_no_rosc.generate()
    gen_no_rosc.export_to_csv("cardiac_arrest_without_rosc.csv")
    summary_no_rosc = gen_no_rosc.get_arrest_summary()
    print(f"   ✅ ROSC achieved: {summary_no_rosc['rosc_achieved']}")
    print(f"   ✅ Max lactate: {summary_no_rosc['max_lactate']:.1f} mmol/L")
    print(f"   ✅ Min pH: {summary_no_rosc['min_ph']:.2f}")
    print(f"   ✅ Min SpO2: {summary_no_rosc['min_spo2']:.1f}%")

    print("\n✅ Cardiac Arrest scenarios completed!")
