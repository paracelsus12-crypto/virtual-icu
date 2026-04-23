"""
Virtual ICU - Synthetic Data Generator
Base Patient Generator Class

This module provides the foundation for all patient scenario generators.

Author: Virtual ICU Team
Version: 2.0.0-alpha
Date: April 22, 2024
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class BasePatientGenerator(ABC):
    """
    Abstract base class for generating synthetic patient vital sign data.
    """
    
    # Vital sign ranges (min, baseline, max)
    VITAL_SIGN_RANGES = {
        'heart_rate': (40, 70, 180),
        'systolic_bp': (50, 120, 200),
        'diastolic_bp': (30, 80, 130),
        'respiratory_rate': (6, 14, 40),
        'spo2': (70, 98, 100),
        'temperature': (32, 36.8, 42),
        'lactate': (0.5, 1.0, 20),
        'ph': (6.8, 7.40, 7.8),
    }
    
    # Noise levels for each vital sign
    NOISE_LEVELS = {
        'heart_rate': 2,
        'systolic_bp': 3,
        'diastolic_bp': 2,
        'respiratory_rate': 1,
        'spo2': 1,
        'temperature': 0.3,
        'lactate': 0.1,
        'ph': 0.01,
    }
    
    def __init__(
        self,
        patient_id: str,
        duration_hours: int = 8,
        severity: str = "moderate",
        sample_rate_minutes: int = 1
    ):
        """Initialize base patient generator."""
        
        if duration_hours <= 0:
            raise ValueError("Duration must be positive")
            
        self.patient_id = patient_id
        self.duration_hours = duration_hours
        self.severity = severity
        self.sample_rate_minutes = sample_rate_minutes
        self.total_samples = int((duration_hours * 60) / sample_rate_minutes)
        
        # Initialize data containers
        self.baseline_vitals = {}
        self.vitals_history = []
        self.timestamps = []
        self.start_time = datetime.now()
        
        # Initialize baseline vitals
        self._initialize_baseline_vitals()
    
    def _initialize_baseline_vitals(self) -> None:
        """Initialize baseline vital signs with realistic variation."""
        self.baseline_vitals = {
            'heart_rate': 70 + np.random.normal(0, 5),
            'systolic_bp': 120 + np.random.normal(0, 8),
            'diastolic_bp': 80 + np.random.normal(0, 6),
            'respiratory_rate': 14 + np.random.normal(0, 2),
            'spo2': 98 + np.random.normal(0, 1),
            'temperature': 36.8 + np.random.normal(0, 0.2),
            'lactate': 1.0 + np.random.normal(0, 0.1),
            'ph': 7.40 + np.random.normal(0, 0.01),
        }
        
        # Ensure values are within realistic ranges
        self._validate_vitals(self.baseline_vitals)
    
    def _validate_vitals(self, vitals: Dict[str, float]) -> None:
        """Validate vital signs are within physiological ranges."""
        for vital, value in vitals.items():
            if vital in self.VITAL_SIGN_RANGES:
                min_val, _, max_val = self.VITAL_SIGN_RANGES[vital]
                vitals[vital] = np.clip(value, min_val, max_val)
    
    def add_noise(
        self,
        value: float,
        vital_name: str,
        noise_factor: float = 1.0
    ) -> float:
        """Add realistic physiological noise to vital sign value."""
        noise_level = self.NOISE_LEVELS.get(vital_name, 0.5)
        noise = np.random.normal(0, noise_level * noise_factor)
        return value + noise
    
    def apply_physiological_correlations(self, vitals: Dict[str, float]) -> None:
        """
        Ensure physiological correlations between vital signs.
        
        Correlations:
        - Low SpO2 (< 94%) increases HR by 15%
        - Low SBP (< 90) increases HR by 20%
        - Low SpO2 (< 92%) increases RR by 8 breaths/min
        - High lactate (> 3) decreases pH
        - High HR (> 130%) increases RR
        """
        
        # Hypoxemia response: low SpO2 increases HR
        if vitals['spo2'] < 94:
            vitals['heart_rate'] *= 1.15
        if vitals['spo2'] < 92:
            vitals['respiratory_rate'] += 8
        
        # Hypotension response: low BP increases HR
        if vitals['systolic_bp'] < 90:
            vitals['heart_rate'] *= 1.20
        if vitals['systolic_bp'] < 70:
            vitals['respiratory_rate'] += 5
        
        # Acidosis response: high lactate decreases pH
        if vitals['lactate'] > 3:
            lactate_effect = 0.02 * (vitals['lactate'] - 1)
            vitals['ph'] = max(6.8, vitals['ph'] - lactate_effect)
        
        # Tachycardia response: high HR increases RR
        if vitals['heart_rate'] > 130:
            vitals['respiratory_rate'] += 5
        
        # Validate after correlations
        self._validate_vitals(vitals)
    
    def _get_time_progress(self, sample_index: int) -> float:
        """Get normalized progress through simulation (0 to 1)."""
        return min(1.0, sample_index / max(1, self.total_samples - 1))
    
    def get_current_vitals(self, sample_index: int) -> Dict[str, float]:
        """Get current vital signs at specific sample index."""
        # Default: return baseline vitals with noise
        current_vitals = self.baseline_vitals.copy()
        
        for vital_name in current_vitals.keys():
            current_vitals[vital_name] = self.add_noise(
                current_vitals[vital_name],
                vital_name
            )
        
        self.apply_physiological_correlations(current_vitals)
        return current_vitals
    
    def generate(self) -> pd.DataFrame:
        """
        Main generation method - must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement generate() method"
        )
    
    def export_to_csv(self, filepath: str) -> str:
        """Export generated data to CSV file."""
        if not self.vitals_history:
            raise ValueError("No data to export. Call generate() first.")
        
        df = pd.DataFrame(self.vitals_history)
        df.insert(0, 'timestamp', self.timestamps)
        df.insert(1, 'patient_id', self.patient_id)
        
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(filepath, index=False)
        print(f"✅ Exported {len(df)} records to {filepath}")
        
        return str(filepath)
    
    def export_to_json(self, filepath: str) -> str:
        """Export generated data to JSON file."""
        if not self.vitals_history:
            raise ValueError("No data to export. Call generate() first.")
        
        data = {
            'metadata': {
                'patient_id': self.patient_id,
                'duration_hours': self.duration_hours,
                'severity': self.severity,
                'scenario': self.__class__.__name__,
                'generated_at': self.start_time.isoformat(),
                'total_samples': len(self.vitals_history),
                'sample_rate_minutes': self.sample_rate_minutes,
            },
            'baseline_vitals': self.baseline_vitals,
            'vitals_data': []
        }
        
        for i, (timestamp, vitals) in enumerate(
            zip(self.timestamps, self.vitals_history)
        ):
            data['vitals_data'].append({
                'index': i,
                'timestamp': timestamp,
                **vitals
            })
        
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Exported {len(data['vitals_data'])} records to {filepath}")
        
        return str(filepath)
    
    def get_summary_statistics(self) -> Dict:
        """Calculate summary statistics for generated data."""
        if not self.vitals_history:
            raise ValueError("No data. Call generate() first.")
        
        df = pd.DataFrame(self.vitals_history)
        
        summary = {}
        for column in df.columns:
            summary[column] = {
                'min': float(df[column].min()),
                'max': float(df[column].max()),
                'mean': float(df[column].mean()),
                'std': float(df[column].std()),
            }
        
        return summary


class DataValidator:
    """Utility class for validating synthetic data quality."""
    
    @staticmethod
    def validate_dataset(df: pd.DataFrame) -> Dict[str, any]:
        """Validate synthetic dataset for quality and completeness."""
        report = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'warnings': [],
            'valid': True,
        }
        
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            report['warnings'].append("Found missing values")
            report['valid'] = False
        
        # Check vital sign ranges
        vital_ranges = BasePatientGenerator.VITAL_SIGN_RANGES
        for vital, (min_val, _, max_val) in vital_ranges.items():
            if vital in df.columns:
                if (df[vital] < min_val).any() or (df[vital] > max_val).any():
                    report['warnings'].append(
                        f"WARNING: {vital} out of range: [{min_val}, {max_val}]"
                    )
                    report['valid'] = False
        
        return report


if __name__ == "__main__":
    print("Virtual ICU - Synthetic Data Generator")
    print("=" * 50)
    print("Base Generator module loaded successfully")
