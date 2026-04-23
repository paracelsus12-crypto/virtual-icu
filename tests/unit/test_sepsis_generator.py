"""Unit tests for SepsisGenerator (Sepsis-3 Compliant)"""

import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from synthetic_data_generator.scenario_generators.sepsis_generator import SepsisGenerator, SepsisSOFACalculator


class TestSepsisGeneratorBasics:
    """Test basic functionality"""
    
    def test_initialization_sepsis(self):
        """Test SEPSIS variant initialization"""
        gen = SepsisGenerator("P001", duration_hours=8, variant="sepsis")
        assert gen.patient_id == "P001"
        assert gen.duration_hours == 8
        assert gen.variant == "sepsis"
        assert gen.total_samples == 480
    
    def test_initialization_septic_shock(self):
        """Test SEPTIC SHOCK variant initialization"""
        gen = SepsisGenerator("P002", duration_hours=8, variant="septic_shock")
        assert gen.variant == "septic_shock"
    
    def test_invalid_variant(self):
        """Test that invalid variant raises error"""
        with pytest.raises(ValueError):
            SepsisGenerator("P001", variant="invalid")
    
    def test_generation_sepsis(self):
        """Test SEPSIS data generation"""
        gen = SepsisGenerator("P001", variant="sepsis")
        data = gen.generate()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 480
        assert 'heart_rate' in data.columns
        assert 'lactate' in data.columns
        assert 'sofa' not in data.columns  # SOFA calculated, not stored
    
    def test_generation_septic_shock(self):
        """Test SEPTIC SHOCK data generation"""
        gen = SepsisGenerator("P001", variant="septic_shock")
        data = gen.generate()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 480
        assert 'vasopressor_required' in data.columns


class TestSepsisSOFACalculator:
    """Test SOFA score calculation"""
    
    def test_normal_sofa_score(self):
        """Test SOFA calculation for normal patient"""
        vitals = {
            'spo2': 98,
            'platelets': 250,
            'bilirubin': 1.0,
            'map': 85,
            'gcs': 15,
            'creatinine': 1.0,
            'vasopressor_required': False,
        }
        score = SepsisSOFACalculator.calculate_sofa(vitals)
        assert score <= 2  # Normal should be 0-2
    
    def test_sepsis_sofa_score(self):
        """Test SOFA calculation for sepsis patient"""
        vitals = {
            'spo2': 90,
            'platelets': 100,
            'bilirubin': 2.0,
            'map': 75,
            'gcs': 13,
            'creatinine': 1.5,
            'vasopressor_required': False,
        }
        score = SepsisSOFACalculator.calculate_sofa(vitals)
        assert score >= 2  # Sepsis criteria


class TestSepsisClinicalProgression:
    """Test clinical accuracy of Sepsis-3 progression"""
    
    def test_sepsis_lactate_increases(self):
        """Lactate should increase during sepsis"""
        gen = SepsisGenerator("P001", variant="sepsis")
        data = gen.generate()
        
        initial_lactate = data['lactate'].iloc[0]
        final_lactate = data['lactate'].iloc[-1]
        
        assert final_lactate > initial_lactate, "Lactate should increase"
    
    def test_sepsis_sofa_increases(self):
        """SOFA components should show organ dysfunction"""
        gen = SepsisGenerator("P001", variant="sepsis")
        data = gen.generate()
        
        # Calculate initial and final SOFA
        initial_sofa = SepsisSOFACalculator.calculate_sofa(data.iloc[0].to_dict())
        final_sofa = SepsisSOFACalculator.calculate_sofa(data.iloc[-1].to_dict())
        sofa_increase = final_sofa - initial_sofa
        
        assert sofa_increase >= 2, "SOFA should increase by at least 2 (Sepsis-3 criteria)"
    
    def test_septic_shock_vasopressor_required(self):
        """SEPTIC SHOCK should require vasopressors"""
        gen = SepsisGenerator("P001", variant="septic_shock")
        data = gen.generate()
        
        # Check if vasopressor is required at any point
        has_vasopressor = data['vasopressor_required'].any()
        assert has_vasopressor, "Septic Shock should require vasopressors"
    
    def test_septic_shock_high_lactate(self):
        """SEPTIC SHOCK should have lactate > 2"""
        gen = SepsisGenerator("P001", variant="septic_shock")
        data = gen.generate()
        
        max_lactate = data['lactate'].max()
        assert max_lactate > 2.0, "Septic Shock should have lactate > 2 mmol/L"
    
    def test_septic_shock_low_map(self):
        """SEPTIC SHOCK should show low MAP"""
        gen = SepsisGenerator("P001", variant="septic_shock")
        data = gen.generate()
        
        min_map = data['map'].min()
        assert min_map < 75, "Septic Shock should show low MAP"
    
    def test_septic_shock_organ_dysfunction(self):
        """SEPTIC SHOCK should show organ dysfunction markers"""
        gen = SepsisGenerator("P001", variant="septic_shock")
        data = gen.generate()
        
        # Check GCS decreases (CNS dysfunction)
        initial_gcs = data['gcs'].iloc[0]
        final_gcs = data['gcs'].iloc[-1]
        assert final_gcs < initial_gcs, "GCS should decrease (CNS dysfunction)"
        
        # Check creatinine increases (renal dysfunction)
        initial_creat = data['creatinine'].iloc[0]
        final_creat = data['creatinine'].iloc[-1]
        assert final_creat > initial_creat, "Creatinine should increase (renal dysfunction)"


class TestDataQuality:
    """Test data quality and validity"""
    
    def test_no_missing_values(self):
        """Generated data should have no missing values"""
        gen = SepsisGenerator("P001", variant="sepsis")
        data = gen.generate()
        
        assert data.isnull().sum().sum() == 0, "No missing values allowed"
    
    def test_vital_signs_in_range(self):
        """All vital signs should stay within physiological ranges"""
        gen = SepsisGenerator("P001", variant="septic_shock")
        data = gen.generate()
        
        # Heart rate: 40-180 bpm
        assert (data['heart_rate'] >= 40).all()
        assert (data['heart_rate'] <= 180).all()
        
        # SpO2: 70-100%
        assert (data['spo2'] >= 70).all()
        assert (data['spo2'] <= 100).all()
        
        # Temperature: 32-42°C
        assert (data['temperature'] >= 32).all()
        assert (data['temperature'] <= 42).all()
        
        # pH: 6.8-7.8
        assert (data['ph'] >= 6.8).all()
        assert (data['ph'] <= 7.8).all()
        
        # Lactate: 0.5-20 mmol/L
        assert (data['lactate'] >= 0.5).all()
        assert (data['lactate'] <= 20).all()
    
    def test_csv_export_sepsis(self, tmp_path):
        """Test CSV export for SEPSIS"""
        gen = SepsisGenerator("P001", variant="sepsis")
        data = gen.generate()
        
        filepath = tmp_path / "test_sepsis.csv"
        gen.export_to_csv(str(filepath))
        
        assert filepath.exists()
        exported_data = pd.read_csv(filepath)
        assert len(exported_data) == len(data)
    
    def test_json_export_septic_shock(self, tmp_path):
        """Test JSON export for SEPTIC SHOCK"""
        gen = SepsisGenerator("P001", variant="septic_shock")
        data = gen.generate()
        
        filepath = tmp_path / "test_septic_shock.json"
        gen.export_to_json(str(filepath))
        
        assert filepath.exists()
    
    def test_sepsis3_summary(self):
        """Test Sepsis-3 compliant summary generation"""
        gen = SepsisGenerator("P001", variant="sepsis")
        data = gen.generate()
        
        summary = gen.get_sepsis3_summary()
        
        assert 'sepsis3_compliant' in summary
        assert summary['sepsis3_compliant'] == True
        assert 'sofa_increase' in summary
        assert 'meets_sepsis_criteria' in summary
        assert summary['meets_sepsis_criteria'] == True
        assert summary['final_lactate'] > summary['initial_lactate']


class TestVariantDifferences:
    """Test differences between SEPSIS and SEPTIC SHOCK variants"""
    
    def test_septic_shock_more_severe_than_sepsis(self):
        """SEPTIC SHOCK should have more severe progression than SEPSIS"""
        gen_sepsis = SepsisGenerator("P001", variant="sepsis")
        gen_shock = SepsisGenerator("P002", variant="septic_shock")
        
        data_sepsis = gen_sepsis.generate()
        data_shock = gen_shock.generate()
        
        # SEPTIC SHOCK should have higher final lactate
        lactate_increase_sepsis = data_sepsis['lactate'].iloc[-1] - data_sepsis['lactate'].iloc[0]
        lactate_increase_shock = data_shock['lactate'].iloc[-1] - data_shock['lactate'].iloc[0]
        
        assert lactate_increase_shock > lactate_increase_sepsis * 1.5, "Septic Shock should have steeper lactate increase"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
