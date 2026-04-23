"""Unit tests for CardiacArrestGenerator"""

import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from synthetic_data_generator.scenario_generators.cardiac_arrest_generator import CardiacArrestGenerator


class TestCardiacArrestGeneratorBasics:
    """Test basic functionality"""
    
    def test_initialization_with_rosc(self):
        """Test WITH_ROSC variant initialization"""
        gen = CardiacArrestGenerator("P001", duration_minutes=10, variant="with_rosc")
        assert gen.patient_id == "P001"
        assert gen.duration_minutes == 10
        assert gen.variant == "with_rosc"
        assert gen.total_samples == 60  # 10 minutes * 60 seconds / 10 seconds per sample
    
    def test_initialization_without_rosc(self):
        """Test WITHOUT_ROSC variant initialization"""
        gen = CardiacArrestGenerator("P002", variant="without_rosc")
        assert gen.variant == "without_rosc"
    
    def test_invalid_variant(self):
        """Test that invalid variant raises error"""
        with pytest.raises(ValueError):
            CardiacArrestGenerator("P001", variant="invalid")
    
    def test_cpr_quality_options(self):
        """Test different CPR quality levels"""
        gen_poor = CardiacArrestGenerator("P001", cpr_quality="poor")
        gen_adequate = CardiacArrestGenerator("P002", cpr_quality="adequate")
        gen_excellent = CardiacArrestGenerator("P003", cpr_quality="excellent")
        
        assert gen_poor.cpr_quality == "poor"
        assert gen_adequate.cpr_quality == "adequate"
        assert gen_excellent.cpr_quality == "excellent"
    
    def test_generation_with_rosc(self):
        """Test data generation WITH ROSC"""
        gen = CardiacArrestGenerator("P001", variant="with_rosc")
        data = gen.generate()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 60
        assert 'heart_rate' in data.columns
        assert 'lactate' in data.columns
        assert 'etco2' in data.columns
    
    def test_generation_without_rosc(self):
        """Test data generation WITHOUT ROSC"""
        gen = CardiacArrestGenerator("P001", variant="without_rosc")
        data = gen.generate()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 60


class TestCardiacArrestClinicalProgression:
    """Test clinical accuracy of cardiac arrest"""
    
    def test_with_rosc_rosc_achieved(self):
        """WITH_ROSC should achieve ROSC"""
        gen = CardiacArrestGenerator("P001", variant="with_rosc", cpr_quality="adequate")
        data = gen.generate()
        summary = gen.get_arrest_summary()
        
        assert summary['rosc_achieved'] == True, "WITH_ROSC should achieve ROSC"
    
    def test_with_rosc_rosc_time_realistic(self):
        """ROSC time should be between 8-10 minutes"""
        gen = CardiacArrestGenerator("P001", variant="with_rosc")
        data = gen.generate()
        summary = gen.get_arrest_summary()
        
        assert summary['rosc_time_minutes'] is not None
        assert 8 <= summary['rosc_time_minutes'] <= 10, "ROSC should occur at 8-10 minutes"
    
    def test_without_rosc_no_rosc(self):
        """WITHOUT_ROSC should NOT achieve ROSC"""
        gen = CardiacArrestGenerator("P001", variant="without_rosc")
        data = gen.generate()
        summary = gen.get_arrest_summary()
        
        assert summary['rosc_achieved'] == False, "WITHOUT_ROSC should not achieve ROSC"
    
    def test_hr_drops_to_zero(self):
        """HR should drop to ~0 during arrest"""
        gen = CardiacArrestGenerator("P001", variant="without_rosc")
        data = gen.generate()
        
        # Check that HR reaches very low values during arrest
        min_hr = data['heart_rate'].min()
        assert min_hr <= 50, "HR should drop significantly during arrest"
    
    def test_bp_drops_severely(self):
        """BP should drop to near zero during arrest"""
        gen = CardiacArrestGenerator("P001", variant="without_rosc")
        data = gen.generate()
        
        min_sbp = data['systolic_bp'].min()
        assert min_sbp <= 50, "SBP should drop severely during arrest"
    
    def test_lactate_increases_dramatically(self):
        """Lactate should increase dramatically (anaerobic metabolism)"""
        gen = CardiacArrestGenerator("P001", variant="without_rosc")
        data = gen.generate()
        
        initial_lactate = data['lactate'].iloc[0]
        final_lactate = data['lactate'].iloc[-1]
        
        assert final_lactate > initial_lactate * 10, "Lactate should increase dramatically"
    
    def test_acidosis_develops(self):
        """pH should decrease severely (acidosis)"""
        gen = CardiacArrestGenerator("P001", variant="without_rosc")
        data = gen.generate()
        
        initial_ph = data['ph'].iloc[0]
        final_ph = data['ph'].iloc[-1]
        
        assert final_ph < initial_ph, "pH should decrease (acidosis)"
    
    def test_spo2_drops_critically(self):
        """SpO2 should drop to critical levels"""
        gen = CardiacArrestGenerator("P001", variant="without_rosc")
        data = gen.generate()
        
        min_spo2 = data['spo2'].min()
        assert min_spo2 < 80, "SpO2 should drop critically during arrest"
    
    def test_etco2_decreases(self):
        """ETCO2 should decrease (poor perfusion = no CO2 production)"""
        gen = CardiacArrestGenerator("P001", variant="without_rosc")
        data = gen.generate()
        
        initial_etco2 = data['etco2'].iloc[0]
        final_etco2 = data['etco2'].iloc[-1]
        
        assert final_etco2 < initial_etco2, "ETCO2 should decrease during arrest"
    
    def test_cpr_quality_affects_outcomes(self):
        """Better CPR quality should improve ROSC chances"""
        gen_poor = CardiacArrestGenerator("P001", variant="with_rosc", cpr_quality="poor")
        gen_excellent = CardiacArrestGenerator("P002", variant="with_rosc", cpr_quality="excellent")
        
        # Both should attempt ROSC but excellent CPR has better baseline
        data_poor = gen_poor.generate()
        data_excellent = gen_excellent.generate()
        
        # Excellent CPR should maintain slightly higher BP during arrest
        avg_bp_poor = data_poor['systolic_bp'].iloc[10:30].mean()
        avg_bp_excellent = data_excellent['systolic_bp'].iloc[10:30].mean()
        
        assert avg_bp_excellent >= avg_bp_poor, "Better CPR should maintain higher BP"


class TestDataQuality:
    """Test data quality and validity"""
    
    def test_no_missing_values(self):
        """Generated data should have no missing values"""
        gen = CardiacArrestGenerator("P001", variant="with_rosc")
        data = gen.generate()
        
        assert data.isnull().sum().sum() == 0, "No missing values allowed"
    
    def test_vital_signs_in_range(self):
        """All vital signs should stay within physiological ranges"""
        gen = CardiacArrestGenerator("P001", variant="with_rosc")
        data = gen.generate()
        
        # Heart rate: 0-300 (includes arrest and VF/VT)
        assert (data['heart_rate'] >= 0).all()
        assert (data['heart_rate'] <= 300).all()
        
        # SpO2: 0-100%
        assert (data['spo2'] >= 0).all()
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
    
    def test_csv_export(self, tmp_path):
        """Test CSV export"""
        gen = CardiacArrestGenerator("P001", variant="with_rosc")
        data = gen.generate()
        
        filepath = tmp_path / "test_arrest.csv"
        gen.export_to_csv(str(filepath))
        
        assert filepath.exists()
        exported_data = pd.read_csv(filepath)
        assert len(exported_data) == len(data)
    
    def test_json_export(self, tmp_path):
        """Test JSON export"""
        gen = CardiacArrestGenerator("P001", variant="without_rosc")
        data = gen.generate()
        
        filepath = tmp_path / "test_arrest.json"
        gen.export_to_json(str(filepath))
        
        assert filepath.exists()
    
    def test_arrest_summary(self):
        """Test arrest summary generation"""
        gen = CardiacArrestGenerator("P001", variant="with_rosc")
        data = gen.generate()
        
        summary = gen.get_arrest_summary()
        
        assert 'rosc_achieved' in summary
        assert 'max_lactate' in summary
        assert 'min_ph' in summary
        assert 'rosc_time_minutes' in summary


class TestVariantDifferences:
    """Test differences between variants"""
    
    def test_with_rosc_shows_recovery(self):
        """WITH_ROSC should show HR recovery"""
        gen = CardiacArrestGenerator("P001", variant="with_rosc")
        data = gen.generate()
        
        # Should have high HR at start (VF/VT), drop to near 0, then recover
        first_hr = data['heart_rate'].iloc[0]
        min_hr = data['heart_rate'].min()
        last_hr = data['heart_rate'].iloc[-1]
        
        assert first_hr > 100, "Should start with VF/VT"
        assert min_hr < 50, "Should drop during arrest"
        assert last_hr > min_hr, "Should recover after ROSC"
    
    def test_without_rosc_stays_low(self):
        """WITHOUT_ROSC should stay low throughout"""
        gen = CardiacArrestGenerator("P001", variant="without_rosc")
        data = gen.generate()
        
        # Should have VF/VT at start, then stay at near 0
        avg_hr_mid = data['heart_rate'].iloc[10:].mean()
        
        assert avg_hr_mid < 50, "Should stay low without ROSC"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
