"""Unit tests for Clinical Scoring Systems"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from synthetic_data_generator.clinical_scorer import NEWS2Calculator, qSOFACalculator, CARTCalculator, ClinicalRecommendations


class TestNEWS2:
    """Test NEWS2 Calculator"""
    
    def test_normal_vitals(self):
        """Normal vitals should give low score"""
        vitals = {
            'respiratory_rate': 16,
            'spo2': 98,
            'temperature': 37.0,
            'systolic_bp': 120,
            'heart_rate': 75,
            'alert_status': 'Alert',
            'supplemental_oxygen': False
        }
        score = NEWS2Calculator.calculate(vitals)
        assert score['total'] <= 4
        assert score['risk_level'] == 'Low'
    
    def test_septic_patient(self):
        """Septic patient should give high score"""
        vitals = {
            'respiratory_rate': 28,
            'spo2': 92,
            'temperature': 38.5,
            'systolic_bp': 95,
            'heart_rate': 120,
            'alert_status': 'Alert',
            'supplemental_oxygen': True
        }
        score = NEWS2Calculator.calculate(vitals)
        assert score['total'] >= 7
        assert score['risk_level'] == 'High'
    
    def test_score_components(self):
        """Test individual component scoring"""
        vitals = {
            'respiratory_rate': 16,
            'spo2': 98,
            'temperature': 37.0,
            'systolic_bp': 120,
            'heart_rate': 75,
            'alert_status': 'Alert',
            'supplemental_oxygen': False
        }
        score = NEWS2Calculator.calculate(vitals)
        assert 'components' in score
        assert len(score['components']) == 7


class TestqSOFA:
    """Test qSOFA Calculator"""
    
    def test_normal_patient(self):
        """Normal patient should score <2"""
        vitals = {
            'alert_status': 'Alert',
            'systolic_bp': 120,
            'respiratory_rate': 16
        }
        score = qSOFACalculator.calculate(vitals)
        assert score['total'] < 2
        assert not score['needs_investigation']
    
    def test_sepsis_criteria(self):
        """Patient meeting sepsis criteria should score >=2"""
        vitals = {
            'alert_status': 'Confused',
            'systolic_bp': 95,
            'respiratory_rate': 22
        }
        score = qSOFACalculator.calculate(vitals)
        assert score['total'] >= 2
        assert score['needs_investigation']
    
    def test_individual_components(self):
        """Test each qSOFA component"""
        # Low BP alone
        vitals_bp = {
            'alert_status': 'Alert',
            'systolic_bp': 95,
            'respiratory_rate': 16
        }
        assert qSOFACalculator.calculate(vitals_bp)['components']['low_bp'] == 1
        
        # High RR alone
        vitals_rr = {
            'alert_status': 'Alert',
            'systolic_bp': 120,
            'respiratory_rate': 25
        }
        assert qSOFACalculator.calculate(vitals_rr)['components']['high_rr'] == 1


class TestCART:
    """Test CART Calculator"""
    
    def test_low_risk_young(self):
        """Young healthy patient should be low risk"""
        vitals = {
            'age': 35,
            'systolic_bp': 130,
            'heart_rate': 70,
            'respiratory_rate': 14,
            'alert_status': 'Alert'
        }
        score = CARTCalculator.calculate(vitals)
        assert score['risk_category'] == 'Low'
    
    def test_high_risk_elderly(self):
        """Elderly patient with abnormal vitals should be high risk"""
        vitals = {
            'age': 75,
            'systolic_bp': 85,
            'heart_rate': 115,
            'respiratory_rate': 28,
            'alert_status': 'Confused'
        }
        score = CARTCalculator.calculate(vitals)
        assert score['risk_category'] in ['Medium', 'High', 'Highest']
    
    def test_unconscious_highest_risk(self):
        """Unconscious patient should be highest risk"""
        vitals = {
            'age': 60,
            'systolic_bp': 100,
            'heart_rate': 100,
            'respiratory_rate': 20,
            'alert_status': 'Unresponsive'
        }
        score = CARTCalculator.calculate(vitals)
        # Should have high score due to consciousness
        assert score['total'] > 10


class TestRecommendations:
    """Test Clinical Recommendations"""
    
    def test_high_risk_patient(self):
        """High risk patient should get urgent recommendations"""
        news2 = {'risk_level': 'High', 'total': 12}
        qsofa = {'needs_investigation': True, 'total': 2}
        cart = {'risk_category': 'High', 'total': 16}
        
        rec = ClinicalRecommendations.get_recommendations(news2, qsofa, cart)
        assert rec['urgency'] == 'High'
        assert len(rec['recommendations']) > 0
        assert any('physician' in r.lower() for r in rec['recommendations'])
    
    def test_low_risk_patient(self):
        """Low risk patient should get minimal recommendations"""
        news2 = {'risk_level': 'Low', 'total': 2}
        qsofa = {'needs_investigation': False, 'total': 0}
        cart = {'risk_category': 'Low', 'total': 8}
        
        rec = ClinicalRecommendations.get_recommendations(news2, qsofa, cart)
        assert rec['urgency'] == 'Low'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
