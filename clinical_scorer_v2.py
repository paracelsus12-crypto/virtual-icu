"""
Virtual ICU v2 - Enhanced Clinical Scoring
Покращена реалізація з типізацією, валідацією та UK RCP guidelines
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class NEWS2Scale(Enum):
    """NEWS2 Scale versions"""
    STANDARD = "standard"
    HOCNL = "hocnl"  # High O2 Need - for COPD/hypercapnia


@dataclass
class NEWS2ComponentScore:
    """NEWS2 component score"""
    name: str
    value: int
    parameter_value: float
    unit: str


class NEWS2CalculatorV2:
    """National Early Warning Score 2 (UK RCP) - Enhanced Version"""
    
    # UK RCP Guidelines thresholds
    RR_THRESHOLDS = {
        'critical_low': (0, 8),
        'abnormal_low': (9, 11),
        'normal': (12, 20),
        'abnormal_high': (21, 24),
        'critical_high': (25, 100),
    }
    
    SLO2_THRESHOLDS_SCALE1 = {
        'critical': (0, 91),
        'abnormal': (92, 93),
        'marginal': (94, 95),
        'normal': (96, 100),
    }
    
    SLO2_THRESHOLDS_SCALE2 = {
        # For patients with HOCNL (COPD, chronic hypercapnia)
        'critical': (0, 87),
        'abnormal': (88, 92),
        'marginal': (93, 94),
        'normal': (95, 100),
    }
    
    TEMP_THRESHOLDS = {
        'critical_low': (0, 35.0),
        'abnormal_low': (35.1, 36.0),
        'normal': (36.1, 38.0),
        'abnormal_high': (38.1, 39.0),
        'critical_high': (39.1, 50),
    }
    
    SBP_THRESHOLDS = {
        'critical_low': (0, 90),
        'abnormal_low': (91, 100),
        'marginal': (101, 110),
        'normal': (111, 219),
        'abnormal_high': (220, 500),
    }
    
    HR_THRESHOLDS = {
        'critical_low': (0, 40),
        'abnormal_low': (41, 50),
        'normal': (51, 90),
        'abnormal_high': (91, 110),
        'critical_high': (111, 500),
    }
    
    @classmethod
    def calculate(cls, vitals: Dict, scale: NEWS2Scale = NEWS2Scale.STANDARD) -> Dict:
        """
        Calculate NEWS2 score with UK RCP guidelines
        
        Args:
            vitals: Dictionary with vital signs
            scale: NEWS2 scale (standard or HOCNL for COPD)
        
        Returns:
            Dictionary with score, components, risk level
        """
        components = {}
        total_score = 0
        
        # 1. Respiratory Rate (0-3)
        rr = vitals.get('respiratory_rate')
        try:
            rr = float(rr) if rr is not None else None
        except (ValueError, TypeError):
            rr = None
        if rr is None or rr != rr:
            components['respiratory_rate'] = 0  # Missing = skip
        elif rr <= 8:
            components['respiratory_rate'] = 3
        elif 9 <= rr <= 11:
            components['respiratory_rate'] = 1
        elif 12 <= rr <= 20:
            components['respiratory_rate'] = 0
        elif 21 <= rr <= 24:
            components['respiratory_rate'] = 2
        else:
            components['respiratory_rate'] = 3
        
        # 2. Oxygen Saturation (0-3)
        spo2 = vitals.get('spo2', 98)
        supplemental_o2 = vitals.get('supplemental_oxygen', False)
        
        # Select threshold based on scale
        if scale == NEWS2Scale.HOCNL:
            thresholds = cls.SLO2_THRESHOLDS_SCALE2
        else:
            thresholds = cls.SLO2_THRESHOLDS_SCALE1
        
        if spo2 <= int(thresholds['critical'][1]):
            components['spo2'] = 3
        elif spo2 <= int(thresholds['abnormal'][1]):
            components['spo2'] = 2
        elif spo2 <= int(thresholds['marginal'][1]):
            components['spo2'] = 1
        else:
            components['spo2'] = 0
        
        # Add oxygen therapy score (0-2)
        # Score 2 if on supplemental oxygen (applies to both scales)
        components['supplemental_oxygen'] = 2 if supplemental_o2 else 0
        
        # 3. Temperature (0-3)
        temp = vitals.get('temperature')
        try:
            temp = float(temp) if temp is not None else None
        except (ValueError, TypeError):
            temp = None
        if temp is None or temp != temp:
            components['temperature'] = 0  # Missing = skip
        elif temp <= 35.0:
            components['temperature'] = 3
        elif 35.1 <= temp <= 36.0:
            components['temperature'] = 1
        elif 36.1 <= temp <= 38.0:
            components['temperature'] = 0
        elif 38.1 <= temp <= 39.0:
            components['temperature'] = 1
        else:
            components['temperature'] = 2
        
        # 4. Systolic Blood Pressure (0-3)
        sbp = vitals.get('systolic_bp')
        try:
            sbp = float(sbp) if sbp is not None else None
        except (ValueError, TypeError):
            sbp = None
        if sbp is None or sbp != sbp:
            components['systolic_bp'] = 0  # Missing = skip
        elif sbp <= 90:
            components['systolic_bp'] = 3
        elif 91 <= sbp <= 100:
            components['systolic_bp'] = 2
        elif 101 <= sbp <= 110:
            components['systolic_bp'] = 1
        elif 111 <= sbp <= 219:
            components['systolic_bp'] = 0
        else:
            components['systolic_bp'] = 3
        
        # 5. Heart Rate (0-3)
        hr = vitals.get('heart_rate')
        if hr is None or hr != hr:
            components['heart_rate'] = 0  # Missing = skip
        elif hr <= 40:
            components['heart_rate'] = 3
        elif 41 <= hr <= 50:
            components['heart_rate'] = 1
        elif 51 <= hr <= 90:
            components['heart_rate'] = 0
        elif 91 <= hr <= 110:
            components['heart_rate'] = 1
        elif 111 <= hr <= 130:
            components['heart_rate'] = 1
        else:
            components['heart_rate'] = 3
        
        # 6. Alert Status (0-3)
        alert_status = vitals.get('alert_status', 'Alert')
        components['alert_status'] = 0 if str(alert_status).lower() == 'alert' else 3
        
        # Calculate total
        total_score = sum(components.values())
        
        # Determine risk level
        if total_score <= 4:
            risk_level = "Low"
            recommendation = "Routine monitoring"
        elif total_score <= 6:
            risk_level = "Medium"
            recommendation = "Increased monitoring frequency"
        else:
            risk_level = "High"
            recommendation = "Urgent assessment required"
        
        return {
            'total': total_score,
            'components': components,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'max_possible': 20,
            'scale_used': scale.value
        }


class qSOFACalculatorV2:
    """quick Sequential Organ Failure Assessment - Enhanced Version"""
    
    @classmethod
    def calculate(cls, vitals: Dict) -> Dict:
        """
        Calculate qSOFA score (Sepsis-3)
        
        Returns score >= 2 indicates high risk of poor outcomes outside ICU
        """
        score = 0
        components = {}
        
        # 1. Altered mental status (1 point)
        alert_status = str(vitals.get('alert_status', 'Alert')).lower()
        components['altered_mental_status'] = 0 if alert_status == 'alert' else 1
        score += components['altered_mental_status']
        
        # 2. Systolic BP <= 100 mmHg (1 point)
        sbp = vitals.get('systolic_bp')
        components['low_bp'] = 1 if (sbp is not None and sbp == sbp and sbp <= 100) else 0
        score += components['low_bp']
        
        # 3. Respiratory rate >= 22 breaths/min (1 point)
        rr = vitals.get('respiratory_rate', 14)
        components['high_rr'] = 1 if (rr is not None and rr == rr and rr >= 22) else 0
        score += components['high_rr']
        
        needs_investigation = score >= 2
        risk_level = "High (Sepsis likely)" if needs_investigation else "Low"
        
        return {
            'total': score,
            'components': components,
            'risk_level': risk_level,
            'needs_investigation': needs_investigation,
            'max_possible': 3,
            'recommendation': 'Blood cultures, lactate, antibiotics if sepsis suspected' if needs_investigation else 'Standard monitoring'
        }


class CARTCalculatorV2:
    """Cardiac Arrest Risk Triage - Enhanced Version"""
    
    @classmethod
    def calculate(cls, vitals: Dict) -> Dict:
        """
        Calculate CART score for cardiac arrest risk
        """
        score = 0
        
        # Age scoring
        age = vitals.get('age', 50)
        if age < 43:
            age_score = 0
        elif age < 50:
            age_score = 1
        elif age < 60:
            age_score = 2
        elif age < 70:
            age_score = 3
        else:
            age_score = 4
        score += age_score
        
        # Systolic BP scoring
        sbp = vitals.get('systolic_bp')
        if sbp is None or sbp != sbp:
            sbp_score = 0
        elif sbp >= 111:
            sbp_score = 0
        elif sbp >= 100:
            sbp_score = 1
        elif sbp >= 90:
            sbp_score = 2
        elif sbp >= 80:
            sbp_score = 3
        else:
            sbp_score = 4
        score += sbp_score
        
        # Heart rate scoring
        hr = vitals.get('heart_rate')
        if hr is None or hr != hr:
            hr_score = 0
        elif hr < 60:
            hr_score = 0
        elif hr < 100:
            hr_score = 1
        elif hr < 110:
            hr_score = 2
        elif hr < 120:
            hr_score = 3
        else:
            hr_score = 4
        score += hr_score
        
        # Respiratory rate scoring
        rr = vitals.get('respiratory_rate')
        if rr is None or rr != rr:
            rr_score = 0
        elif rr < 14:
            rr_score = 0
        elif rr < 20:
            rr_score = 1
        elif rr < 25:
            rr_score = 2
        elif rr < 30:
            rr_score = 3
        else:
            rr_score = 4
        score += rr_score
        
        # Consciousness scoring
        alert_status = str(vitals.get('alert_status', 'Alert')).lower()
        consciousness_score = 0 if alert_status == 'alert' else 4
        score += consciousness_score
        
        # Categorize risk
        if score <= 16:
            risk_category = "Low"
            recommendation = "Standard monitoring"
        elif score <= 20:
            risk_category = "Medium"
            recommendation = "Increased monitoring, consider ICU"
        elif score <= 24:
            risk_category = "High"
            recommendation = "ICU monitoring recommended"
        else:
            risk_category = "Highest"
            recommendation = "Urgent ICU admission and continuous monitoring"
        
        return {
            'total': score,
            'risk_category': risk_category,
            'max_possible': 20,
            'percentile': min(100, (score / 20) * 100),
            'recommendation': recommendation
        }


class ClinicalRecommendationsEngineV2:
    """Rule-based clinical recommendations engine"""
    
    @classmethod
    def generate(cls, news2: Dict, qsofa: Dict, cart: Dict) -> Dict:
        """
        Generate prioritized clinical recommendations
        """
        recommendations = []
        urgency_level = 0  # 0=Low, 1=Medium, 2=High
        
        # NEWS2-based recommendations
        if news2['risk_level'] == 'High':
            recommendations.append({
                'priority': 1,
                'action': '🔴 URGENT: Physician evaluation required immediately',
                'rationale': f"NEWS2 score {news2['total']}/20 indicates critical risk",
                'protocol': 'NEWS2 High Risk Protocol'
            })
            urgency_level = max(urgency_level, 2)
        elif news2['risk_level'] == 'Medium':
            recommendations.append({
                'priority': 2,
                'action': '🟡 Physician reassessment within 1 hour',
                'rationale': f"NEWS2 score {news2['total']}/20 indicates medium risk",
                'protocol': 'NEWS2 Medium Risk Protocol'
            })
            urgency_level = max(urgency_level, 1)
        
        # qSOFA-based recommendations
        if qsofa['needs_investigation']:
            recommendations.append({
                'priority': 1,
                'action': '🔴 SEPSIS ALERT: Initiate sepsis protocol',
                'rationale': f"qSOFA score {qsofa['total']}/3 - high risk of sepsis-related mortality",
                'protocol': 'Sepsis-3 Bundle'
            })
            recommendations.append({
                'priority': 2,
                'action': 'Draw blood cultures and lactate',
                'rationale': 'Essential for sepsis diagnosis and prognostication',
                'protocol': 'Sepsis Workup'
            })
            recommendations.append({
                'priority': 2,
                'action': 'Consider ICU consultation',
                'rationale': 'qSOFA >= 2 associated with ICU mortality 9.4-15.4%',
                'protocol': 'ICU Escalation'
            })
            urgency_level = max(urgency_level, 2)
        
        # CART-based recommendations
        if cart['risk_category'] in ['High', 'Highest']:
            recommendations.append({
                'priority': 1,
                'action': f"🔴 CARDIAC ARREST RISK: {cart['risk_category']} ({cart['percentile']:.0f}th percentile)",
                'rationale': f"CART score {cart['total']}/20 indicates high arrest risk",
                'protocol': 'Cardiac Monitoring Protocol'
            })
            recommendations.append({
                'priority': 2,
                'action': 'Continuous cardiac monitoring required',
                'rationale': 'ECG monitoring essential for early detection',
                'protocol': 'Monitoring'
            })
            if cart['risk_category'] == 'Highest':
                recommendations.append({
                    'priority': 1,
                    'action': 'Consider ICU/HDU admission',
                    'rationale': f"Highest risk category requires intensive monitoring",
                    'protocol': 'ICU Escalation'
                })
            urgency_level = max(urgency_level, 2)
        
        # General recommendations
        if urgency_level == 2:
            recommendations.append({
                'priority': 1,
                'action': '🚨 ACTIVATE RAPID RESPONSE TEAM',
                'rationale': 'Multiple high-risk indicators present',
                'protocol': 'RRT Activation'
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        
        urgency_text = ['Low', 'Medium', 'High'][urgency_level]
        
        return {
            'urgency': urgency_text,
            'urgency_level': urgency_level,
            'recommendations': recommendations,
            'total_actions': len(recommendations),
            'summary': f"{len(recommendations)} actions recommended (Urgency: {urgency_text})"
        }
