"""
Virtual ICU - Clinical Scoring Systems

Implements three major clinical scoring systems:
1. NEWS2 (National Early Warning Score 2)
2. qSOFA (quick Sequential Organ Failure Assessment)
3. CART (Cardiac Arrest Risk Triage)
"""

import pandas as pd
import numpy as np


class NEWS2Calculator:
    """
    National Early Warning Score 2
    
    7 parameters scored 0-3 points each (except O2 which is 0-2)
    Total: 0-20 points
    
    Risk levels:
    - 0-4: Low risk
    - 5-6: Medium risk
    - 7+: High risk (may need escalation)
    """
    
    @staticmethod
    def calculate(vitals: dict) -> dict:
        """Calculate NEWS2 score from vital signs.
        
        Parameters:
            vitals: Dict with keys: respiratory_rate, spo2, temperature,
                   systolic_bp, heart_rate, alert_status, supplemental_oxygen
        
        Returns:
            Dict with score, component scores, risk level, recommendations
        """
        score = 0
        components = {}
        
        # 1. RESPIRATORY RATE (0-3)
        rr = vitals.get('respiratory_rate', 14)
        if rr <= 8:
            components['respiratory_rate'] = 3
        elif 9 <= rr <= 11:
            components['respiratory_rate'] = 1
        elif 12 <= rr <= 20:
            components['respiratory_rate'] = 0
        elif 21 <= rr <= 24:
            components['respiratory_rate'] = 2
        else:  # >= 25
            components['respiratory_rate'] = 3
        
        # 2. OXYGEN SATURATION (0-3)
        spo2 = vitals.get('spo2', 98)
        if spo2 <= 91:
            components['spo2'] = 3
        elif 92 <= spo2 <= 93:
            components['spo2'] = 2
        elif 94 <= spo2 <= 95:
            components['spo2'] = 1
        else:  # >= 96
            components['spo2'] = 0
        
        # 3. TEMPERATURE (0-3)
        temp = vitals.get('temperature', 36.8)
        if temp <= 35.0:
            components['temperature'] = 3
        elif 35.1 <= temp <= 36.0:
            components['temperature'] = 1
        elif 36.1 <= temp <= 38.0:
            components['temperature'] = 0
        elif 38.1 <= temp <= 39.0:
            components['temperature'] = 1
        else:  # > 39.0
            components['temperature'] = 2
        
        # 4. SYSTOLIC BLOOD PRESSURE (0-3)
        sbp = vitals.get('systolic_bp', 120)
        if sbp <= 90:
            components['systolic_bp'] = 3
        elif 91 <= sbp <= 100:
            components['systolic_bp'] = 2
        elif 101 <= sbp <= 110:
            components['systolic_bp'] = 1
        elif 111 <= sbp <= 219:
            components['systolic_bp'] = 0
        else:  # >= 220
            components['systolic_bp'] = 3
        
        # 5. HEART RATE (0-3)
        hr = vitals.get('heart_rate', 70)
        if hr <= 40:
            components['heart_rate'] = 3
        elif 41 <= hr <= 50:
            components['heart_rate'] = 1
        elif 51 <= hr <= 90:
            components['heart_rate'] = 0
        elif 91 <= hr <= 110:
            components['heart_rate'] = 1
        elif 111 <= hr <= 130:
            components['heart_rate'] = 2
        else:  # > 130
            components['heart_rate'] = 3
        
        # 6. ALERT/CONSCIOUSNESS (0-3)
        alert = vitals.get('alert_status', 'Alert')
        if alert.lower() == 'alert':
            components['alert_status'] = 0
        else:
            components['alert_status'] = 3
        
        # 7. SUPPLEMENTAL OXYGEN (0-2)
        oxygen = vitals.get('supplemental_oxygen', False)
        components['supplemental_oxygen'] = 2 if oxygen else 0
        
        # Calculate total
        total_score = sum(components.values())
        
        # Risk level
        if total_score <= 4:
            risk_level = "Low"
        elif total_score <= 6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'total': total_score,
            'components': components,
            'risk_level': risk_level,
            'max_possible': 20
        }


class qSOFACalculator:
    """
    quick Sequential Organ Failure Assessment
    
    3 parameters, 1 point each
    Total: 0-3 points
    
    Risk:
    - <2: Low risk
    - >=2: High risk (sepsis likely)
    """
    
    @staticmethod
    def calculate(vitals: dict) -> dict:
        """Calculate qSOFA score.
        
        Parameters:
            vitals: Dict with keys: alert_status, systolic_bp, respiratory_rate
        
        Returns:
            Dict with score, components, risk level
        """
        score = 0
        components = {}
        
        # 1. ALTERED MENTAL STATUS (0-1)
        alert = vitals.get('alert_status', 'Alert')
        if alert.lower() != 'alert':
            components['altered_mental_status'] = 1
            score += 1
        else:
            components['altered_mental_status'] = 0
        
        # 2. SYSTOLIC BP <= 100 mmHg (0-1)
        sbp = vitals.get('systolic_bp', 120)
        if sbp <= 100:
            components['low_bp'] = 1
            score += 1
        else:
            components['low_bp'] = 0
        
        # 3. RESPIRATORY RATE >= 22 breaths/min (0-1)
        rr = vitals.get('respiratory_rate', 14)
        if rr >= 22:
            components['high_rr'] = 1
            score += 1
        else:
            components['high_rr'] = 0
        
        # Risk level
        if score >= 2:
            risk_level = "High (Sepsis likely)"
            needs_investigation = True
        else:
            risk_level = "Low"
            needs_investigation = False
        
        return {
            'total': score,
            'components': components,
            'risk_level': risk_level,
            'needs_investigation': needs_investigation,
            'max_possible': 3
        }


class CARTCalculator:
    """
    Cardiac Arrest Risk Triage
    
    Predicts risk of in-hospital cardiac arrest
    
    Risk Categories:
    - Low
    - Medium
    - High
    - Highest
    """
    
    @staticmethod
    def calculate(vitals: dict) -> dict:
        """Calculate CART score.
        
        Parameters:
            vitals: Dict with keys: age, systolic_bp, heart_rate, 
                   respiratory_rate, alert_status
        
        Returns:
            Dict with risk category, score, recommendations
        """
        score = 0
        
        # Age (0-4 points)
        age = vitals.get('age', 50)
        if age < 43:
            score += 0
        elif age < 50:
            score += 1
        elif age < 60:
            score += 2
        elif age < 70:
            score += 3
        else:
            score += 4
        
        # Systolic BP (0-4 points)
        sbp = vitals.get('systolic_bp', 120)
        if sbp >= 111:
            score += 0
        elif sbp >= 100:
            score += 1
        elif sbp >= 90:
            score += 2
        elif sbp >= 80:
            score += 3
        else:
            score += 4
        
        # Heart Rate (0-4 points)
        hr = vitals.get('heart_rate', 70)
        if hr < 60:
            score += 0
        elif hr < 100:
            score += 1
        elif hr < 110:
            score += 2
        elif hr < 120:
            score += 3
        else:
            score += 4
        
        # Respiratory Rate (0-4 points)
        rr = vitals.get('respiratory_rate', 14)
        if rr < 14:
            score += 0
        elif rr < 20:
            score += 1
        elif rr < 25:
            score += 2
        elif rr < 30:
            score += 3
        else:
            score += 4
        
        # Consciousness (0-4 points)
        alert = vitals.get('alert_status', 'Alert')
        if alert.lower() == 'alert':
            score += 0
        else:
            score += 4
        
        # Determine risk category
        if score <= 16:
            risk_category = "Low"
        elif score <= 20:
            risk_category = "Medium"
        elif score <= 24:
            risk_category = "High"
        else:
            risk_category = "Highest"
        
        return {
            'total': score,
            'risk_category': risk_category,
            'max_possible': 20,
            'percentile': min(100, (score / 20) * 100)
        }


class ClinicalRecommendations:
    """Generate clinical recommendations based on scores."""
    
    @staticmethod
    def get_recommendations(news2: dict, qsofa: dict, cart: dict) -> dict:
        """Generate integrated clinical recommendations.
        
        Parameters:
            news2: NEWS2 score dict
            qsofa: qSOFA score dict
            cart: CART score dict
        
        Returns:
            Dict with recommendations and urgency level
        """
        recommendations = []
        urgency = "Low"
        
        # NEWS2-based recommendations
        if news2['risk_level'] == 'High':
            recommendations.append("Urgent physician evaluation required")
            urgency = "High"
        elif news2['risk_level'] == 'Medium':
            recommendations.append("Physician reassessment within 1 hour")
            urgency = "Medium"
        
        # qSOFA-based recommendations
        if qsofa['needs_investigation']:
            recommendations.append("Investigate for sepsis (blood cultures, lactate)")
            recommendations.append("Consider ICU admission")
            if urgency != "High":
                urgency = "High"
        
        # CART-based recommendations
        if cart['risk_category'] in ['High', 'Highest']:
            recommendations.append("Continuous cardiac monitoring")
            recommendations.append("Consider ICU/HDU admission")
            if urgency != "High":
                urgency = "High"
        
        # Combined assessment
        if urgency == "High":
            recommendations.append("Activate rapid response team if not already done")
        
        return {
            'urgency': urgency,
            'recommendations': recommendations,
            'summary': f"{len(recommendations)} action(s) recommended"
        }


if __name__ == "__main__":
    print("Virtual ICU - Clinical Scoring Systems\n" + "=" * 70)
    
    # Example patient data
    vitals = {
        'respiratory_rate': 28,
        'spo2': 92,
        'temperature': 38.5,
        'systolic_bp': 95,
        'heart_rate': 120,
        'alert_status': 'Alert',
        'supplemental_oxygen': True,
        'age': 65
    }
    
    print("\nExample Patient Vitals:")
    print(f"  RR: {vitals['respiratory_rate']}, SpO2: {vitals['spo2']}%")
    print(f"  Temp: {vitals['temperature']}°C, SBP: {vitals['systolic_bp']} mmHg")
    print(f"  HR: {vitals['heart_rate']} bpm, Alert: {vitals['alert_status']}")
    
    # Calculate scores
    news2 = NEWS2Calculator.calculate(vitals)
    qsofa = qSOFACalculator.calculate(vitals)
    cart = CARTCalculator.calculate(vitals)
    
    print(f"\nNEWS2 Score: {news2['total']}/{news2['max_possible']} - {news2['risk_level']} risk")
    print(f"qSOFA Score: {qsofa['total']}/{qsofa['max_possible']} - {qsofa['risk_level']}")
    print(f"CART Score: {cart['total']}/{cart['max_possible']} - {cart['risk_category']} risk")
    
    # Get recommendations
    recommendations = ClinicalRecommendations.get_recommendations(news2, qsofa, cart)
    print(f"\nUrgency: {recommendations['urgency']}")
    print("Recommendations:")
    for rec in recommendations['recommendations']:
        print(f"  • {rec}")
    
    print("\n✅ Clinical scoring systems operational!")
