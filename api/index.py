import json
import random

# Simple mock model for deployment (lightweight version)
class SimpleMedicalModel:
    def __init__(self):
        # Predefined diagnoses for demo
        self.diagnoses = ['influenza', 'common cold', 'migraine', 'gastritis', 'anxiety', 'fatigue syndrome']
        self.risk_levels = ['low', 'moderate', 'high']
    
    def predict(self, symptoms, medical_history, age, gender, vital_signs):
        # Simple rule-based prediction for demo
        risk_score = min(10, len(symptoms) + len(medical_history) * 0.5 + max(0, (age - 30)) * 0.1)
        
        if risk_score <= 3:
            risk_level = 'low'
        elif risk_score <= 7:
            risk_level = 'moderate'
        else:
            risk_level = 'high'
        
        # Simple diagnosis based on symptoms
        diagnosis = 'common cold'
        if 'fever' in symptoms and 'headache' in symptoms:
            diagnosis = 'influenza'
        elif 'chest pain' in symptoms:
            diagnosis = 'chest pain syndrome'
            risk_level = 'high'
        elif 'headache' in symptoms and 'dizziness' in symptoms:
            diagnosis = 'migraine'
        elif 'nausea' in symptoms or 'vomiting' in symptoms:
            diagnosis = 'gastritis'
        
        return {
            'diagnosis': diagnosis,
            'risk_level': risk_level,
            'risk_score': round(risk_score, 1),
            'confidence': round(0.7 + random.random() * 0.25, 2)
        }

# Initialize simple model
model = SimpleMedicalModel()

def get_recommendations(diagnosis, risk_level):
    """Get recommendations based on diagnosis and risk level"""
    recommendations = {
        'influenza': [
            'Get plenty of rest',
            'Stay hydrated',
            'Consider antiviral medication',
            'Monitor temperature'
        ],
        'migraine': [
            'Rest in a dark, quiet room',
            'Apply cold compress to head',
            'Stay hydrated',
            'Consider over-the-counter pain medication'
        ],
        'gastritis': [
            'Avoid spicy and acidic foods',
            'Eat smaller, frequent meals',
            'Stay hydrated',
            'Consider antacids'
        ]
    }
    
    # Default recommendations based on risk level
    if diagnosis.lower() not in recommendations:
        if risk_level == 'high':
            return [
                'Seek immediate medical attention',
                'Contact your healthcare provider',
                'Monitor symptoms closely',
                'Follow up as recommended'
            ]
        elif risk_level == 'moderate':
            return [
                'Schedule appointment with healthcare provider',
                'Monitor symptoms',
                'Follow prescribed treatments',
                'Maintain healthy lifestyle'
            ]
        else:
            return [
                'Monitor symptoms',
                'Rest and stay hydrated',
                'Contact doctor if symptoms worsen',
                'Maintain healthy habits'
            ]
    
    return recommendations.get(diagnosis.lower(), [])

from http.server import BaseHTTPRequestHandler
import urllib.parse as urlparse

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = json.dumps({'message': 'Medical Diagnosis API', 'status': 'healthy'})
        self.wfile.write(response.encode())
        return

    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Parse JSON data
            try:
                data = json.loads(post_data.decode('utf-8'))
            except:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = json.dumps({'error': 'Invalid JSON data'})
                self.wfile.write(response.encode())
                return
            
            # Extract required fields
            symptoms = data.get('symptoms', [])
            medical_history = data.get('medical_history', [])
            age = data.get('age', 40)
            gender = data.get('gender', 'unknown')
            vital_signs = data.get('vital_signs', {
                'temperature': 98.6,
                'blood_pressure': '120/80',
                'heart_rate': 70
            })
            
            if not symptoms:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = json.dumps({'error': 'Symptoms are required'})
                self.wfile.write(response.encode())
                return
            
            # Get diagnosis
            prediction = model.predict(
                symptoms=symptoms,
                medical_history=medical_history,
                age=age,
                gender=gender,
                vital_signs=vital_signs
            )
            
            # Get recommendations
            recommendations = get_recommendations(prediction['diagnosis'], prediction['risk_level'])
            
            response_data = {
                'input_data': {
                    'symptoms': symptoms,
                    'medical_history': medical_history,
                    'age': age,
                    'gender': gender,
                    'vital_signs': vital_signs
                },
                'diagnosis': prediction['diagnosis'],
                'risk_level': prediction['risk_level'],
                'risk_score': prediction['risk_score'],
                'confidence': prediction['confidence'],
                'recommendations': recommendations
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps(response_data)
            self.wfile.write(response.encode())
            return
        
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({'error': f'An error occurred: {str(e)}'})
            self.wfile.write(response.encode())
            return
