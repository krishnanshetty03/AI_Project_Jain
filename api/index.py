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
import cgi
import io
import re

def extract_pdf_text(pdf_content):
    """Simple text extraction for PDF (mock implementation)"""
    # For deployment simplicity, we'll simulate PDF text extraction
    # In a real implementation, you'd use PyPDF2 or similar library
    
    mock_pdf_text = """
    Patient Name: John Doe
    Age: 45
    Gender: Male
    
    Chief Complaint: Fever and cough for 3 days
    
    History of Present Illness:
    Patient reports onset of fever (101.2°F) three days ago, accompanied by persistent dry cough.
    Also experiencing fatigue, headache, and mild body aches.
    
    Vital Signs:
    Temperature: 101.2°F
    Blood Pressure: 135/85
    Heart Rate: 88 bpm
    Respiratory Rate: 18
    
    Past Medical History:
    - Hypertension
    - Diabetes Type 2
    
    Assessment and Plan:
    Likely viral upper respiratory infection
    Recommend rest, fluids, and symptom management
    """
    
    return mock_pdf_text

def parse_medical_text(text):
    """Extract medical information from text"""
    # Simple regex patterns to extract medical info
    symptoms = []
    medical_history = []
    vital_signs = {}
    
    # Extract symptoms (simple keyword matching)
    symptom_keywords = ['fever', 'cough', 'headache', 'fatigue', 'nausea', 'vomiting', 
                       'chest pain', 'shortness of breath', 'dizziness', 'body aches']
    
    text_lower = text.lower()
    for symptom in symptom_keywords:
        if symptom in text_lower:
            symptoms.append(symptom)
    
    # Extract medical history
    history_keywords = ['hypertension', 'diabetes', 'heart disease', 'asthma', 'cancer']
    for condition in history_keywords:
        if condition in text_lower:
            medical_history.append(condition)
    
    # Extract vital signs (simple regex)
    temp_match = re.search(r'temperature[:\s]*(\d+\.?\d*)', text_lower)
    if temp_match:
        vital_signs['temperature'] = float(temp_match.group(1))
    
    bp_match = re.search(r'blood pressure[:\s]*(\d+)/(\d+)', text_lower)
    if bp_match:
        vital_signs['blood_pressure'] = f"{bp_match.group(1)}/{bp_match.group(2)}"
    
    hr_match = re.search(r'heart rate[:\s]*(\d+)', text_lower)
    if hr_match:
        vital_signs['heart_rate'] = int(hr_match.group(1))
    
    return symptoms, medical_history, vital_signs

class handler(BaseHTTPRequestHandler):
    def _set_cors_headers(self):
        """Set CORS headers to allow cross-origin requests"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Access-Control-Max-Age', '86400')
    
    def do_OPTIONS(self):
        """Handle preflight CORS requests"""
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()
        return
    
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self._set_cors_headers()
        self.end_headers()
        response = json.dumps({'message': 'Medical Diagnosis API', 'status': 'healthy'})
        self.wfile.write(response.encode())
        return

    def do_POST(self):
        try:
            # Check if this is a PDF upload request
            content_type = self.headers.get('Content-Type', '')
            
            if 'multipart/form-data' in content_type:
                # Handle PDF upload
                self._handle_pdf_upload()
                return
            else:
                # Handle regular JSON diagnosis request
                self._handle_diagnosis_request()
                return
        
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self._set_cors_headers()
            self.end_headers()
            response = json.dumps({'error': f'An error occurred: {str(e)}'})
            self.wfile.write(response.encode())
            return
    
    def _handle_pdf_upload(self):
        """Handle PDF upload and analysis"""
        try:
            # For simplicity, we'll simulate PDF processing
            # In a real deployment, you'd need proper multipart parsing
            
            # Mock extracted data from PDF
            symptoms = ['fever', 'cough', 'headache', 'fatigue']
            medical_history = ['hypertension', 'diabetes']
            vital_signs = {
                'temperature': 101.2,
                'blood_pressure': '135/85',
                'heart_rate': 88
            }
            age = 45
            gender = 'male'
            
            # Get diagnosis using the mock extracted data
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
                'extracted_info': {
                    'symptoms': symptoms,
                    'medical_history': medical_history,
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
            self._set_cors_headers()
            self.end_headers()
            response = json.dumps(response_data)
            self.wfile.write(response.encode())
        
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self._set_cors_headers()
            self.end_headers()
            response = json.dumps({'error': f'PDF processing error: {str(e)}'})
            self.wfile.write(response.encode())
    
    def _handle_diagnosis_request(self):
        """Handle regular diagnosis request"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Parse JSON data
            try:
                data = json.loads(post_data.decode('utf-8'))
            except:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
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
                self._set_cors_headers()
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
            self._set_cors_headers()
            self.end_headers()
            response = json.dumps(response_data)
            self.wfile.write(response.encode())
        
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self._set_cors_headers()
            self.end_headers()
            response = json.dumps({'error': f'Diagnosis processing error: {str(e)}'})
            self.wfile.write(response.encode())
