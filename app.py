from flask import Flask, request, jsonify, render_template_string
import json
import os
import PyPDF2
import io
import re
import random

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Simple mock model for deployment (lightweight version)
class SimpleMedicalModel:
    def __init__(self):
        # Predefined diagnoses for demo
        self.diagnoses = ['influenza', 'common cold', 'migraine', 'gastritis', 'anxiety', 'fatigue syndrome']
        self.risk_levels = ['low', 'moderate', 'high']
    
    def predict(self, symptoms, medical_history, age, gender, vital_signs):
        # Simple rule-based prediction for demo
        risk_score = min(10, len(symptoms) + len(medical_history) * 0.5 + (age - 30) * 0.1)
        
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

def extract_medical_info_from_pdf(pdf_file):
    """Extract medical information from PDF file"""
    try:
        # Use PyPDF2 only (lightweight)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        
        return extract_symptoms_from_text(text)
    
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return {"symptoms": [], "medical_history": [], "vital_signs": {}}

def extract_symptoms_from_text(text):
    """Extract symptoms and medical information from text"""
    text_lower = text.lower()
    
    # Common symptoms
    symptoms = []
    symptom_patterns = [
        r'fever|temperature.*?(\d+\.?\d*)', r'cough', r'headache', r'nausea', 
        r'vomiting', r'fatigue', r'weakness', r'chest pain', r'shortness of breath',
        r'dizziness', r'abdominal pain', r'joint pain', r'muscle pain', 
        r'back pain', r'sore throat', r'runny nose', r'rash', r'swelling'
    ]
    
    for pattern in symptom_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            symptom = pattern.replace(r'.*?(\d+\.?\d*)', '').replace('r\'', '').replace('|', ' or ')
            symptoms.append(symptom.split('|')[0])
    
    # Medical history
    medical_history = []
    history_patterns = [
        r'diabetes', r'hypertension', r'heart disease', r'asthma', r'allergies',
        r'cancer', r'stroke', r'arthritis', r'depression', r'anxiety'
    ]
    
    for pattern in history_patterns:
        if re.search(pattern, text_lower):
            medical_history.append(pattern.replace('r\'', ''))
    
    # Extract vital signs
    vital_signs = {}
    
    # Temperature
    temp_match = re.search(r'temperature.*?(\d+\.?\d*)', text_lower)
    if temp_match:
        vital_signs['temperature'] = float(temp_match.group(1))
    
    # Blood pressure
    bp_match = re.search(r'blood pressure.*?(\d+/\d+)', text_lower)
    if bp_match:
        vital_signs['blood_pressure'] = bp_match.group(1)
    
    # Heart rate
    hr_match = re.search(r'heart rate.*?(\d+)', text_lower)
    if hr_match:
        vital_signs['heart_rate'] = int(hr_match.group(1))
    
    return {
        "symptoms": list(set(symptoms)),
        "medical_history": list(set(medical_history)),
        "vital_signs": vital_signs
    }

@app.route('/')
def home():
    """Simple home page with API documentation"""
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical Diagnosis API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 5px; }
            .method { background: #007bff; color: white; padding: 5px 10px; border-radius: 3px; }
            code { background: #e9ecef; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>Medical Diagnosis API</h1>
        <p>This API provides medical diagnosis and risk assessment based on symptoms and medical reports.</p>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /upload_medical_pdf</h3>
            <p>Upload a medical PDF report for analysis</p>
            <p><strong>Parameters:</strong></p>
            <ul>
                <li><code>pdf_file</code> - PDF file containing medical report</li>
                <li><code>age</code> - Patient age (optional)</li>
                <li><code>gender</code> - Patient gender (optional)</li>
            </ul>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /diagnose</h3>
            <p>Get diagnosis based on symptoms and patient information</p>
            <p><strong>JSON Body Example:</strong></p>
            <pre>{
    "symptoms": ["fever", "cough", "fatigue"],
    "medical_history": ["diabetes"],
    "age": 45,
    "gender": "male",
    "vital_signs": {
        "temperature": 102.5,
        "blood_pressure": "140/90",
        "heart_rate": 95
    }
}</pre>
        </div>
        
        <p><strong>Note:</strong> For a better user interface, use the Streamlit frontend.</p>
    </body>
    </html>
    '''
    return html

@app.route('/upload_medical_pdf', methods=['POST'])
def upload_medical_pdf():
    """Endpoint to upload medical PDF reports and get diagnosis"""
    try:
        # Check if file is present
        if 'pdf_file' not in request.files:
            return jsonify({'error': 'No PDF file provided'}), 400
        
        file = request.files['pdf_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'File must be a PDF'}), 400
        
        # Get additional parameters
        age = request.form.get('age', 40)
        gender = request.form.get('gender', 'unknown')
        
        try:
            age = int(age)
        except:
            age = 40
        
        # Extract medical information from PDF
        medical_info = extract_medical_info_from_pdf(file)
        
        # If no symptoms found, return error
        if not medical_info['symptoms']:
            return jsonify({
                'error': 'Could not extract medical information from PDF',
                'suggestion': 'Please ensure the PDF contains clear medical text'
            }), 400
        
        # Set default vital signs if not extracted
        if not medical_info['vital_signs']:
            medical_info['vital_signs'] = {
                'temperature': 98.6,
                'blood_pressure': '120/80',
                'heart_rate': 70
            }
        
        # Get diagnosis using the model
        prediction = model.predict(
            symptoms=medical_info['symptoms'],
            medical_history=medical_info['medical_history'],
            age=age,
            gender=gender,
            vital_signs=medical_info['vital_signs']
        )
        
        # Prepare response
        response = {
            'extracted_info': medical_info,
            'patient_info': {
                'age': age,
                'gender': gender
            },
            'diagnosis': prediction['diagnosis'],
            'risk_level': prediction['risk_level'],
            'risk_score': prediction['risk_score'],
            'confidence': prediction['confidence'],
            'recommendations': get_recommendations(prediction['diagnosis'], prediction['risk_level'])
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/diagnose', methods=['POST'])
def diagnose():
    """Endpoint to get diagnosis based on provided symptoms and patient info"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
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
            return jsonify({'error': 'Symptoms are required'}), 400
        
        # Get diagnosis
        prediction = model.predict(
            symptoms=symptoms,
            medical_history=medical_history,
            age=age,
            gender=gender,
            vital_signs=vital_signs
        )
        
        response = {
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
            'recommendations': get_recommendations(prediction['diagnosis'], prediction['risk_level'])
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def get_recommendations(diagnosis, risk_level):
    """Get recommendations based on diagnosis and risk level"""
    recommendations = {
        'influenza': [
            'Get plenty of rest',
            'Stay hydrated',
            'Consider antiviral medication',
            'Monitor temperature'
        ],
        'heart attack': [
            'Seek immediate emergency medical attention',
            'Call 911 immediately',
            'Take aspirin if not allergic',
            'Stay calm and rest'
        ],
        'migraine': [
            'Rest in a dark, quiet room',
            'Apply cold compress to head',
            'Stay hydrated',
            'Consider over-the-counter pain medication'
        ],
        'appendicitis': [
            'Seek immediate medical attention',
            'Do not eat or drink',
            'Do not take laxatives',
            'Go to emergency room'
        ],
        'diabetes': [
            'Monitor blood sugar levels',
            'Follow prescribed medication regimen',
            'Maintain healthy diet',
            'Regular exercise as recommended'
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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True}), 200

@app.route('/metrics', methods=['GET'])
def get_model_metrics():
    """Get model performance metrics"""
    try:
        if hasattr(model, 'metrics') and model.metrics:
            return jsonify(model.metrics), 200
        else:
            return jsonify({'error': 'No metrics available. Model needs to be retrained.'}), 404
    except Exception as e:
        return jsonify({'error': f'Error retrieving metrics: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6001)
