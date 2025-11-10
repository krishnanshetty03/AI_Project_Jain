import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.multiclass import OneVsRestClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from textblob import TextBlob
import re

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class MedicalDiagnosisModel:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.diagnosis_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.diagnosis_encoder = LabelEncoder()
        self.risk_encoder = LabelEncoder()
        self.metrics = {}  # Store model performance metrics
        
    def preprocess_text(self, text_list):
        """Convert list of symptoms/conditions to a single text string"""
        if isinstance(text_list, list):
            return ' '.join(text_list).lower()
        return str(text_list).lower()
    
    def extract_features(self, data):
        """Extract features from medical data"""
        features = []
        
        for record in data:
            # Combine symptoms and medical history
            symptoms_text = self.preprocess_text(record.get('symptoms', []))
            history_text = self.preprocess_text(record.get('medical_history', []))
            combined_text = f"{symptoms_text} {history_text}"
            
            # Add numerical features
            age = record.get('age', 0)
            gender = 1 if record.get('gender', '').lower() == 'male' else 0
            
            # Extract vital signs
            vital_signs = record.get('vital_signs', {})
            temperature = vital_signs.get('temperature', 98.6)
            heart_rate = vital_signs.get('heart_rate', 70)
            
            # Parse blood pressure
            bp = vital_signs.get('blood_pressure', '120/80')
            try:
                systolic, diastolic = map(int, bp.split('/'))
            except:
                systolic, diastolic = 120, 80
            
            features.append({
                'text': combined_text,
                'age': age,
                'gender': gender,
                'temperature': temperature,
                'heart_rate': heart_rate,
                'systolic_bp': systolic,
                'diastolic_bp': diastolic,
                'diagnosis': record.get('diagnosis', ''),
                'risk_level': record.get('risk_level', ''),
                'risk_score': record.get('risk_score', 0)
            })
        
        return features
    
    def calculate_metrics(self, y_true, y_pred, y_proba, model_name, classes):
        """Calculate comprehensive metrics for model evaluation"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_score_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_score_weighted'] = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Calculate sensitivity (recall) and specificity for each class
        n_classes = len(classes)
        sensitivity = []
        specificity = []
        
        # Ensure we don't exceed the confusion matrix dimensions
        actual_classes = min(n_classes, cm.shape[0])
        
        for i in range(actual_classes):
            tp = cm[i, i]
            fn = sum(cm[i, :]) - tp
            fp = sum(cm[:, i]) - tp
            tn = sum(sum(cm)) - tp - fn - fp
            
            # Sensitivity (True Positive Rate / Recall)
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            sensitivity.append(sens)
            
            # Specificity (True Negative Rate)
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity.append(spec)
        
        metrics['sensitivity_per_class'] = sensitivity
        metrics['specificity_per_class'] = specificity
        metrics['average_sensitivity'] = np.mean(sensitivity)
        metrics['average_specificity'] = np.mean(specificity)
        
        # ROC curve data (for multiclass, we'll use one-vs-rest approach)
        try:
            actual_classes = min(n_classes, y_proba.shape[1])
            if actual_classes > 2:
                # Binarize the output for multiclass ROC
                y_true_bin = label_binarize(y_true, classes=range(actual_classes))
                
                # Calculate ROC for each class
                fpr = {}
                tpr = {}
                roc_auc = {}
                
                for i in range(actual_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Convert to lists for JSON serialization
                metrics['roc_data'] = {
                    'fpr': {str(k): v.tolist() for k, v in fpr.items()},
                    'tpr': {str(k): v.tolist() for k, v in tpr.items()},
                    'auc': roc_auc,
                    'classes': classes
                }
            else:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                metrics['roc_data'] = {
                    'fpr': {'0': fpr.tolist()},
                    'tpr': {'0': tpr.tolist()},
                    'auc': {'0': auc(fpr, tpr)},
                    'classes': classes
                }
        except Exception as e:
            print(f"Warning: Could not calculate ROC curve: {e}")
            metrics['roc_data'] = None
        
        return metrics
    
    def train(self, json_file_path):
        """Train the model on medical data"""
        print("Loading medical data...")
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        print("Extracting features...")
        features = self.extract_features(data)
        df = pd.DataFrame(features)
        
        # Prepare text features
        print("Vectorizing text features...")
        text_features = self.tfidf_vectorizer.fit_transform(df['text'])
        
        # Prepare numerical features
        numerical_features = df[['age', 'gender', 'temperature', 'heart_rate', 'systolic_bp', 'diastolic_bp']].values
        
        # Combine features
        X = np.hstack([text_features.toarray(), numerical_features])
        
        # Prepare labels
        y_diagnosis = self.diagnosis_encoder.fit_transform(df['diagnosis'])
        y_risk = self.risk_encoder.fit_transform(df['risk_level'])
        
        # Split data
        X_train, X_test, y_diag_train, y_diag_test, y_risk_train, y_risk_test = train_test_split(
            X, y_diagnosis, y_risk, test_size=0.2, random_state=42
        )
        
        # Train models
        print("Training diagnosis model...")
        self.diagnosis_model.fit(X_train, y_diag_train)
        
        print("Training risk assessment model...")
        self.risk_model.fit(X_train, y_risk_train)
        
        # Evaluate models
        diag_pred = self.diagnosis_model.predict(X_test)
        risk_pred = self.risk_model.predict(X_test)
        
        # Get prediction probabilities
        diag_proba = self.diagnosis_model.predict_proba(X_test)
        risk_proba = self.risk_model.predict_proba(X_test)
        
        # Calculate comprehensive metrics
        diag_classes = self.diagnosis_encoder.classes_.tolist()
        risk_classes = self.risk_encoder.classes_.tolist()
        
        print("\nCalculating comprehensive metrics...")
        diag_metrics = self.calculate_metrics(y_diag_test, diag_pred, diag_proba, 'diagnosis', diag_classes)
        risk_metrics = self.calculate_metrics(y_risk_test, risk_pred, risk_proba, 'risk', risk_classes)
        
        # Store metrics
        self.metrics = {
            'diagnosis_metrics': diag_metrics,
            'risk_metrics': risk_metrics
        }
        
        print("\nDiagnosis Model Performance:")
        print(f"Accuracy: {diag_metrics['accuracy']:.3f}")
        print(f"F1 Score (Macro): {diag_metrics['f1_score_macro']:.3f}")
        print(f"F1 Score (Weighted): {diag_metrics['f1_score_weighted']:.3f}")
        print(f"Average Sensitivity: {diag_metrics['average_sensitivity']:.3f}")
        print(f"Average Specificity: {diag_metrics['average_specificity']:.3f}")
        
        print("\nRisk Assessment Model Performance:")
        print(f"Accuracy: {risk_metrics['accuracy']:.3f}")
        print(f"F1 Score (Macro): {risk_metrics['f1_score_macro']:.3f}")
        print(f"F1 Score (Weighted): {risk_metrics['f1_score_weighted']:.3f}")
        print(f"Average Sensitivity: {risk_metrics['average_sensitivity']:.3f}")
        print(f"Average Specificity: {risk_metrics['average_specificity']:.3f}")
        
        # Save models
        self.save_model()
        print("Models saved successfully!")
    
    def predict(self, symptoms, medical_history, age, gender, vital_signs):
        """Make prediction for new patient data"""
        # Prepare input
        symptoms_text = self.preprocess_text(symptoms)
        history_text = self.preprocess_text(medical_history)
        combined_text = f"{symptoms_text} {history_text}"
        
        # Text features
        text_features = self.tfidf_vectorizer.transform([combined_text])
        
        # Numerical features
        gender_num = 1 if gender.lower() == 'male' else 0
        temperature = vital_signs.get('temperature', 98.6)
        heart_rate = vital_signs.get('heart_rate', 70)
        
        bp = vital_signs.get('blood_pressure', '120/80')
        try:
            systolic, diastolic = map(int, bp.split('/'))
        except:
            systolic, diastolic = 120, 80
        
        numerical_features = np.array([[age, gender_num, temperature, heart_rate, systolic, diastolic]])
        
        # Combine features
        X = np.hstack([text_features.toarray(), numerical_features])
        
        # Make predictions
        diagnosis_pred = self.diagnosis_model.predict(X)[0]
        risk_pred = self.risk_model.predict(X)[0]
        
        # Get probabilities for confidence scores
        diagnosis_proba = self.diagnosis_model.predict_proba(X)[0]
        risk_proba = self.risk_model.predict_proba(X)[0]
        
        # Decode predictions
        diagnosis = self.diagnosis_encoder.inverse_transform([diagnosis_pred])[0]
        risk_level = self.risk_encoder.inverse_transform([risk_pred])[0]
        
        # Calculate risk score (1-10 scale)
        risk_score = int(max(diagnosis_proba) * 10)
        
        return {
            'diagnosis': diagnosis,
            'risk_level': risk_level,
            'risk_score': risk_score,
            'confidence': float(max(diagnosis_proba))
        }
    
    def save_model(self):
        """Save trained models and metrics"""
        joblib.dump(self.tfidf_vectorizer, 'tfidf_vectorizer.pkl')
        joblib.dump(self.diagnosis_model, 'diagnosis_model.pkl')
        joblib.dump(self.risk_model, 'risk_model.pkl')
        joblib.dump(self.diagnosis_encoder, 'diagnosis_encoder.pkl')
        joblib.dump(self.risk_encoder, 'risk_encoder.pkl')
        joblib.dump(self.metrics, 'model_metrics.pkl')
    
    def load_model(self):
        """Load trained models and metrics"""
        try:
            self.tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
            self.diagnosis_model = joblib.load('diagnosis_model.pkl')
            self.risk_model = joblib.load('risk_model.pkl')
            self.diagnosis_encoder = joblib.load('diagnosis_encoder.pkl')
            self.risk_encoder = joblib.load('risk_encoder.pkl')
            
            # Try to load metrics, but don't fail if they don't exist
            try:
                self.metrics = joblib.load('model_metrics.pkl')
            except FileNotFoundError:
                self.metrics = {}
                print("No metrics file found, metrics will be empty")
            
            return True
        except FileNotFoundError:
            return False

def extract_text_from_pdf_content(pdf_content):
    """Extract text from PDF content using simple pattern matching"""
    # This is a simplified text extraction - in production, you'd use proper PDF parsing
    text = pdf_content.decode('utf-8', errors='ignore')
    
    # Common medical terms and symptoms to look for
    medical_terms = [
        'fever', 'cough', 'headache', 'nausea', 'vomiting', 'fatigue', 'weakness',
        'chest pain', 'shortness of breath', 'dizziness', 'abdominal pain',
        'joint pain', 'muscle pain', 'back pain', 'sore throat', 'runny nose',
        'diabetes', 'hypertension', 'heart disease', 'asthma', 'allergies',
        'blood pressure', 'temperature', 'heart rate', 'weight', 'height'
    ]
    
    found_symptoms = []
    text_lower = text.lower()
    
    for term in medical_terms:
        if term in text_lower:
            found_symptoms.append(term)
    
    return found_symptoms

if __name__ == "__main__":
    # Train the model
    model = MedicalDiagnosisModel()
    model.train('medical_data.json')
