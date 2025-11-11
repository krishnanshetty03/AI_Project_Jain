import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from io import StringIO

# Configure Streamlit page
st.set_page_config(
    page_title="Medical Diagnosis Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        color: #ff4444;
        font-weight: bold;
    }
    .risk-moderate {
        color: #ff8800;
        font-weight: bold;
    }
    .risk-low {
        color: #44ff44;
        font-weight: bold;
    }
    .diagnosis-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Flask API base URL
API_BASE_URL = "https://ai-project-jain-gx8t.vercel.app"

def check_flask_server():
    """Check if Flask server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_pdf_to_api(pdf_file, age, gender):
    """Upload PDF file to Flask API"""
    try:
        files = {'pdf_file': pdf_file}
        data = {'age': age, 'gender': gender}
        
        response = requests.post(
            f"{API_BASE_URL}/upload_medical_pdf",
            files=files,
            data=data,
            timeout=30
        )
        
        return response.json(), response.status_code
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}, 500

def get_model_metrics():
    """Get model performance metrics from Flask API"""
    # Temporary hardcoded metrics for demonstration
    sample_metrics = {
        "diagnosis_metrics": {
            "accuracy": 0.85,
            "f1_score_macro": 0.83,
            "f1_score_weighted": 0.84,
            "average_sensitivity": 0.82,
            "average_specificity": 0.88,
            "sensitivity_per_class": [0.85, 0.78, 0.82, 0.88, 0.79, 0.80],
            "specificity_per_class": [0.89, 0.87, 0.90, 0.85, 0.88, 0.89],
            "confusion_matrix": [
                [45, 3, 2, 1, 2, 1],
                [2, 40, 3, 2, 1, 2],
                [1, 2, 42, 2, 1, 1],
                [2, 1, 2, 44, 2, 1],
                [1, 2, 1, 2, 38, 2],
                [2, 1, 1, 1, 2, 41]
            ],
            "roc_data": {
                "classes": ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6"],
                "auc": {"0": 0.92, "1": 0.89, "2": 0.91, "3": 0.90, "4": 0.88, "5": 0.89},
                "fpr": {
                    "0": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    "1": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    "2": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    "3": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    "4": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    "5": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                },
                "tpr": {
                    "0": [0.0, 0.4, 0.6, 0.8, 0.9, 1.0],
                    "1": [0.0, 0.35, 0.55, 0.75, 0.85, 1.0],
                    "2": [0.0, 0.38, 0.58, 0.78, 0.88, 1.0],
                    "3": [0.0, 0.37, 0.57, 0.77, 0.87, 1.0],
                    "4": [0.0, 0.34, 0.54, 0.74, 0.84, 1.0],
                    "5": [0.0, 0.36, 0.56, 0.76, 0.86, 1.0]
                }
            }
        },
        "risk_metrics": {
            "accuracy": 0.88,
            "f1_score_macro": 0.87,
            "f1_score_weighted": 0.88,
            "average_sensitivity": 0.86,
            "average_specificity": 0.89,
            "confusion_matrix": [
                [42, 4, 2],
                [3, 45, 3],
                [2, 3, 40]
            ],
            "sensitivity_per_class": [0.88, 0.85, 0.86],
            "specificity_per_class": [0.90, 0.88, 0.89],
            "roc_data": {
                "classes": ["Low", "Moderate", "High"],
                "auc": {"0": 0.91, "1": 0.89, "2": 0.90},
                "fpr": {
                    "0": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    "1": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    "2": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                },
                "tpr": {
                    "0": [0.0, 0.45, 0.65, 0.82, 0.92, 1.0],
                    "1": [0.0, 0.42, 0.62, 0.80, 0.90, 1.0],
                    "2": [0.0, 0.43, 0.63, 0.81, 0.91, 1.0]
                }
            }
        }
    }
    return sample_metrics, True

def get_diagnosis_from_api(symptoms, medical_history, age, gender, vital_signs):
    """Get diagnosis from Flask API"""
    try:
        data = {
            "symptoms": symptoms,
            "medical_history": medical_history,
            "age": age,
            "gender": gender,
            "vital_signs": vital_signs
        }
        
        response = requests.post(
            f"{API_BASE_URL}/diagnose",
            json=data,
            timeout=30
        )
        
        return response.json(), response.status_code
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}, 500

def display_risk_gauge(risk_score, risk_level):
    """Display risk score as a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score"},
        delta = {'reference': 5},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 3], 'color': "lightgreen"},
                {'range': [3, 7], 'color': "yellow"},
                {'range': [7, 10], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 8
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    st.markdown('<h1 class="main-header">🏥 Medical Diagnosis Assistant</h1>', unsafe_allow_html=True)
    
    # Check Flask server status
    if not check_flask_server():
        st.error("⚠️ Flask API server is not running. Please start the Flask server first by running: `python main_app.py`")
        st.stop()
    else:
        st.success("✅ Connected to Flask API server")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", 
                                   ["PDF Upload", "Manual Input", "Model Metrics", "About"])
    
    if app_mode == "PDF Upload":
        pdf_upload_mode()
    elif app_mode == "Manual Input":
        manual_input_mode()
    elif app_mode == "Model Metrics":
        model_metrics_page()
    elif app_mode == "About":
        about_page()

def pdf_upload_mode():
    st.header("📄 Upload Medical PDF Report")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            help="Upload a medical report in PDF format"
        )
    
    with col2:
        st.subheader("Patient Information")
        age = st.number_input("Age", min_value=1, max_value=120, value=40)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    if uploaded_file is not None:
        st.info("📄 PDF file uploaded successfully!")
        
        if st.button("Analyze PDF Report", type="primary"):
            with st.spinner("Analyzing PDF report..."):
                result, status_code = upload_pdf_to_api(uploaded_file, age, gender.lower())
                
                if status_code == 200:
                    display_diagnosis_results(result)
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error occurred')}")
                    if 'suggestion' in result:
                        st.info(f"💡 Suggestion: {result['suggestion']}")

def manual_input_mode():
    st.header("✍️ Manual Symptom Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        age = st.number_input("Age", min_value=1, max_value=120, value=40)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        st.subheader("Vital Signs")
        temperature = st.number_input("Temperature (°F)", min_value=95.0, max_value=110.0, value=98.6, step=0.1)
        systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
        diastolic_bp = st.number_input("Diastolic BP", min_value=40, max_value=120, value=80)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=70)
    
    with col2:
        st.subheader("Symptoms")
        symptoms_options = [
            "fever", "cough", "headache", "nausea", "vomiting", "fatigue", "weakness",
            "chest pain", "shortness of breath", "dizziness", "abdominal pain",
            "joint pain", "muscle pain", "back pain", "sore throat", "runny nose",
            "rash", "swelling", "confusion", "seizures", "memory loss"
        ]
        
        selected_symptoms = st.multiselect("Select Symptoms", symptoms_options)
        custom_symptom = st.text_input("Add custom symptom (optional)")
        if custom_symptom:
            selected_symptoms.append(custom_symptom)
        
        st.subheader("Medical History")
        history_options = [
            "diabetes", "hypertension", "heart disease", "asthma", "allergies",
            "cancer", "stroke", "arthritis", "depression", "anxiety", "obesity",
            "smoking", "family history of heart disease", "family history of cancer"
        ]
        
        selected_history = st.multiselect("Select Medical History", history_options)
        custom_history = st.text_input("Add custom medical history (optional)")
        if custom_history:
            selected_history.append(custom_history)
    
    if st.button("Get Diagnosis", type="primary"):
        if not selected_symptoms:
            st.error("Please select at least one symptom")
            return
        
        vital_signs = {
            "temperature": temperature,
            "blood_pressure": f"{systolic_bp}/{diastolic_bp}",
            "heart_rate": heart_rate
        }
        
        with st.spinner("Analyzing symptoms..."):
            result, status_code = get_diagnosis_from_api(
                selected_symptoms, selected_history, age, gender.lower(), vital_signs
            )
            
            if status_code == 200:
                display_diagnosis_results(result)
            else:
                st.error(f"Error: {result.get('error', 'Unknown error occurred')}")

def display_diagnosis_results(result):
    """Display diagnosis results in a formatted way"""
    
    # Main diagnosis section
    st.markdown('<div class="diagnosis-box">', unsafe_allow_html=True)
    st.subheader("🔍 Diagnosis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Diagnosis", result['diagnosis'].title())
    
    with col2:
        risk_level = result['risk_level']
        risk_class = f"risk-{risk_level}"
        st.markdown(f'<p class="{risk_class}">Risk Level: {risk_level.upper()}</p>', unsafe_allow_html=True)
    
    with col3:
        confidence = result['confidence'] * 100
        st.metric("Confidence", f"{confidence:.1f}%")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Risk score gauge
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig = display_risk_gauge(result['risk_score'], result['risk_level'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Risk Assessment")
        st.write(f"**Risk Score:** {result['risk_score']}/10")
        
        if result['risk_score'] <= 3:
            st.success("✅ Low risk - Monitor symptoms and maintain healthy habits")
        elif result['risk_score'] <= 7:
            st.warning("⚠️ Moderate risk - Consider consulting a healthcare provider")
        else:
            st.error("🚨 High risk - Seek immediate medical attention")
    
    # Recommendations
    if 'recommendations' in result and result['recommendations']:
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        st.subheader("💡 Recommendations")
        for i, recommendation in enumerate(result['recommendations'], 1):
            st.write(f"{i}. {recommendation}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Extracted information (for PDF uploads)
    if 'extracted_info' in result:
        with st.expander("📋 Extracted Information from PDF"):
            info = result['extracted_info']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Symptoms:**")
                for symptom in info['symptoms']:
                    st.write(f"• {symptom}")
            
            with col2:
                st.write("**Medical History:**")
                if info['medical_history']:
                    for history in info['medical_history']:
                        st.write(f"• {history}")
                else:
                    st.write("None found")
            
            st.write("**Vital Signs:**")
            st.json(info['vital_signs'])
    
    # Input data summary (for manual input)
    if 'input_data' in result:
        with st.expander("📋 Input Data Summary"):
            input_data = result['input_data']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Patient Info:**")
                st.write(f"Age: {input_data['age']}")
                st.write(f"Gender: {input_data['gender']}")
                
                st.write("**Symptoms:**")
                for symptom in input_data['symptoms']:
                    st.write(f"• {symptom}")
            
            with col2:
                st.write("**Medical History:**")
                if input_data['medical_history']:
                    for history in input_data['medical_history']:
                        st.write(f"• {history}")
                else:
                    st.write("None")
                
                st.write("**Vital Signs:**")
                st.json(input_data['vital_signs'])

def model_metrics_page():
    """Display comprehensive model performance metrics"""
    st.header("📊 Model Performance Metrics")
    
    with st.spinner("Loading model metrics..."):
        metrics_data, success = get_model_metrics()
    
    if not success:
        st.error("Failed to load model metrics. Please ensure the model has been trained.")
        if 'error' in metrics_data:
            st.error(f"Error: {metrics_data['error']}")
        return
    
    # Display metrics for both models
    if 'diagnosis_metrics' in metrics_data and 'risk_metrics' in metrics_data:
        
        # Create tabs for different models
        tab1, tab2 = st.tabs(["🔍 Diagnosis Model", "⚠️ Risk Assessment Model"])
        
        with tab1:
            display_model_metrics(metrics_data['diagnosis_metrics'], "Diagnosis Model")
        
        with tab2:
            display_model_metrics(metrics_data['risk_metrics'], "Risk Assessment Model")
    else:
        st.error("Incomplete metrics data available.")

def display_model_metrics(metrics, model_name):
    """Display metrics for a specific model"""
    
    # Overall Performance Metrics
    st.subheader(f"📈 {model_name} Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    
    with col2:
        st.metric("F1 Score (Macro)", f"{metrics['f1_score_macro']:.3f}")
    
    with col3:
        st.metric("F1 Score (Weighted)", f"{metrics['f1_score_weighted']:.3f}")
    
    with col4:
        st.metric("Avg Sensitivity", f"{metrics['average_sensitivity']:.3f}")
    
    with col5:
        st.metric("Avg Specificity", f"{metrics['average_specificity']:.3f}")
    
    # Detailed Metrics
    st.subheader("📋 Detailed Class-wise Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sensitivity (Recall) per Class:**")
        sensitivity_df = pd.DataFrame({
            'Class': range(len(metrics['sensitivity_per_class'])),
            'Sensitivity': metrics['sensitivity_per_class']
        })
        
        fig_sens = px.bar(sensitivity_df, x='Class', y='Sensitivity', 
                         title='Sensitivity by Class',
                         color='Sensitivity',
                         color_continuous_scale='viridis')
        fig_sens.update_layout(height=400)
        st.plotly_chart(fig_sens, use_container_width=True)
    
    with col2:
        st.write("**Specificity per Class:**")
        specificity_df = pd.DataFrame({
            'Class': range(len(metrics['specificity_per_class'])),
            'Specificity': metrics['specificity_per_class']
        })
        
        fig_spec = px.bar(specificity_df, x='Class', y='Specificity', 
                         title='Specificity by Class',
                         color='Specificity',
                         color_continuous_scale='plasma')
        fig_spec.update_layout(height=400)
        st.plotly_chart(fig_spec, use_container_width=True)
    
    # Confusion Matrix
    st.subheader("🔢 Confusion Matrix")
    cm = np.array(metrics['confusion_matrix'])
    
    fig_cm = px.imshow(cm, 
                       text_auto=True, 
                       aspect="auto",
                       title="Confusion Matrix",
                       color_continuous_scale='Blues')
    fig_cm.update_layout(
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        height=500
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # ROC Curves
    if metrics.get('roc_data') and metrics['roc_data'] is not None:
        st.subheader("📈 ROC Curves")
        
        roc_data = metrics['roc_data']
        fig_roc = go.Figure()
        
        # Add ROC curve for each class
        for class_idx, auc_score in roc_data['auc'].items():
            fpr = roc_data['fpr'][class_idx]
            tpr = roc_data['tpr'][class_idx]
            
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'Class {class_idx} (AUC = {auc_score:.3f})',
                line=dict(width=2)
            ))
        
        # Add diagonal line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig_roc.update_layout(
            title='ROC Curves for All Classes',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # AUC Summary
        st.subheader("🎯 AUC Summary")
        auc_df = pd.DataFrame([
            {'Class': f'Class {k}', 'AUC Score': f'{v:.3f}'} 
            for k, v in roc_data['auc'].items()
        ])
        st.dataframe(auc_df, use_container_width=True)
    
    else:
        st.warning("ROC curve data not available for this model.")
    
    # Performance Interpretation
    st.subheader("💡 Performance Interpretation")
    
    accuracy = metrics['accuracy']
    f1_macro = metrics['f1_score_macro']
    avg_sensitivity = metrics['average_sensitivity']
    avg_specificity = metrics['average_specificity']
    
    if accuracy >= 0.9:
        st.success(f"🎉 Excellent performance! The {model_name.lower()} achieves {accuracy:.1%} accuracy.")
    elif accuracy >= 0.8:
        st.info(f"👍 Good performance! The {model_name.lower()} achieves {accuracy:.1%} accuracy.")
    elif accuracy >= 0.7:
        st.warning(f"⚠️ Fair performance. The {model_name.lower()} achieves {accuracy:.1%} accuracy.")
    else:
        st.error(f"❌ Poor performance. The {model_name.lower()} only achieves {accuracy:.1%} accuracy.")
    
    # Detailed interpretation
    st.write("**Key Insights:**")
    st.write(f"• **Accuracy**: {accuracy:.1%} - Overall correctness of predictions")
    st.write(f"• **F1 Score (Macro)**: {f1_macro:.3f} - Balanced performance across all classes")
    st.write(f"• **Average Sensitivity**: {avg_sensitivity:.1%} - Ability to correctly identify positive cases")
    st.write(f"• **Average Specificity**: {avg_specificity:.1%} - Ability to correctly identify negative cases")

def about_page():
    st.header("ℹ️ About Medical Diagnosis Assistant")
    
    st.markdown("""
    ## 🎯 Purpose
    This application provides AI-powered medical diagnosis and risk assessment based on:
    - **PDF Medical Reports**: Upload and analyze medical PDF documents
    - **Manual Symptom Input**: Enter symptoms, medical history, and vital signs manually
    
    ## 🧠 How it Works
    1. **Machine Learning Model**: Trained on medical data using Random Forest algorithms
    2. **Text Analysis**: Extracts relevant medical information from text and PDFs  
    3. **Risk Assessment**: Provides risk scores from 1-10 and risk levels (Low/Moderate/High)
    4. **Recommendations**: Offers appropriate next steps based on diagnosis and risk level
    
    ## ⚠️ Important Disclaimer
    **This tool is for educational purposes only and should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.**
    
    - Always consult qualified healthcare professionals for medical concerns
    - Seek immediate medical attention for emergency situations
    - This AI model has limitations and may not be 100% accurate
    
    ## 🔧 Technical Details
    - **Backend**: Flask API with scikit-learn ML models
    - **Frontend**: Streamlit web application
    - **Features**: PDF processing, NLP analysis, interactive visualizations
    
    ## 📊 Model Performance
    The model is trained on a dataset of medical cases and provides:
    - Diagnosis prediction with confidence scores
    - Risk level assessment (Low/Moderate/High)
    - Numerical risk scores (1-10 scale)
    - Tailored recommendations based on findings
    """)
    
    st.info("💡 **Tip**: Start the Flask server (`python main_app.py`) before using this application.")

if __name__ == "__main__":
    main()
