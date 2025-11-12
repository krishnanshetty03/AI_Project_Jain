# 🏥 Medical Diagnosis Assistant

AI-powered medical diagnosis and risk assessment system with beautiful interactive interface.

## 🚀 Live Demo

- **Streamlit App**: [Deploy to Streamlit Cloud](https://streamlit.io/)
- **Web Interface**: [https://ai-project-jain-gx8t.vercel.app/](https://ai-project-jain-gx8t.vercel.app/)
- **API Backend**: [https://ai-project-jain-gx8t.vercel.app/api/](https://ai-project-jain-gx8t.vercel.app/api/)

## ✨ Features

- 🏥 **Medical Diagnosis**: AI-powered symptom analysis
- 📄 **PDF Analysis**: Upload and analyze medical reports
- 📊 **Interactive UI**: Beautiful Streamlit interface with charts
- 🎯 **Risk Assessment**: Low/Moderate/High risk levels with scores
- 💡 **Smart Recommendations**: Personalized medical advice
- 📈 **Model Metrics**: Performance visualization and statistics

## 🛠️ Tech Stack

- **Frontend**: Streamlit + Plotly + HTML/CSS/JavaScript
- **Backend**: Vercel Serverless Functions (Python)
- **ML**: Rule-based medical logic (deployable without heavy ML libraries)
- **Deployment**: Streamlit Community Cloud + Vercel

## 🎯 Quick Start

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/krishnanshetty03/AI_Project_Jain.git
   cd AI_Project_Jain
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

### 🌐 Deploy to Streamlit Cloud

1. **Fork this repository** to your GitHub account
2. **Visit** [share.streamlit.io](https://share.streamlit.io/)
3. **Connect your GitHub** and select this repository
4. **Set main file path**: `streamlit_app.py`
5. **Deploy** - Your app will be live in minutes!

## 📱 Usage

### Web Interface (HTML)
- Visit the live web app
- Enter patient information and symptoms
- Get instant diagnosis with recommendations

### Streamlit Interface (Full Features)
- **PDF Upload**: Analyze medical PDF reports
- **Manual Input**: Enter symptoms and vital signs manually
- **Model Metrics**: View detailed performance statistics
- **Interactive Charts**: Risk gauges, confusion matrices, ROC curves

## 🏗️ Architecture

```
┌─────────────────────┐    ┌─────────────────────┐
│   Streamlit App     │    │   Web Interface     │
│  (Full Features)    │    │  (Simple & Fast)    │
└─────────┬───────────┘    └─────────┬───────────┘
          │                          │
          └──────────┬─────────────────┘
                     │
          ┌─────────────────────┐
          │   Vercel API        │
          │ (Python Backend)    │
          │  Medical Logic      │
          └─────────────────────┘
```

## 🤖 API Endpoints

- **GET** `/api/` - Health check
- **POST** `/api/` - Medical diagnosis

### Sample API Request
```json
{
  "symptoms": ["fever", "headache", "fatigue"],
  "medical_history": ["diabetes"],
  "age": 35,
  "gender": "female",
  "vital_signs": {
    "temperature": 101.5,
    "blood_pressure": "130/85",
    "heart_rate": 88
  }
}
```

## ⚠️ Disclaimer

**This tool is for educational purposes only and should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.**

- Always consult qualified healthcare professionals
- Seek immediate medical attention for emergencies
- This AI model has limitations and may not be 100% accurate

