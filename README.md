# 🏥 Student Health Analytics

An AI-powered student wellness platform built with Streamlit, LangGraph & Groq AI.

## Features
- 📊 Dashboard with key health metrics
- 📈 Advanced analytics & correlation heatmaps
- 👤 Individual student health profiles
- 🧠 Mental health risk prediction (Random Forest)
- 📄 PDF health report analyzer (RAG)
- 🤖 AI health assistant chatbot

## Setup

### 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/student-health-analytics.git
cd student-health-analytics

### 2. Install dependencies
pip install -r requirements.txt

### 3. Set your API key
Create `.streamlit/secrets.toml`:
GROQ_API_KEY = "your_groq_api_key"

### 4. Run the app
streamlit run app.py

## Tech Stack
- Streamlit · Plotly · Pandas · scikit-learn
- LangChain · LangGraph · Groq (Llama 3.1)
- HuggingFace Embeddings · ChromaDB (RAG)
- pdfplumber