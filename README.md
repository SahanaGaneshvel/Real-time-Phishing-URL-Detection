# 🛡️ Real-time Phishing URL Detection System

A comprehensive AI-powered phishing detection system with an intelligent chatbot for cybersecurity education. This system combines machine learning models with Retrieval-Augmented Generation (RAG) technology to provide real-time phishing detection and educational support.

## 🌟 Key Capabilities

### 🔍 **Real-Time Phishing Detection**
- **Instant URL Scoring**: Production XGBoost model served through `/api/check-url`
- **Explainable AI**: SHAP-backed feature attributions via `/api/explain`
- **Threat Intelligence Bridge**: Optional Google Safe Browsing enrichment
- **Resilient Fallbacks**: Heuristic scoring when models or dependencies are missing

### 🧠 **Adaptive Knowledge & Feedback Loop**
- **Feedback Capture**: Users can label URLs; stored in `feedback_data.db`
- **Continuous Learning**: Background job-ready retraining pipeline fed by feedback
- **Feature Analytics**: `/api/feature-importances` exposes top influencing signals
- **WHOIS & URL Intelligence**: Rich feature extraction handled in `model/feature_extractor.py`

### 🤖 **AI Security Coach**
- **Gemini-Powered RAG**: Retrieval-Augmented responses grounded in curated cyber content
- **Vector Knowledge Base**: FAISS index persisted under `phishing_knowledge_db/`
- **Graceful Degradation**: Local embeddings and contextual fallbacks if Gemini is unavailable
- **Actionable Guidance**: Short, user-friendly answers focused on phishing prevention

### 🎛️ **Operational Tooling & UX**
- **One-Command Orchestration**: `start.py` boots API, chatbot, and React UI with health checks
- **Smart Dependency Checks**: Auto-verifies Python, Node.js, npm, and key libraries
- **Cross-Platform Support**: Works on Windows, macOS, and Linux (`start.sh` provided for Unix shells)
- **Modern Frontend**: React + Tailwind dashboard with live detection results and chatbot

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- Node.js 16+
- npm or yarn
- Google Gemini API key (for chatbot)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Real-time-Phishing-URL-Detection
```

2. **(Optional) Create a virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Create your `.env` file (project root)**
```env
# Google Gemini API Key (Required for chatbot)
GEMINI_API_KEY=your_gemini_api_key_here
# Alternatively, you can set GOOGLE_API_KEY for Gemini

# Google Safe Browsing API Key (Optional)
GOOGLE_SAFE_BROWSING_API_KEY=your_safe_browsing_api_key_here

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True

# Database Configuration
DATABASE_URL=sqlite:///feedback_data.db

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

4. **Start the system**
```bash
python start.py
```

The system will:
- ✅ Check and install all dependencies automatically
- ✅ Start the backend API (Port 5001)
- ✅ Start the RAG chatbot (Port 5002)
- ✅ Start the React frontend (Port 3000)

### Manual Installation

If you prefer manual installation:

1. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

2. **Install frontend dependencies**
```bash
cd frontend
npm install
cd ..
```

3. **Start services individually**
```bash
# Terminal 1: Backend API
python api.py

# Terminal 2: RAG Chatbot
python rag_backend.py

# Terminal 3: Frontend
cd frontend && npm start
```

## 🏗️ System Architecture

### Backend Services

#### 1. **Main API Server** (`api.py`)
- **Port**: 5001
- **Purpose**: Phishing URL detection and analysis
- **Features**:
  - URL feature extraction
  - ML model inference
  - Threat intelligence integration
  - Feedback collection
  - SHAP explanations

#### 2. **RAG Chatbot Server** (`rag_backend.py`)
- **Port**: 5002
- **Purpose**: AI-powered cybersecurity education
- **Features**:
  - Gemini AI integration
  - Vector-based knowledge retrieval
  - Context-aware responses
  - Phishing education content

### Frontend Application

#### **React Web Interface** (`frontend/`)
- **Port**: 3000
- **Purpose**: User interface for the detection system
- **Features**:
  - URL input and analysis
  - Real-time results display
  - Interactive chatbot
  - Visual analytics
  - Mobile-responsive design

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Google Gemini AI API key for chatbot | Yes (one of them) |
| `GOOGLE_SAFE_BROWSING_API_KEY` | Google Safe Browsing API key | No |
| `FLASK_ENV` | Flask environment (development/production) | No |
| `FLASK_DEBUG` | Enable Flask debug mode | No |
| `DATABASE_URL` | Database connection string | No |
| `CORS_ORIGINS` | Allowed CORS origins | No |

### API Endpoints

#### Main API (`http://localhost:5001`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/check-url` | POST | Predict if URL is phishing |
| `/api/explain` | POST | Get SHAP explanations |
| `/api/feedback` | POST | Submit feedback |
| `/api/feature-importances` | GET | Get top feature importance scores |

#### RAG Chatbot API (`http://localhost:5002`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/rag/health` | GET | RAG system health check |
| `/status` | GET | Service status and configuration details |
| `/test_gemini` | GET | Validate Gemini API connectivity |
| `/api/chat` | POST | Send message to chatbot (frontend endpoint) |
| `/query` | POST | Direct RAG query endpoint |

## 🤖 Chatbot Features

### **True RAG Implementation**
The chatbot uses Retrieval-Augmented Generation (RAG) to provide accurate, context-aware responses:

1. **Knowledge Base**: Comprehensive phishing education content
2. **Vector Search**: Semantic similarity search for relevant information
3. **Context Retrieval**: Retrieves relevant chunks based on user queries
4. **Gemini AI**: Generates responses using retrieved context
5. **Educational Focus**: Specialized in cybersecurity and phishing prevention

### **Chatbot Capabilities**
- ✅ Phishing attack identification and prevention
- ✅ Online safety best practices
- ✅ Cybersecurity education
- ✅ Real-time threat analysis assistance
- ✅ Interactive learning experience

## 📊 Machine Learning Models

### **Phishing Detection Model**
- **Algorithm**: XGBoost with feature engineering
- **Features**: 50+ URL-based features including:
  - Domain characteristics
  - URL structure analysis
  - SSL certificate validation
  - WHOIS information
  - Suspicious patterns
- **Accuracy**: High precision and recall on phishing datasets

### **Feature Engineering**
- **URL Parsing**: Domain, path, query parameters analysis
- **Domain Analysis**: TLD extraction, subdomain detection
- **Security Features**: HTTPS validation, certificate analysis
- **Pattern Recognition**: Suspicious character patterns
- **WHOIS Integration**: Domain registration information

## 🛠️ Development

### **Project Structure**
```
Real-time-Phishing-URL-Detection/
├── api.py                     # Main Flask API (port 5001)
├── rag_backend.py             # RAG chatbot service (port 5002)
├── start.py                   # Cross-platform orchestrator script
├── start.sh                   # Helper shell script for Unix shells
├── requirements.txt           # Python dependencies
├── feedback_data.db           # SQLite feedback store (created at runtime)
├── model/                     # ML models and feature extraction
│   ├── feature_extractor.py
│   ├── phishing_model.pkl
│   ├── production_phishing_model.pkl
│   └── production_phishing_model_metadata.json
├── utils/                     # Supporting utilities
│   └── whois_lookup.py
├── phishing_knowledge_db/     # Persisted FAISS index for RAG
│   ├── index.faiss
│   └── index.pkl
├── data/                      # Knowledge base and datasets
│   └── phishing_education/
│       └── content.txt
├── frontend/                  # React frontend application
│   ├── package.json
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── api.js
│       ├── App.js
│       ├── components/
│       │   ├── FeatureChart.js
│       │   ├── FeatureExplanationCard.js
│       │   ├── Header.js
│       │   ├── PhishingChatbot.js
│       │   ├── ResultDisplay.js
│       │   ├── ThreatInsightsPanel.js
│       │   └── URLInput.js
│       ├── index.css
│       └── index.js
├── static/                    # Flask static assets (emails, logos, etc.)
├── templates/                 # Flask templates for landing pages
└── README.md
```

### **Adding New Features**

1. **Backend API**: Modify `api.py` for new endpoints
2. **Chatbot**: Update `rag_backend.py` for new chatbot features
3. **Frontend**: Add components in `frontend/src/components/`
4. **Models**: Update feature extraction in `model/feature_extractor.py`

### **Testing**

```bash
# API health checks
curl http://localhost:5001/api/health
curl http://localhost:5002/api/rag/health

# URL scoring example
curl -X POST http://localhost:5001/api/check-url -H "Content-Type: application/json" -d "{\"url\": \"https://example.com/secure-login\"}"

# Chatbot question example
curl -X POST http://localhost:5002/api/chat -H "Content-Type: application/json" -d "{\"question\": \"How do I spot a phishing email?\"}"
```

## 🔒 Security Features

### **Input Validation**
- URL format validation
- Malicious input filtering
- Rate limiting protection
- CORS configuration

### **Data Protection**
- Secure API key storage
- Environment variable protection
- Database encryption
- Secure communication (HTTPS ready)

## 📈 Performance

### **Optimization Features**
- **Caching**: Model and feature caching for faster responses
- **Async Processing**: Non-blocking API operations
- **Resource Management**: Efficient memory and CPU usage
- **Scalability**: Ready for horizontal scaling

### **Monitoring**
- Health check endpoints
- Performance metrics
- Error logging
- Usage analytics

 

## Acknowledgments

- **Google Gemini AI** for providing the AI chatbot capabilities
- **Google Safe Browsing API** for threat intelligence
- **OpenAI** for additional AI model support
- **LangChain** for RAG implementation
- **React** and **Tailwind CSS** for the frontend framework

 
## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Advanced threat intelligence integration
- [ ] Mobile app development
- [ ] Machine learning model improvements
- [ ] Real-time threat monitoring dashboard
- [ ] API rate limiting and authentication
- [ ] Docker containerization
- [ ] Cloud deployment support

---

**Built with ❤️ for cybersecurity education and phishing prevention**
