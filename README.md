# Real-time Phishing URL Detection

A professional web application for detecting phishing URLs using advanced machine learning. It features a modern React frontend and a Flask API backend with CORS enabled.

## 🚀 Features

- **Real-time URL Analysis**: Instant phishing detection with 30+ extracted features
- **Confidence & Insights**: Probability score and top feature importances
- **Modern UI**: React + Tailwind CSS, responsive and fast
- **Privacy First**: Local processing with no data retention
- **RESTful API**: Clean endpoints for integration

## 🧱 Architecture

- **Frontend**: React 18, Tailwind CSS, Lucide React icons, Recharts
- **Backend**: Flask API with CORS
- **Model**: Tree-based classifier compatible with scikit-learn/XGBoost

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

## 🔧 Installation

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd Real-time-Phishing-URL-Detection-main
   ```

2. Install Python dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Install frontend dependencies
   ```bash
   cd frontend
   npm install
   ```

## ▶️ Running the Application

1. Start the Flask API backend
   ```bash
   # From the repository root
   python api.py
   ```
   The API will be available at `http://localhost:5001`.

2. Start the React frontend
   ```bash
   # From the frontend directory
   npm start
   ```
   The app will be available at `http://localhost:3000`.

## 🔌 API Endpoints

- `POST /api/check-url` — Check if a URL is phishing
- `GET /api/feature-importances` — Get model feature importance data
- `GET /api/health` — Health check

Example request:
```bash
curl -s -X POST http://localhost:5001/api/check-url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com"}'
```

## 🔍 Model Information

The system evaluates URLs using features like:
- URL length and structure
- Domain characteristics and entropy
- Special character patterns and counts
- Suspicious keywords and TLDs
- HTTPS usage, redirects, and ports
- Brand name and homograph indicators

## 🧑‍💻 Development

### Frontend
- Located in `frontend/`
- React hooks and functional components
- Tailwind CSS styling, Recharts visualizations

### Backend
- `api.py` exposes REST endpoints
- CORS enabled for the frontend
- Graceful model loading with fallbacks

## 📁 Project Structure

```
├── api.py                          # Flask API backend
├── requirements.txt                # Python dependencies
├── frontend/                       # React frontend
│   ├── package.json               # Node.js dependencies
│   ├── public/                    # Static files
│   ├── src/                       # React source code
│   │   ├── components/            # React components
│   │   ├── App.js                # Main app component
│   │   └── index.js              # Entry point
│   └── tailwind.config.js        # Tailwind configuration
├── model/                         # ML model files
├── data/                          # Training data
├── templates/                     # Optional Flask templates
├── static/                        # Optional Flask static assets
└── utils/                         # Utility functions
```


 


