# Phishing URL Detection System

A real-time machine learning-based web application for detecting phishing URLs using advanced feature extraction and classification algorithms.

## 🚀 Features

- **Real-time URL Analysis**: Instant phishing detection with 30+ extracted features
- **Machine Learning Model**: Random Forest classifier trained on synthetic data
- **Modern Web Interface**: Clean, responsive UI with real-time feedback
- **Feature Breakdown**: Detailed analysis showing which factors contributed to the prediction
- **Confidence Scoring**: Probability-based confidence levels for predictions
- **API Endpoint**: RESTful API for integration with other applications

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn, Random Forest
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Feature Extraction**: Custom URL analysis algorithms
- **Styling**: Modern CSS with gradients and animations

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd PHISHING_URL_DETECTION
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (First time setup)
   ```bash
   python train_model.py
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your browser and navigate to: `http://localhost:5000`

## 📊 Model Training

The application includes a training script that generates synthetic data and trains a Random Forest model:

```bash
python train_model.py
```

This will:
- Generate 2000 synthetic URL samples (1000 safe, 1000 phishing)
- Train a Random Forest classifier
- Save the model to `model/phishing_model.pkl`
- Display model performance metrics
- Test with example URLs

## 🔍 Features Extracted

The system analyzes URLs using 30+ features:

### URL Structure Features
- URL length, domain length, path length
- Number of dots, slashes, special characters
- Query parameter count and analysis

### Security Features
- HTTPS usage detection
- Port number presence
- Redirect detection
- Suspicious file extensions

### Domain Analysis
- IP address detection
- Suspicious TLD identification (.tk, .ml, .ga, .cf, .gq)
- Domain entropy calculation
- Brand name presence in domain

### Pattern Analysis
- Suspicious keyword detection
- Random string pattern analysis
- Homograph attack detection
- Special character frequency

## 🎯 Usage

### Web Interface
1. Enter a URL in the input field
2. Click "Check URL" or press Enter
3. View the prediction result with confidence score
4. Explore the feature breakdown for detailed analysis

### API Usage
```bash
curl -X POST http://localhost:5000/check_url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

Response:
```json
{
  "success": true,
  "url": "https://example.com",
  "is_phishing": false,
  "confidence": 0.85,
  "prediction_text": "Safe",
  "features": {
    "url_length": 19,
    "domain_length": 11,
    "https_used": 1,
    ...
  }
}
```

## 🧪 Testing

Test the system with these example URLs:

**Safe URLs:**
- `https://www.google.com`
- `https://www.facebook.com`
- `https://www.amazon.com`

**Phishing URLs:**
- `http://secure-google-verify.tk/login`
- `http://paypal-secure-verify.ml/account`
- `http://facebook-login-verify.ga/auth`

## 📁 Project Structure

```
PHISHING_URL_DETECTION/
├── app.py                 # Main Flask application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── model/
│   ├── __init__.py       # Package initialization
│   ├── feature_extractor.py  # URL feature extraction
│   └── phishing_model.pkl    # Trained ML model (generated)
├── templates/
│   └── index.html        # Main web interface
├── static/
│   ├── style.css         # CSS styling
│   └── script.js         # Frontend JavaScript
└── utils/                # Utility functions (future)
```
 

## 🚀 Deployment

### Local Development
```bash
python app.py
```

## 📈 Performance

- **Response Time**: < 500ms for URL analysis
- **Accuracy**: ~95% on synthetic test data
- **Features**: 30+ extracted features per URL
- **Concurrent Users**: Supports multiple simultaneous requests

 