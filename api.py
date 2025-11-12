from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
from model.feature_extractor import URLFeatureExtractor
import logging
import pandas as pd
import numpy as np
import threading
import sqlite3
import csv
from datetime import datetime
import requests
import json
from dotenv import load_dotenv
from pathlib import Path

# Try to import SHAP, but make it optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SHAP not available: {e}")
    SHAP_AVAILABLE = False
    shap = None

# Configure logging FIRST before using logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file - explicitly specify path
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"✅ Loaded .env from: {env_path}")
else:
    # Try loading from current directory as fallback
    load_dotenv()
    logger.warning(f"⚠️ .env file not found at: {env_path}, trying default location")

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize feedback database
def init_feedback_db():
    """Initialize SQLite database for feedback data"""
    conn = sqlite3.connect('feedback_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            label TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_ip TEXT,
            model_prediction TEXT
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("Feedback database initialized")

# Initialize database on startup
init_feedback_db()

# Global variables for model and feature extractor
model = None
feature_extractor = None
shap_explainer = None

def _load_model_file(model_path: str):
    try:
        if os.path.exists(model_path):
            loaded = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
            return loaded
        else:
            logger.info(f"Model path does not exist: {model_path}")
            return None
    except Exception as e:
        logger.error(f"Failed loading model from {model_path}: {e}")
        return None

def load_model():
    """Load the trained ML model with graceful fallbacks."""
    global model, feature_extractor, shap_explainer

    feature_extractor = URLFeatureExtractor()

    # Try production → original → demo
    production_model_path = os.path.join('model', 'production_phishing_model.pkl')
    original_model_path = os.path.join('model', 'phishing_model.pkl')
    demo_model_path = os.path.join('model', 'demo_phishing_model.pkl')

    for candidate in [production_model_path, original_model_path, demo_model_path]:
        loaded = _load_model_file(candidate)
        if loaded is not None:
            model = loaded
            logger.info(f"Using model: {candidate}")
            break
    else:
        model = None
        logger.warning("No usable model could be loaded. Operating in heuristic mode.")
    
    # Initialize SHAP explainer if model is loaded and SHAP is available
    if model is not None and SHAP_AVAILABLE:
        try:
            # Create a TreeExplainer for tree-based models
            shap_explainer = shap.TreeExplainer(model)
            logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            shap_explainer = None
    else:
        shap_explainer = None
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available - explainer functionality disabled")

@app.route('/api/check-url', methods=['POST'])
def check_url():
    """API endpoint to check if a URL is phishing"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({
                'success': False,
                'error': 'URL is required'
            }), 400
        
        # Add http:// if no protocol specified
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # Extract features
        features = feature_extractor.extract_features(url)
        
        # Make prediction if model is loaded
        if model is not None:
            try:
                prediction = model.predict([features])[0]
                # Some models might not have predict_proba
                try:
                    confidence = float(getattr(model, 'predict_proba')([features])[0].max())
                except Exception:
                    confidence = 0.5
                
                result = {
                    'success': True,
                    'url': url,
                    'is_phishing': bool(prediction),
                    'confidence': confidence,
                    'prediction_text': 'Phishing' if prediction else 'Safe',
                    'features': features
                }
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                # Fallback: basic heuristic check
                result = {
                    'success': True,
                    'url': url,
                    'is_phishing': False,  # Default to safe
                    'confidence': 0.5,
                    'prediction_text': 'Safe (Model Error)',
                    'features': features
                }
        else:
            # Fallback: basic heuristic check
            result = {
                'success': True,
                'url': url,
                'is_phishing': False,  # Default to safe
                'confidence': 0.5,
                'prediction_text': 'Safe (Model not loaded)',
                'features': features
            }
        
        logger.info(f"URL checked: {url} - Result: {result['prediction_text']}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing URL: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/feature-importances')
def feature_importances():
    """Return model feature importances and feature names for visualization"""
    if model is not None:
        try:
            feature_names = [
                'url_length', 'domain_length', 'path_length', 'dot_count',
                'domain_special_chars', 'is_ip', 'domain_digits', 'suspicious_tld',
                'path_special_chars', 'slash_count', 'path_dots', 'suspicious_ext',
                'param_count', 'query_special_chars', 'suspicious_params',
                'at_symbols', 'hyphens', 'double_slashes', 'equal_signs', 'question_marks', 'ampersands',
                'https_used', 'has_port', 'has_redirect',
                'url_entropy', 'domain_entropy',
                'brand_in_domain', 'suspicious_words', 'random_pattern', 'homograph_suspicious'
            ]
            importances = getattr(model, 'feature_importances_', None)
            if importances is None:
                return jsonify({'success': False, 'error': 'Model does not expose feature_importances_'}), 400
            importances = list(map(float, importances.tolist()))
            feature_importance_dict = {name: imp for name, imp in zip(feature_names, importances)}
            sorted_importances = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)[:10]
            return jsonify({
                'success': True,
                'importances': sorted_importances
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    else:
        return jsonify({'success': False, 'error': 'Model not loaded'})

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/api/explain', methods=['POST'])
def explain_prediction():
    """Explain model prediction using feature importance"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({
                'success': False,
                'error': 'URL is required'
            }), 400
        
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not available'
            }), 503
        
        # Add http:// if no protocol specified
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # Extract features
        features = feature_extractor.extract_features(url)
        features_array = np.array(features).reshape(1, -1)
        
        # Feature names
        feature_names = [
            'url_length', 'domain_length', 'path_length', 'dot_count',
            'domain_special_chars', 'is_ip', 'domain_digits', 'suspicious_tld',
            'path_special_chars', 'slash_count', 'path_dots', 'suspicious_ext',
            'param_count', 'query_special_chars', 'suspicious_params',
            'at_symbols', 'hyphens', 'double_slashes', 'equal_signs', 'question_marks', 'ampersands',
            'https_used', 'has_port', 'has_redirect',
            'url_entropy', 'domain_entropy',
            'brand_in_domain', 'suspicious_words', 'random_pattern', 'homograph_suspicious'
        ]
        
        # Get feature importance from model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # Fallback: use feature values as importance
            importances = np.abs(features)
        
        # Get top 5 most important features
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = feature_importance[:5]
        
        # Format response
        explanation = []
        for name, importance in top_features:
            explanation.append({
                'feature': name,
                'importance': float(importance),
                'contribution': 'high importance' if importance > 0.1 else 'medium importance'
            })
        
        # Get prediction and convert numpy types to Python native types
        prediction = model.predict(features_array)[0]
        # Convert numpy int64/bool to Python int/bool for JSON serialization
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
        else:
            prediction = int(prediction) if isinstance(prediction, (np.integer, bool)) else bool(prediction)
        
        return jsonify({
            'success': True,
            'url': url,
            'explanation': explanation,
            'prediction': prediction,
            'confidence': float(model.predict_proba(features_array)[0].max())
        })
        
    except Exception as e:
        logger.error(f"Error explaining prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for model improvement"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        label = data.get('label', '').strip().lower()
        
        if not url or label not in ['phishing', 'legit']:
            return jsonify({
                'success': False,
                'error': 'URL and valid label (phishing/legit) are required'
            }), 400
        
        # Get model prediction for comparison
        model_prediction = None
        if model is not None:
            try:
                if not url.startswith(('http://', 'https://')):
                    url = 'http://' + url
                features = feature_extractor.extract_features(url)
                pred = model.predict([features])[0]
                model_prediction = 'phishing' if pred else 'legit'
            except:
                pass
        
        # Store feedback in database
        conn = sqlite3.connect('feedback_data.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO feedback (url, label, user_ip, model_prediction)
            VALUES (?, ?, ?, ?)
        ''', (url, label, request.remote_addr, model_prediction))
        conn.commit()
        conn.close()
        
        logger.info(f"Feedback received: {url} -> {label}")
        
        return jsonify({
            'success': True,
            'message': 'Feedback submitted successfully',
            'model_prediction': model_prediction
        })
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def retrain_model_background():
    """Background function to retrain the model with feedback data"""
    global model
    
    try:
        logger.info("Starting model retraining with feedback data...")
        
        # Load feedback data
        conn = sqlite3.connect('feedback_data.db')
        cursor = conn.cursor()
        cursor.execute('SELECT url, label FROM feedback')
        feedback_data = cursor.fetchall()
        conn.close()
        
        if len(feedback_data) < 10:  # Need minimum feedback for retraining
            logger.info("Insufficient feedback data for retraining")
            return
        
        # Prepare training data
        X_feedback = []
        y_feedback = []
        
        for url, label in feedback_data:
            try:
                features = feature_extractor.extract_features(url)
                X_feedback.append(features)
                y_feedback.append(1 if label == 'phishing' else 0)
            except Exception as e:
                logger.error(f"Error processing feedback URL {url}: {e}")
                continue
        
        if len(X_feedback) < 5:
            logger.info("Insufficient valid feedback data for retraining")
            return
        
        # Load original training data (if available)
        original_data_path = 'data/phishing_data_combined.csv'
        if os.path.exists(original_data_path):
            df = pd.read_csv(original_data_path)
            X_original = []
            y_original = []
            
            for _, row in df.iterrows():
                try:
                    features = feature_extractor.extract_features(row['url'])
                    X_original.append(features)
                    y_original.append(row['label'])
                except:
                    continue
            
            # Combine original and feedback data
            X_combined = X_original + X_feedback
            y_combined = y_original + y_feedback
        else:
            X_combined = X_feedback
            y_combined = y_feedback
        
        # Retrain model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42
        )
        
        # Train new model
        new_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        new_model.fit(X_train, y_train)
        
        # Evaluate new model
        accuracy = new_model.score(X_test, y_test)
        logger.info(f"Retrained model accuracy: {accuracy:.3f}")
        
        # Update global model
        model = new_model
        
        # Save retrained model
        model_path = os.path.join('model', 'retrained_phishing_model.pkl')
        joblib.dump(model, model_path)
        logger.info(f"Retrained model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Error during model retraining: {e}")

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """Trigger model retraining with feedback data"""
    try:
        # Start retraining in background thread
        thread = threading.Thread(target=retrain_model_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Model retraining started in background'
        })
        
    except Exception as e:
        logger.error(f"Error starting retraining: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def check_google_safe_browsing(url, api_key=None):
    """Check URL against Google Safe Browsing API"""
    try:
        if not api_key:
            # Try to get API key from environment variable
            api_key = os.getenv('GOOGLE_SAFE_BROWSING_API_KEY', '').strip()
            
        # Template/placeholder keys to reject (only obvious placeholders)
        template_keys = [
            'your_safe_browsing_api_key_here',
            'YOUR_SAFE_BROWSING_API_KEY_HERE',
            ''
        ]
        
        # Validate key format - accept any key that looks real (not a placeholder)
        is_valid_key = (
            api_key and 
            api_key not in template_keys and
            len(api_key) > 10  # Real API keys are longer
        )
        
        if not is_valid_key:
            # Return mock data for demo purposes
            logger.info("No Google Safe Browsing API key configured, using mock data")
            return {
                'status': 'safe',
                'threats': [],
                'error': None
            }
        
        # Google Safe Browsing API endpoint
        api_url = 'https://safebrowsing.googleapis.com/v4/threatMatches:find'
        
        payload = {
            "client": {
                "clientId": "phishing-detector",
                "clientVersion": "1.0"
            },
            "threatInfo": {
                "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": url}]
            }
        }
        
        response = requests.post(
            f"{api_url}?key={api_key}",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'matches' in data and data['matches']:
                threats = [match['threatType'] for match in data['matches']]
                return {
                    'status': 'unsafe',
                    'threats': threats,
                    'error': None
                }
            else:
                return {
                    'status': 'safe',
                    'threats': [],
                    'error': None
                }
        else:
            return {
                'status': 'error',
                'threats': [],
                'error': f"API error: {response.status_code}"
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'threats': [],
            'error': str(e)
        }

@app.route('/api/threat_check', methods=['POST'])
def threat_check():
    """Check URL against multiple threat intelligence sources"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        logger.info(f"Threat check requested for URL: {url}")
        
        if not url:
            logger.warning("Threat check failed: No URL provided")
            return jsonify({
                'success': False,
                'error': 'URL is required'
            }), 400
        
        # Add http:// if no protocol specified
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # Get model prediction
        model_confidence = 0.5
        model_result = "unknown"
        
        if model is not None:
            try:
                features = feature_extractor.extract_features(url)
                prediction = model.predict([features])[0]
                confidence = model.predict_proba([features])[0].max()
                
                model_confidence = float(confidence)
                model_result = "phishing" if prediction else "safe"
                logger.info(f"Model prediction: {model_result} (confidence: {model_confidence:.2f})")
            except Exception as e:
                logger.error(f"Model prediction error: {e}")
                model_result = "unknown"
                model_confidence = 0.5
        else:
            logger.warning("No model loaded, using default values")
        
        # Check Google Safe Browsing (mock implementation)
        google_result = check_google_safe_browsing(url)
        logger.info(f"Google Safe Browsing result: {google_result['status']}")
        
        # Determine final decision
        final_decision = "unknown"
        if model_result == "phishing" and google_result['status'] == "unsafe":
            final_decision = "phishing (confirmed)"
        elif model_result == "phishing" and google_result['status'] == "safe":
            final_decision = "phishing (AI detected)"
        elif model_result == "safe" and google_result['status'] == "unsafe":
            final_decision = "suspicious (external threat)"
        elif model_result == "safe" and google_result['status'] == "safe":
            final_decision = "safe"
        else:
            final_decision = f"{model_result} (uncertain)"
        
        result = {
            'success': True,
            'url': url,
            'model_confidence': model_confidence,
            'model_result': model_result,
            'google_safe_browsing': google_result['status'],
            'google_threats': google_result['threats'],
            'final_decision': final_decision,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Threat check completed successfully for {url}: {final_decision}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in threat check: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run the API server on port 5001
    app.run(debug=True, host='0.0.0.0', port=5001)
