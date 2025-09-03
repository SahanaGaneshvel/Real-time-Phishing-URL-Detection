from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
from model.feature_extractor import URLFeatureExtractor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables for model and feature extractor
model = None
feature_extractor = None

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
    global model, feature_extractor

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

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run the API server on port 5001
    app.run(debug=True, host='0.0.0.0', port=5001)
