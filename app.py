from flask import Flask, render_template, request, jsonify
import joblib
import os
from model.feature_extractor import URLFeatureExtractor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Global variables for model and feature extractor
model = None
feature_extractor = None

def load_model():
    """Load the trained ML model"""
    global model, feature_extractor
    try:
        # Try to load production model first
        production_model_path = os.path.join('model', 'production_phishing_model.pkl')
        production_extractor_path = os.path.join('model', 'production_phishing_model_extractor.pkl')
        
        if os.path.exists(production_model_path) and os.path.exists(production_extractor_path):
            model = joblib.load(production_model_path)
            feature_extractor = joblib.load(production_extractor_path)
            logger.info("Production model loaded successfully")
        else:
            # Fallback to original model
            model_path = os.path.join('model', 'phishing_model.pkl')
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                feature_extractor = URLFeatureExtractor()
                logger.info("Original model loaded successfully")
            else:
                logger.warning("No model files found. Please train the model first.")
                model = None
                feature_extractor = URLFeatureExtractor()
        
        if feature_extractor is None:
            feature_extractor = URLFeatureExtractor()
        logger.info("Feature extractor initialized")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
        feature_extractor = URLFeatureExtractor()

@app.route('/')
def index():
    """Main page with URL input form"""
    return render_template('index.html')

@app.route('/check_url', methods=['POST'])
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
            prediction = model.predict([features])[0]
            confidence = model.predict_proba([features])[0].max()
            
            result = {
                'success': True,
                'url': url,
                'is_phishing': bool(prediction),
                'confidence': float(confidence),
                'prediction_text': 'Phishing' if prediction else 'Safe',
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

@app.route('/feature_importances')
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
            importances = model.feature_importances_.tolist()
            feature_importance_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}
            sorted_importances = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)[:10]
            return jsonify({
                'success': True,
                'importances': sorted_importances
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    else:
        return jsonify({'success': False, 'error': 'Model not loaded'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000) 