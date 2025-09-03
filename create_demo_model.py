import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

def create_demo_model():
    """Create a simple demo model for testing"""
    print("Creating demo model...")
    
    # Create some dummy feature importances
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
    
    # Generate some dummy feature importances
    importances = np.random.rand(len(feature_names))
    importances = importances / importances.sum()  # Normalize
    
    # Set some realistic importances
    importances[0] = 0.15  # url_length
    importances[1] = 0.12  # domain_length
    importances[4] = 0.10  # domain_special_chars
    importances[7] = 0.08  # suspicious_tld
    importances[21] = 0.06  # https_used
    importances[24] = 0.05  # url_entropy
    importances[25] = 0.05  # domain_entropy
    
    # Create dummy training data with realistic patterns
    n_samples = 1000
    X = np.random.rand(n_samples, len(feature_names))
    
    # Create labels based on some heuristic rules
    y = np.zeros(n_samples)
    for i in range(n_samples):
        # Simple heuristic: longer URLs with special chars are more likely to be phishing
        if (X[i, 0] > 0.7 and X[i, 4] > 0.5) or (X[i, 7] > 0.8) or (X[i, 21] < 0.3):
            y[i] = 1
    
    # Create a simple Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X, y)
    
    # Save the model
    model_path = os.path.join('model', 'demo_phishing_model.pkl')
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, model_path)
    
    print(f"Demo model saved to {model_path}")
    print("Model feature importances:")
    for name, importance in zip(feature_names, model.feature_importances_):
        print(f"  {name}: {importance:.4f}")
    
    return model_path

if __name__ == "__main__":
    create_demo_model()
