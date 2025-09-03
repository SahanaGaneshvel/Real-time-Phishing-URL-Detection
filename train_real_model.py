import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our feature extractor
from model.feature_extractor import URLFeatureExtractor

class RealModelTrainer:
    def __init__(self):
        self.feature_extractor = URLFeatureExtractor()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        
    def load_real_data(self, data_file=None):
        """Load real phishing data from CSV file"""
        print("Loading real phishing data...")
        
        if data_file is None:
            # Find the most recent combined data file
            data_files = glob.glob("data/real_phishing_data_combined_*.csv")
            if not data_files:
                print("❌ No real data files found. Please run 'python data_collector.py' first.")
                return None
            
            # Get the most recent file
            data_file = max(data_files, key=os.path.getctime)
            print(f"Using data file: {data_file}")
        
        try:
            df = pd.read_csv(data_file)
            print(f"Loaded {len(df)} URLs from {data_file}")
            print(f"Phishing URLs: {len(df[df['label'] == 1])}")
            print(f"Legitimate URLs: {len(df[df['label'] == 0])}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def extract_features(self, df):
        """Extract features from URLs"""
        print("Extracting features from URLs...")
        
        features_list = []
        valid_urls = []
        
        for idx, row in df.iterrows():
            try:
                url = row['url']
                features = self.feature_extractor.extract_features(url)
                features_list.append(features)
                valid_urls.append(row)
            except Exception as e:
                print(f"Error extracting features from URL {row['url']}: {e}")
                continue
        
        if not features_list:
            print("❌ No valid features extracted")
            return None, None
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Create labels array
        labels = [row['label'] for row in valid_urls]
        
        print(f"Successfully extracted features for {len(features_df)} URLs")
        return features_df, labels
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and compare performance"""
        print("\n=== Training Multiple Models ===")
        
        # Define models to try
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models = results
        return results
    
    def select_best_model(self, results):
        """Select the best model based on F1 score and cross-validation"""
        print("\n=== Model Selection ===")
        
        # Sort models by F1 score (primary) and CV score (secondary)
        sorted_models = sorted(results.items(), 
                             key=lambda x: (x[1]['f1'], x[1]['cv_mean']), 
                             reverse=True)
        
        best_name, best_results = sorted_models[0]
        
        print(f"Best model: {best_name}")
        print(f"F1 Score: {best_results['f1']:.4f}")
        print(f"CV Score: {best_results['cv_mean']:.4f}")
        
        self.best_model = best_results['model']
        self.best_model_name = best_name
        
        return best_name, best_results
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning on the best model"""
        print(f"\n=== Hyperparameter Tuning for {self.best_model_name} ===")
        
        if self.best_model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=42)
            
        elif self.best_model_name == 'GradientBoosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
            base_model = GradientBoostingClassifier(random_state=42)
            
        elif self.best_model_name == 'LogisticRegression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            base_model = LogisticRegression(random_state=42, max_iter=1000)
            
        elif self.best_model_name == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
            base_model = SVC(random_state=42, probability=True)
        
        else:
            print("Skipping hyperparameter tuning for this model type")
            return self.best_model
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update best model with tuned parameters
        self.best_model = grid_search.best_estimator_
        
        return self.best_model
    
    def evaluate_final_model(self, X_test, y_test):
        """Evaluate the final tuned model"""
        print(f"\n=== Final Model Evaluation ===")
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Final Model ({self.best_model_name}) Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Print detailed classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
        
        # Print confusion matrix
        print(f"\nConfidence Score Distribution:")
        confidence_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for low, high in confidence_ranges:
            count = np.sum((y_pred_proba >= low) & (y_pred_proba < high))
            percentage = count / len(y_pred_proba) * 100
            print(f"  {low:.1f}-{high:.1f}: {count} predictions ({percentage:.1f}%)")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def save_model(self, model_name="production_phishing_model"):
        """Save the trained model and feature extractor"""
        print(f"\n=== Saving Model ===")
        
        # Create model directory if it doesn't exist
        os.makedirs("model", exist_ok=True)
        
        # Save the model
        model_file = f"model/{model_name}.pkl"
        joblib.dump(self.best_model, model_file)
        print(f"Model saved to: {model_file}")
        
        # Save the feature extractor
        extractor_file = f"model/{model_name}_extractor.pkl"
        joblib.dump(self.feature_extractor, extractor_file)
        print(f"Feature extractor saved to: {extractor_file}")
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'training_date': datetime.now().isoformat(),
            'feature_names': [
                'url_length', 'domain_length', 'path_length', 'dot_count',
                'domain_special_chars', 'is_ip', 'domain_digits', 'suspicious_tld',
                'path_special_chars', 'slash_count', 'path_dots', 'suspicious_ext',
                'param_count', 'query_special_chars', 'suspicious_params',
                'at_symbols', 'hyphens', 'double_slashes', 'equal_signs', 'question_marks', 'ampersands',
                'https_used', 'has_port', 'has_redirect',
                'url_entropy', 'domain_entropy',
                'brand_in_domain', 'suspicious_words', 'random_pattern', 'homograph_suspicious'
            ],
            'model_type': type(self.best_model).__name__,
            'model_parameters': self.best_model.get_params() if hasattr(self.best_model, 'get_params') else {}
        }
        
        metadata_file = f"model/{model_name}_metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Model metadata saved to: {metadata_file}")
        
        return model_file
    
    def generate_training_report(self, results, final_results, data_file):
        """Generate a comprehensive training report"""
        print(f"\n=== Generating Training Report ===")
        
        report = f"""
# Phishing URL Detection Model Training Report

## Training Data
- Data file: {data_file}
- Training date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Comparison
"""
        
        for name, result in results.items():
            report += f"""
### {name}
- Accuracy: {result['accuracy']:.4f}
- Precision: {result['precision']:.4f}
- Recall: {result['recall']:.4f}
- F1-Score: {result['f1']:.4f}
- CV Score: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})
"""
        
        report += f"""
## Best Model: {self.best_model_name}

### Final Performance
- Accuracy: {final_results['accuracy']:.4f}
- Precision: {final_results['precision']:.4f}
- Recall: {final_results['recall']:.4f}
- F1-Score: {final_results['f1']:.4f}

### Model Details
- Model Type: {type(self.best_model).__name__}
- Parameters: {self.best_model.get_params() if hasattr(self.best_model, 'get_params') else 'N/A'}

### Recommendations
1. The model achieves {final_results['f1']:.1%} F1-score, which is {'excellent' if final_results['f1'] > 0.9 else 'good' if final_results['f1'] > 0.8 else 'acceptable' if final_results['f1'] > 0.7 else 'needs improvement'} for production use.
2. {'Consider retraining with more data' if final_results['f1'] < 0.8 else 'Model is ready for production deployment'}.
3. Monitor model performance on new data and retrain periodically.
"""
        
        # Save report
        report_file = f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Training report saved to: {report_file}")
        return report_file

def main():
    """Main training function"""
    print("=== Real Phishing URL Detection Model Training ===")
    print("This script will train a production-ready model using real phishing data.\n")
    
    # Initialize trainer
    trainer = RealModelTrainer()
    
    # Load real data
    df = trainer.load_real_data()
    if df is None:
        return
    
    # Extract features
    features_df, labels = trainer.extract_features(df)
    if features_df is None:
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nData split:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train multiple models
    results = trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Select best model
    best_name, best_results = trainer.select_best_model(results)
    
    # Hyperparameter tuning
    trainer.hyperparameter_tuning(X_train, y_train)
    
    # Evaluate final model
    final_results = trainer.evaluate_final_model(X_test, y_test)
    
    # Save model
    model_file = trainer.save_model()
    
    # Generate report
    report_file = trainer.generate_training_report(results, final_results, "real_data")
    
    print(f"\n✅ Training completed successfully!")
    print(f"📁 Model saved to: {model_file}")
    print(f"📄 Training report: {report_file}")
    print(f"\nNext steps:")
    print(f"1. Test the model: python test_real_model.py")
    print(f"2. Deploy the model: python app.py")
    print(f"3. Monitor performance and retrain as needed")

if __name__ == "__main__":
    main() 