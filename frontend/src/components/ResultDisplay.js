import React from 'react';
import { CheckCircle, AlertTriangle, Shield, BarChart3, Info } from 'lucide-react';

const ResultDisplay = ({ result }) => {
  if (!result) return null;

  const { url, is_phishing, confidence, prediction_text, features } = result;
  
  const confidencePercentage = Math.round(confidence * 100);
  const isSafe = !is_phishing;

  return (
    <div className="space-y-6">
      {/* Main Result Card */}
      <div className={`card border-l-4 ${
        isSafe ? 'border-success-500 bg-success-50' : 'border-danger-500 bg-danger-50'
      }`}>
        <div className="flex items-start space-x-4">
          <div className={`p-3 rounded-full ${
            isSafe ? 'bg-success-100' : 'bg-danger-100'
          }`}>
            {isSafe ? (
              <CheckCircle className="w-8 h-8 text-success-600" />
            ) : (
              <AlertTriangle className="w-8 h-8 text-danger-600" />
            )}
          </div>
          
          <div className="flex-1">
            <h3 className="text-xl font-semibold mb-2">
              {isSafe ? 'Safe URL' : 'Phishing URL Detected'}
            </h3>
            <p className="text-gray-600 mb-3">
              {url}
            </p>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Shield className="w-4 h-4 text-gray-500" />
                <span className="text-sm text-gray-600">Confidence:</span>
                <span className={`font-semibold ${
                  confidencePercentage >= 80 ? 'text-success-600' :
                  confidencePercentage >= 60 ? 'text-warning-600' : 'text-danger-600'
                }`}>
                  {confidencePercentage}%
                </span>
              </div>
              
              <div className="flex items-center space-x-2">
                <BarChart3 className="w-4 h-4 text-gray-500" />
                <span className="text-sm text-gray-600">Prediction:</span>
                <span className={`font-semibold ${
                  isSafe ? 'text-success-600' : 'text-danger-600'
                }`}>
                  {prediction_text}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Confidence Bar */}
      <div className="card">
        <h4 className="text-lg font-semibold mb-4 flex items-center">
          <BarChart3 className="w-5 h-5 mr-2" />
          Confidence Score
        </h4>
        
        <div className="space-y-3">
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">Confidence Level</span>
            <span className="font-semibold">{confidencePercentage}%</span>
          </div>
          
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div 
              className={`h-3 rounded-full transition-all duration-500 ${
                confidencePercentage >= 80 ? 'bg-success-500' :
                confidencePercentage >= 60 ? 'bg-warning-500' : 'bg-danger-500'
              }`}
              style={{ width: `${confidencePercentage}%` }}
            ></div>
          </div>
          
          <div className="flex justify-between text-xs text-gray-500">
            <span>Low</span>
            <span>Medium</span>
            <span>High</span>
          </div>
        </div>
      </div>

      {/* Key Features */}
      <div className="card">
        <h4 className="text-lg font-semibold mb-4 flex items-center">
          <Info className="w-5 h-5 mr-2" />
          Key Analysis Features
        </h4>
        
        <div className="grid md:grid-cols-2 gap-4">
          <FeatureItem 
            label="URL Length" 
            value={features.url_length} 
            unit="characters"
          />
          <FeatureItem 
            label="Domain Length" 
            value={features.domain_length} 
            unit="characters"
          />
          <FeatureItem 
            label="Path Length" 
            value={features.path_length} 
            unit="characters"
          />
          <FeatureItem 
            label="Dot Count" 
            value={features.dot_count} 
            unit="dots"
          />
          <FeatureItem 
            label="Special Characters" 
            value={features.domain_special_chars} 
            unit="chars"
          />
          <FeatureItem 
            label="Suspicious TLD" 
            value={features.suspicious_tld ? 'Yes' : 'No'} 
            unit=""
          />
          <FeatureItem 
            label="HTTPS Used" 
            value={features.https_used ? 'Yes' : 'No'} 
            unit=""
          />
          <FeatureItem 
            label="Has Redirect" 
            value={features.has_redirect ? 'Yes' : 'No'} 
            unit=""
          />
        </div>
      </div>
    </div>
  );
};

const FeatureItem = ({ label, value, unit }) => (
  <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
    <span className="text-sm font-medium text-gray-700">{label}</span>
    <span className="text-sm font-semibold text-gray-900">
      {value}{unit && ` ${unit}`}
    </span>
  </div>
);

export default ResultDisplay;
