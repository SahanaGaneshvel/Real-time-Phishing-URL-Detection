import React, { useState, useEffect } from 'react';
import { ShieldCheck, AlertTriangle, Shield, Activity, Clock, CheckCircle, XCircle, AlertCircle } from 'lucide-react';

const ThreatInsightsPanel = ({ url, threatData, isLoading }) => {
  const [animationKey, setAnimationKey] = useState(0);

  useEffect(() => {
    if (threatData) {
      setAnimationKey(prev => prev + 1);
    }
  }, [threatData]);

  const getStatusColor = (status) => {
    switch (status) {
      case 'safe':
        return 'text-success-600 bg-success-100';
      case 'unsafe':
        return 'text-danger-600 bg-danger-100';
      case 'suspicious':
        return 'text-warning-600 bg-warning-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'safe':
        return <CheckCircle className="w-5 h-5" />;
      case 'unsafe':
        return <XCircle className="w-5 h-5" />;
      case 'suspicious':
        return <AlertCircle className="w-5 h-5" />;
      default:
        return <AlertTriangle className="w-5 h-5" />;
    }
  };

  const getConfidenceLevel = (confidence) => {
    if (confidence >= 0.8) return { level: 'High', color: 'text-success-600' };
    if (confidence >= 0.6) return { level: 'Medium', color: 'text-warning-600' };
    return { level: 'Low', color: 'text-danger-600' };
  };

  const getFinalDecisionColor = (decision) => {
    if (decision.includes('safe')) return 'text-success-600 bg-success-100';
    if (decision.includes('phishing')) return 'text-danger-600 bg-danger-100';
    if (decision.includes('suspicious')) return 'text-warning-600 bg-warning-100';
    return 'text-gray-600 bg-gray-100';
  };

  if (isLoading) {
    return (
      <div className="card">
        <div className="p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2" />
            Threat Intelligence Analysis
          </h3>
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-8 bg-gray-200 rounded"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!threatData) {
    return (
      <div className="card border-l-4 border-gray-500 bg-gray-50">
        <div className="p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2" />
            Threat Intelligence Analysis
          </h3>
          <p className="text-gray-600">No threat data available. Run a threat check to see detailed analysis.</p>
        </div>
      </div>
    );
  }

  const confidenceInfo = getConfidenceLevel(threatData.model_confidence);

  return (
    <div className="card" key={animationKey}>
      <div className="p-6">
        <h3 className="text-lg font-semibold mb-6 flex items-center">
          <Activity className="w-5 h-5 mr-2" />
          Threat Intelligence Analysis
        </h3>

        {/* URL Display */}
        <div className="mb-6 p-3 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-600 mb-1">Analyzed URL:</p>
          <p className="text-sm font-mono text-gray-900 break-all">{threatData.url}</p>
        </div>

        <div className="grid gap-4">
          {/* Model Confidence */}
          <div className="p-4 bg-white border border-gray-200 rounded-lg hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center">
                <div className="p-2 bg-primary-100 rounded-full mr-3">
                  <Shield className="w-4 h-4 text-primary-600" />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">AI Model Confidence</h4>
                  <p className="text-sm text-gray-600">Machine learning prediction</p>
                </div>
              </div>
              <div className="text-right">
                <div className={`text-2xl font-bold ${confidenceInfo.color}`}>
                  {Math.round(threatData.model_confidence * 100)}%
                </div>
                <div className={`text-sm font-medium ${confidenceInfo.color}`}>
                  {confidenceInfo.level}
                </div>
              </div>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full transition-all duration-1000 ${
                  threatData.model_confidence >= 0.8 ? 'bg-success-500' :
                  threatData.model_confidence >= 0.6 ? 'bg-warning-500' : 'bg-danger-500'
                }`}
                style={{ width: `${threatData.model_confidence * 100}%` }}
              ></div>
            </div>
          </div>

          {/* Model Result */}
          <div className="p-4 bg-white border border-gray-200 rounded-lg hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <div className="p-2 bg-blue-100 rounded-full mr-3">
                  <ShieldCheck className="w-4 h-4 text-blue-600" />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">Model Result</h4>
                  <p className="text-sm text-gray-600">AI classification</p>
                </div>
              </div>
              <div className={`px-3 py-1 rounded-full text-sm font-semibold flex items-center ${getStatusColor(threatData.model_result)}`}>
                {getStatusIcon(threatData.model_result)}
                <span className="ml-1 capitalize">{threatData.model_result}</span>
              </div>
            </div>
          </div>

          {/* Google Safe Browsing */}
          <div className="p-4 bg-white border border-gray-200 rounded-lg hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <div className="p-2 bg-green-100 rounded-full mr-3">
                  <ShieldCheck className="w-4 h-4 text-green-600" />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">Google Safe Browsing</h4>
                  <p className="text-sm text-gray-600">External threat intelligence</p>
                </div>
              </div>
              <div className={`px-3 py-1 rounded-full text-sm font-semibold flex items-center ${getStatusColor(threatData.google_safe_browsing)}`}>
                {getStatusIcon(threatData.google_safe_browsing)}
                <span className="ml-1 capitalize">{threatData.google_safe_browsing}</span>
              </div>
            </div>
            {threatData.google_threats && threatData.google_threats.length > 0 && (
              <div className="mt-3 pt-3 border-t border-gray-200">
                <p className="text-sm text-gray-600 mb-2">Detected threats:</p>
                <div className="flex flex-wrap gap-2">
                  {threatData.google_threats.map((threat, index) => (
                    <span key={index} className="px-2 py-1 bg-danger-100 text-danger-700 text-xs rounded-full">
                      {threat}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Final Decision */}
          <div className="p-4 bg-gradient-to-r from-primary-50 to-blue-50 border border-primary-200 rounded-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <div className="p-2 bg-primary-100 rounded-full mr-3">
                  <AlertTriangle className="w-4 h-4 text-primary-600" />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">Final Decision</h4>
                  <p className="text-sm text-gray-600">Combined analysis result</p>
                </div>
              </div>
              <div className={`px-4 py-2 rounded-full text-sm font-bold flex items-center ${getFinalDecisionColor(threatData.final_decision)}`}>
                <ShieldCheck className="w-4 h-4 mr-1" />
                <span>{threatData.final_decision}</span>
              </div>
            </div>
          </div>

          {/* Timestamp */}
          <div className="flex items-center justify-center text-sm text-gray-500 pt-2">
            <Clock className="w-4 h-4 mr-1" />
            <span>Analysis completed at {new Date(threatData.timestamp).toLocaleTimeString()}</span>
          </div>
        </div>

        {/* Additional Info */}
        <div className="mt-6 p-4 bg-blue-50 rounded-lg">
          <h5 className="font-semibold text-blue-900 mb-2">Analysis Summary</h5>
          <p className="text-blue-800 text-sm">
            This analysis combines AI-powered URL analysis with external threat intelligence. 
            The final decision considers both the machine learning model's confidence and 
            Google Safe Browsing's threat database to provide comprehensive security assessment.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ThreatInsightsPanel;
