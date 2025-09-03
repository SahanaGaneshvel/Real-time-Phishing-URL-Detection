import React, { useState, useEffect } from 'react';
import { Shield, AlertTriangle, Loader2, BarChart3, Zap, Lock } from 'lucide-react';
import URLInput from './components/URLInput';
import ResultDisplay from './components/ResultDisplay';
import FeatureChart from './components/FeatureChart';
import Header from './components/Header';
import api from './api';

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [featureImportances, setFeatureImportances] = useState([]);
  const [apiHealth, setApiHealth] = useState(null);

  useEffect(() => {
    checkApiHealth();
    loadFeatureImportances();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await api.get('/api/health');
      setApiHealth(response.data);
    } catch (err) {
      setApiHealth({ status: 'unhealthy', model_loaded: false });
    }
  };

  const loadFeatureImportances = async () => {
    try {
      const response = await api.get('/api/feature-importances');
      if (response.data.success) {
        setFeatureImportances(response.data.importances);
      }
    } catch (err) {
      console.error('Failed to load feature importances:', err);
    }
  };

  const handleUrlCheck = async (url) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.post('/api/check-url', { url });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred while checking the URL');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <Header apiHealth={apiHealth} />
      
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          {/* Hero Section */}
          <div className="text-center mb-12">
            <div className="flex justify-center mb-6">
              <div className="p-4 bg-primary-100 rounded-full">
                <Shield className="w-12 h-12 text-primary-600" />
              </div>
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              Phishing URL Detector
            </h1>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Advanced AI-powered URL analysis to protect you from phishing attacks. 
              Get instant results with our machine learning model.
            </p>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-6 mb-12">
            <div className="card text-center">
              <div className="flex justify-center mb-4">
                <Zap className="w-8 h-8 text-primary-600" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Real-time Analysis</h3>
              <p className="text-gray-600">Instant URL scanning with advanced machine learning algorithms</p>
            </div>
            <div className="card text-center">
              <div className="flex justify-center mb-4">
                <BarChart3 className="w-8 h-8 text-primary-600" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Detailed Insights</h3>
              <p className="text-gray-600">Comprehensive analysis with confidence scores and feature breakdown</p>
            </div>
            <div className="card text-center">
              <div className="flex justify-center mb-4">
                <Lock className="w-8 h-8 text-primary-600" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Privacy First</h3>
              <p className="text-gray-600">Your data stays secure with local processing and no data retention</p>
            </div>
          </div>

          {/* Main Content */}
          <div className="grid lg:grid-cols-2 gap-8">
            {/* URL Input and Results */}
            <div className="space-y-6">
              <URLInput onCheck={handleUrlCheck} loading={loading} />
              
              {loading && (
                <div className="card text-center">
                  <Loader2 className="w-8 h-8 text-primary-600 animate-spin mx-auto mb-4" />
                  <p className="text-gray-600">Analyzing URL...</p>
                </div>
              )}

              {error && (
                <div className="card border-l-4 border-danger-500 bg-danger-50">
                  <div className="flex items-center">
                    <AlertTriangle className="w-5 h-5 text-danger-500 mr-3" />
                    <p className="text-danger-700">{error}</p>
                  </div>
                </div>
              )}

              {result && <ResultDisplay result={result} />}
            </div>

            {/* Feature Importance Chart */}
            <div className="space-y-6">
              <div className="card">
                <h3 className="text-lg font-semibold mb-4 flex items-center">
                  <BarChart3 className="w-5 h-5 mr-2" />
                  Model Feature Importance
                </h3>
                {featureImportances.length > 0 ? (
                  <FeatureChart data={featureImportances} />
                ) : (
                  <p className="text-gray-500 text-center py-8">Feature importance data not available</p>
                )}
              </div>

              {/* Model Status */}
              <div className="card">
                <h3 className="text-lg font-semibold mb-4">Model Status</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">API Status:</span>
                    <span className={`px-2 py-1 rounded-full text-sm font-medium ${
                      apiHealth?.status === 'healthy' 
                        ? 'bg-success-100 text-success-800' 
                        : 'bg-danger-100 text-danger-800'
                    }`}>
                      {apiHealth?.status || 'Unknown'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Model Loaded:</span>
                    <span className={`px-2 py-1 rounded-full text-sm font-medium ${
                      apiHealth?.model_loaded 
                        ? 'bg-success-100 text-success-800' 
                        : 'bg-warning-100 text-warning-800'
                    }`}>
                      {apiHealth?.model_loaded ? 'Yes' : 'No'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
