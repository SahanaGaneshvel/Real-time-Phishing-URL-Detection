import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Info, TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react';

const FeatureExplanationCard = ({ features, confidence, prediction }) => {
  if (!features || !Array.isArray(features)) {
    return (
      <div className="card border-l-4 border-warning-500 bg-warning-50">
        <div className="p-4">
          <div className="flex items-center">
            <AlertTriangle className="w-5 h-5 text-warning-500 mr-3" />
            <p className="text-warning-700">Feature explanation not available</p>
          </div>
        </div>
      </div>
    );
  }

  const confidencePercentage = Math.round(confidence * 100);
  const isPhishing = prediction === 1 || prediction === true;

  // Prepare data for the chart
  const chartData = features.map((feature, index) => ({
    name: feature.feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    importance: Math.abs(feature.importance),
    contribution: feature.importance,
    index: index
  }));

  // Sort by importance for better visualization
  chartData.sort((a, b) => b.importance - a.importance);

  const getContributionColor = (contribution) => {
    if (contribution > 0) {
      return isPhishing ? 'text-danger-600' : 'text-success-600';
    } else {
      return isPhishing ? 'text-success-600' : 'text-danger-600';
    }
  };

  const getContributionIcon = (contribution) => {
    return contribution > 0 ? (
      <TrendingUp className="w-4 h-4" />
    ) : (
      <TrendingDown className="w-4 h-4" />
    );
  };

  return (
    <div className="card">
      <div className="p-6">
        {/* Header */}
        <div className="flex items-center mb-6">
          <div className="p-3 bg-primary-100 rounded-full mr-4">
            <Info className="w-6 h-6 text-primary-600" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-gray-900">
              Why this URL was flagged
            </h3>
            <p className="text-gray-600">
              Top features influencing the AI decision
            </p>
          </div>
        </div>

        {/* Confidence Score */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">Model Confidence</span>
            <span className={`text-lg font-bold ${
              confidencePercentage >= 80 ? 'text-success-600' :
              confidencePercentage >= 60 ? 'text-warning-600' : 'text-danger-600'
            }`}>
              {confidencePercentage}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full transition-all duration-500 ${
                confidencePercentage >= 80 ? 'bg-success-500' :
                confidencePercentage >= 60 ? 'bg-warning-500' : 'bg-danger-500'
              }`}
              style={{ width: `${confidencePercentage}%` }}
            ></div>
          </div>
        </div>

        {/* Feature Importance Chart */}
        <div className="mb-6">
          <h4 className="text-lg font-semibold mb-4 text-gray-900">
            Feature Importance Analysis
          </h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  dataKey="name" 
                  tick={{ fontSize: 12 }}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip 
                  formatter={(value, name) => [value.toFixed(3), 'Importance']}
                  labelStyle={{ color: '#374151' }}
                  contentStyle={{ 
                    backgroundColor: '#fff', 
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                  }}
                />
                <Bar 
                  dataKey="importance" 
                  fill="#3b82f6"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Feature Details */}
        <div>
          <h4 className="text-lg font-semibold mb-4 text-gray-900">
            Detailed Feature Analysis
          </h4>
          <div className="space-y-3">
            {features.map((feature, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex-1">
                  <div className="flex items-center mb-1">
                    <span className="font-medium text-gray-900 mr-2">
                      {feature.feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </span>
                    <div className={`flex items-center ${getContributionColor(feature.importance)}`}>
                      {getContributionIcon(feature.importance)}
                    </div>
                  </div>
                  <p className="text-sm text-gray-600">
                    {feature.contribution}
                  </p>
                </div>
                <div className="text-right">
                  <div className={`text-lg font-bold ${getContributionColor(feature.importance)}`}>
                    {feature.importance > 0 ? '+' : ''}{feature.importance.toFixed(3)}
                  </div>
                  <div className="text-xs text-gray-500">
                    Impact Score
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Explanation Summary */}
        <div className="mt-6 p-4 bg-blue-50 rounded-lg">
          <h5 className="font-semibold text-blue-900 mb-2">AI Decision Summary</h5>
          <p className="text-blue-800 text-sm">
            The model analyzed {features.length} key features of this URL. 
            {isPhishing ? 
              ' The combination of these factors indicates a high probability of phishing activity.' :
              ' These features suggest this URL appears legitimate and safe to visit.'
            }
            {' '}The confidence level of {confidencePercentage}% reflects how certain the AI is about this assessment.
          </p>
        </div>
      </div>
    </div>
  );
};

export default FeatureExplanationCard;
