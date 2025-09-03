import React, { useState } from 'react';
import { Search, Globe } from 'lucide-react';

const URLInput = ({ onCheck, loading }) => {
  const [url, setUrl] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (url.trim() && !loading) {
      onCheck(url.trim());
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSubmit(e);
    }
  };

  return (
    <div className="card">
      <h2 className="text-xl font-semibold mb-4 flex items-center">
        <Globe className="w-5 h-5 mr-2" />
        Check URL
      </h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="url" className="block text-sm font-medium text-gray-700 mb-2">
            Enter URL to analyze
          </label>
          <div className="relative">
            <input
              type="text"
              id="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="https://example.com or example.com"
              className="input-field pr-12"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={!url.trim() || loading}
              className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 text-gray-400 hover:text-primary-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Search className="w-5 h-5" />
            </button>
          </div>
        </div>
        
        <button
          type="submit"
          disabled={!url.trim() || loading}
          className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Analyzing...' : 'Analyze URL'}
        </button>
      </form>
      
      <div className="mt-4 p-3 bg-blue-50 rounded-lg">
        <p className="text-sm text-blue-700">
          <strong>Tip:</strong> You can enter URLs with or without the protocol (http:// or https://). 
          Our system will automatically detect and analyze the URL structure.
        </p>
      </div>
    </div>
  );
};

export default URLInput;
