import React from 'react';
import { Shield, Wifi, WifiOff } from 'lucide-react';

const Header = ({ apiHealth }) => {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-primary-100 rounded-lg">
              <Shield className="w-6 h-6 text-primary-600" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Phishing Detector</h1>
              <p className="text-sm text-gray-500">AI-Powered URL Analysis</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              {apiHealth?.status === 'healthy' ? (
                <Wifi className="w-4 h-4 text-success-500" />
              ) : (
                <WifiOff className="w-4 h-4 text-danger-500" />
              )}
              <span className="text-sm text-gray-600">
                {apiHealth?.status === 'healthy' ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
