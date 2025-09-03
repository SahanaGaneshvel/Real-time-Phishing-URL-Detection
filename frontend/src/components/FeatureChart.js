import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const FeatureChart = ({ data }) => {
  // Transform data for the chart
  const chartData = data.map(([feature, importance]) => ({
    feature: feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    importance: Math.round(importance * 1000) / 1000 // Round to 3 decimal places
  }));

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} layout="horizontal" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            type="number" 
            domain={[0, 'dataMax']}
            tickFormatter={(value) => value.toFixed(3)}
          />
          <YAxis 
            type="category" 
            dataKey="feature" 
            width={120}
            tick={{ fontSize: 12 }}
          />
          <Tooltip 
            formatter={(value) => [value.toFixed(3), 'Importance']}
            labelFormatter={(label) => `Feature: ${label}`}
          />
          <Bar 
            dataKey="importance" 
            fill="#3b82f6" 
            radius={[0, 4, 4, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default FeatureChart;
