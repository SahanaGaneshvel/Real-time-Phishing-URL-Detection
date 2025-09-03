#!/bin/bash

echo "🚀 Starting Phishing URL Detector..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo "🔧 Starting Flask API backend..."
python api.py &
API_PID=$!

echo "⏳ Waiting for API to start..."
sleep 3

echo "🎨 Starting React frontend..."
cd frontend
npm start &
FRONTEND_PID=$!

echo "✅ Application started successfully!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔌 API: http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user to stop
wait

# Cleanup on exit
echo "🛑 Stopping servers..."
kill $API_PID 2>/dev/null
kill $FRONTEND_PID 2>/dev/null
echo "✅ Servers stopped"
