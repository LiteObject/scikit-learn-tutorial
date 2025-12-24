#!/bin/bash

echo "🎓 SCIKIT-LEARN NETWORK SECURITY TUTORIAL SETUP"
echo "================================================="

echo ""
echo "📁 Creating virtual environment..."
python3 -m venv tutorial_env

echo ""
echo "🔧 Activating virtual environment..."
source tutorial_env/bin/activate

echo ""
echo "📦 Installing packages..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""
echo "🧪 Running setup verification..."
python setup.py

echo ""
echo "🎉 Setup complete! Virtual environment activated."
echo ""
echo "📚 To run the tutorial:"
echo "   python email_spam_ml.py"
echo ""
echo "💡 To deactivate the environment later, type: deactivate"
echo ""
