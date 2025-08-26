#!/bin/bash

# Script to debug which package is pulling in torch/CUDA

echo "🔍 Debugging which packages pull in torch/CUDA..."
echo ""

# Create a virtual environment for testing
python3 -m venv test_env
source test_env/bin/activate

echo "📦 Installing packages one by one to identify torch-pulling culprit..."

packages=(
    "streamlit>=1.28.0"
    "langchain>=0.1.0" 
    "langchain-core>=0.1.0"
    "langchain-chroma>=0.1.0"
    "langchain-openai>=0.0.8"
    "langchain-community>=0.0.20"
    "chromadb>=0.4.18"
    "docling>=1.0.0"
    "PyPDF2>=3.0.1"
    "requests>=2.31.0"
    "openai>=1.0.0"
    "python-dotenv>=1.0.0"
    "pydantic>=2.0.0"
)

for package in "${packages[@]}"; do
    echo ""
    echo "⚙️  Installing: $package"
    pip install --no-cache-dir "$package"
    
    # Check if torch was installed
    if pip list | grep -q "torch"; then
        echo "❌ WARNING: $package installed torch!"
        pip list | grep -E "(torch|nvidia|cuda)"
        echo ""
        echo "🔍 Checking dependencies of $package:"
        pip show "$package" | grep "Requires"
        break
    else
        echo "✅ $package is clean (no torch)"
    fi
done

# Clean up
deactivate
rm -rf test_env

echo ""
echo "🎯 Debug complete!"