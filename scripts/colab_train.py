"""
Google Colab Training Script
Run this in Colab to train models with GPU
"""
import os
import sys
import subprocess
import time

def install_requirements():
    """Install required packages in Colab"""
    print("📦 Installing requirements...")
    subprocess.run(["pip", "install", "-q", "yfinance", "ta", "tqdm"])
    subprocess.run(["pip", "install", "-q", "tensorflow"])

def download_data(symbols):
    """Download stock data"""
    print(f"📥 Downloading data for {symbols}...")
    for symbol in symbols:
        subprocess.run(["python", "scripts/download_data.py", "--symbols", symbol, "--period", "5y"])

def train_model(symbol, sequence_length=60, epochs=30):
    """Train model with optimized settings"""
    print(f"\n🎓 Training {symbol}...")
    cmd = [
        "python", "scripts/train_improved.py",
        "--symbol", symbol,
        "--sequence-length", str(sequence_length),
        "--epochs", str(epochs),
        "--use-feature-selection",
        "--n-features", "20" if symbol == "AAPL" else "18" if symbol == "MSFT" else "15",
        "--lstm-units", "32",
        "--dropout", "0.2",
        "--learning-rate", "0.001"
    ]
    subprocess.run(cmd)

def main():
    print("=" * 60)
    print("🚀 GOOGLE COLAB TRAINING")
    print("=" * 60)
    
    # Install requirements
    install_requirements()
    
    # Download data
    symbols = ["AAPL", "MSFT", "GOOGL"]
    download_data(symbols)
    
    # Train each model
    for symbol in symbols:
        train_model(symbol)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print("\n📁 Models saved to: models/saved/")
    print("\n🔮 To download models, run:")
    print("   from google.colab import files")
    print("   !zip -r models.zip models/saved/")
    print("   files.download('models.zip')")

if __name__ == "__main__":
    main()
