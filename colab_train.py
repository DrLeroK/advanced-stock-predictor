#!/usr/bin/env python
"""
Google Colab Training Script
Copy and paste this entire file into a Colab cell
"""

print("=" * 60)
print("🚀 ADVANCED STOCK PREDICTOR - COLAB TRAINING")
print("=" * 60)

# Step 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Clone repository
!git clone https://github.com/DrLeroK/advanced-stock-predictor.git /content/stock-predictor
%cd /content/stock-predictor

# Step 3: Install dependencies
!pip install -q yfinance ta tqdm tensorflow

print("\n✅ Setup complete!")

# Step 4: Download data
print("\n" + "=" * 60)
print("📥 DOWNLOADING STOCK DATA")
print("=" * 60)

!python scripts/download_data.py --symbols AAPL MSFT GOOGL --period 5y

# Step 5: Train AAPL
print("\n" + "=" * 60)
print("🎓 TRAINING AAPL (20-30 minutes)")
print("=" * 60)
!python scripts/train_improved.py --symbol AAPL --sequence-length 60 --epochs 30 --use-feature-selection --n-features 20 --lstm-units 32 --dropout 0.2

# Step 6: Train MSFT
print("\n" + "=" * 60)
print("🎓 TRAINING MSFT (20-30 minutes)")
print("=" * 60)
!python scripts/train_improved.py --symbol MSFT --sequence-length 60 --epochs 30 --use-feature-selection --n-features 18 --lstm-units 32 --dropout 0.2

# Step 7: Train GOOGL
print("\n" + "=" * 60)
print("🎓 TRAINING GOOGL (20-30 minutes)")
print("=" * 60)
!python scripts/train_improved.py --symbol GOOGL --sequence-length 60 --epochs 30 --use-feature-selection --n-features 15 --lstm-units 32 --dropout 0.2

# Step 8: Save and download models
print("\n" + "=" * 60)
print("💾 SAVING MODELS")
print("=" * 60)

import os
import shutil
from google.colab import files

# Save to Google Drive
if os.path.exists('models/saved'):
    backup_dir = "/content/drive/MyDrive/stock_predictor_models"
    shutil.copytree('models/saved', backup_dir, dirs_exist_ok=True)
    print(f"✅ Models saved to: {backup_dir}")

# Create zip and download
!zip -r models.zip models/saved/
files.download('models.zip')
print("✅ Download started - check your browser")

# Step 9: Test
print("\n" + "=" * 60)
print("🔮 TEST PREDICTION")
print("=" * 60)
!python scripts/predict.py --symbol AAPL

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
