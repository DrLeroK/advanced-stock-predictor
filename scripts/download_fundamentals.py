#!/usr/bin/env python
"""
Download fundamental data for stocks
"""
import sys
import os
import argparse
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch import fetch_fundamental_data, fetch_earnings_dates
from data.store import AdvancedDataStore
from config.settings import DEFAULT_SYMBOLS

def main():
    parser = argparse.ArgumentParser(description="Download fundamental data")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS[:3],
                       help="Stock symbols")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("📊 FUNDAMENTAL DATA DOWNLOADER")
    print("=" * 80)
    print(f"Symbols: {', '.join(args.symbols)}")
    print("=" * 80)
    
    store = AdvancedDataStore()
    
    for symbol in args.symbols:
        print(f"\n📊 Fetching fundamentals for {symbol}...")
        
        # Fetch fundamentals
        fundamentals = fetch_fundamental_data(symbol)
        if fundamentals:
            store.save_fundamentals(fundamentals, symbol)
            print(f"   ✅ Saved fundamentals")
            
            # Fetch earnings dates
            earnings_dates = fetch_earnings_dates(symbol)
            if earnings_dates:
                earnings_file = os.path.join(store.fundamentals_dir, f"{symbol}_earnings.json")
                with open(earnings_file, 'w') as f:
                    json.dump([d.isoformat() for d in earnings_dates], f)
                print(f"   ✅ Saved {len(earnings_dates)} earnings dates")
            else:
                print(f"   ⚠️ No earnings dates found")
        else:
            print(f"   ❌ Failed to fetch fundamentals")
    
    print("\n" + "=" * 80)
    print("✅ Download complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
