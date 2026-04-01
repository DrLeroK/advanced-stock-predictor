#!/usr/bin/env python
"""
Download historical stock data
"""
import sys
import os
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch import fetch_stock_data
from data.store import AdvancedDataStore
from config.settings import DEFAULT_SYMBOLS

def main():
    parser = argparse.ArgumentParser(description="Download stock data")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                       help="Stock symbols to download")
    parser.add_argument("--period", default="5y", 
                       help="Data period: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max")
    parser.add_argument("--interval", default="1d",
                       help="Data interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("📥 STOCK DATA DOWNLOADER")
    print("=" * 80)
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Period: {args.period}")
    print(f"Interval: {args.interval}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    store = AdvancedDataStore()
    
    downloaded = []
    failed = []
    
    for symbol in args.symbols:
        print(f"\n📊 Downloading {symbol}...")
        
        try:
            df = fetch_stock_data(symbol, period=args.period, interval=args.interval)
            
            if df is not None and not df.empty:
                store.save_raw(df, symbol)
                print(f"   ✅ Saved {len(df)} rows")
                print(f"   📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
                print(f"   💰 Latest close: ${df['close'].iloc[-1]:.2f}")
                downloaded.append(symbol)
            else:
                print(f"   ❌ No data returned")
                failed.append(symbol)
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed.append(symbol)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully downloaded: {len(downloaded)} stocks")
    if downloaded:
        print(f"   {', '.join(downloaded)}")
    print(f"❌ Failed: {len(failed)} stocks")
    if failed:
        print(f"   {', '.join(failed)}")
    print("=" * 80)
    print("\n💡 Next steps:")
    print("1. Download fundamentals: python scripts/download_fundamentals.py")
    print("2. Train ensemble: python scripts/train_ensemble.py --symbol AAPL")

if __name__ == "__main__":
    main()
