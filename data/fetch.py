"""
Advanced data fetching with proper error handling - COMPLETE WORKING VERSION
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

def fetch_stock_data(symbol, period="1y", interval="1d"):
    """
    Fetch historical OHLCV data with proper error handling
    """
    try:
        print(f"   📡 Downloading {symbol} data...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            print(f"   ⚠️ No data returned for {symbol}")
            return None
        
        df = df.reset_index()
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Remove timezone
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'date'})
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error fetching {symbol}: {e}")
        return None

def fetch_fundamental_data(symbol):
    """
    Fetch fundamental data using yfinance
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info:
            return None
        
        fundamental_data = {
            'symbol': symbol,
            'date': datetime.now().date().isoformat(),
            'pe_ratio': info.get('trailingPE', np.nan),
            'forward_pe': info.get('forwardPE', np.nan),
            'eps': info.get('trailingEps', np.nan),
            'market_cap': info.get('marketCap', np.nan),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'revenue_growth': info.get('revenueGrowth', np.nan) * 100 if info.get('revenueGrowth') else 0,
            'earnings_growth': info.get('earningsGrowth', np.nan) * 100 if info.get('earningsGrowth') else 0,
            'profit_margin': info.get('profitMargins', np.nan) * 100 if info.get('profitMargins') else 0,
            'roe': info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') else 0,
            'debt_to_equity': info.get('debtToEquity', np.nan),
            'current_ratio': info.get('currentRatio', np.nan),
            'beta': info.get('beta', np.nan),
            '52_week_high': info.get('fiftyTwoWeekHigh', np.nan),
            '52_week_low': info.get('fiftyTwoWeekLow', np.nan)
        }
        
        return fundamental_data
        
    except Exception as e:
        print(f"Error fetching fundamentals for {symbol}: {e}")
        return None

def fetch_earnings_dates(symbol, years=2):
    """
    Fetch earnings announcement dates with proper error handling
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Try to get earnings dates
        try:
            earnings = ticker.earnings_dates
        except Exception as e:
            print(f"   ⚠️ Earnings dates unavailable: {e}")
            return []
        
        if earnings is None or earnings.empty:
            return []
        
        # Get dates from the last years
        cutoff = datetime.now() - timedelta(days=years * 365)
        
        # Convert earnings index to date without timezone
        earnings_dates = []
        for date in earnings.index:
            try:
                # Convert to naive datetime
                if hasattr(date, 'tzinfo') and date.tzinfo is not None:
                    date_naive = date.tz_localize(None)
                else:
                    date_naive = date
                
                date_obj = date_naive.date()
                
                if date_obj >= cutoff.date():
                    earnings_dates.append(date_obj)
            except Exception:
                continue
        
        return earnings_dates
        
    except Exception as e:
        print(f"   ⚠️ Could not fetch earnings dates: {e}")
        return []

def add_earnings_feature(df, earnings_dates, days_before=3, days_after=3):
    """
    Add earnings period indicator to dataframe
    """
    df = df.copy()
    df['is_earnings_period'] = 0
    df['days_since_earnings'] = 999
    
    if not earnings_dates:
        return df
    
    for date in earnings_dates:
        start_date = date - timedelta(days=days_before)
        end_date = date + timedelta(days=days_after)
        
        mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
        df.loc[mask, 'is_earnings_period'] = 1
        
        # Days since earnings
        days_since = (df['date'].dt.date - date).dt.days
        df.loc[days_since > 0, 'days_since_earnings'] = days_since[days_since > 0]
    
    return df
