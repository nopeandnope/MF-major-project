"""Simple data collector for testing"""
import requests
import json
from datetime import datetime
from api.models import SessionLocal, MutualFund, NAVHistory, init_db

def collect_sample_data():
    """Collect sample mutual fund data"""
    db = SessionLocal()
    
    try:
        # Fetch from public API
        response = requests.get("https://api.mfapi.in/mf")
        schemes = response.json()[:100]  # Get first 100 schemes
        
        for scheme in schemes:
            try:
                scheme_code = str(scheme.get('schemeCode', ''))
                scheme_name = scheme.get('schemeName', '')
                
                if not scheme_code:
                    continue
                
                # Check if exists
                existing = db.query(MutualFund).filter_by(scheme_code=scheme_code).first()
                
                if not existing:
                    # Categorize
                    category = "Equity" if "equity" in scheme_name.lower() else "Debt"
                    risk = "High" if "equity" in scheme_name.lower() else "Low"
                    
                    fund = MutualFund(
                        scheme_code=scheme_code,
                        scheme_name=scheme_name[:500],
                        fund_house=scheme_name.split()[0] if scheme_name else "Unknown",
                        category=category,
                        risk_level=risk,
                        minimum_investment=1000.0,
                        esg_score=3.5
                    )
                    db.add(fund)
                    print(f"Added: {scheme_name[:50]}")
            
            except Exception as e:
                print(f"Error processing scheme: {e}")
                continue
        
        db.commit()
        print(f"✅ Data collection complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
    collect_sample_data()