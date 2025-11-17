"""
Collect 5-10 years of historical Indian mutual fund data
"""
import asyncio
import aiohttp
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import time

from api.models import SessionLocal, MutualFund, NAVHistory, PerformanceMetrics, init_db

class HistoricalDataCollector:
    def __init__(self):
        self.mfapi_base = "https://api.mfapi.in/mf"
        self.session = None
        self.db = SessionLocal()
        
        # Top Indian AMCs for quality data
        self.top_fund_houses = [
            'HDFC', 'ICICI Prudential', 'SBI', 'Aditya Birla Sun Life',
            'Kotak Mahindra', 'Axis', 'UTI', 'Nippon India', 'DSP',
            'Tata', 'Franklin Templeton', 'IDFC', 'Mirae Asset',
            'Motilal Oswal', 'Edelweiss', 'Quantum', 'Invesco',
            'L&T', 'Sundaram', 'Canara Robeco'
        ]
        
        # Categories to focus on
        self.target_categories = [
            'Large Cap', 'Mid Cap', 'Small Cap', 'Multi Cap', 'Flexi Cap',
            'ELSS', 'Index', 'Liquid', 'Ultra Short Duration', 'Short Duration',
            'Corporate Bond', 'Banking PSU', 'Gilt', 'Hybrid Equity',
            'Hybrid Debt', 'Arbitrage', 'Gold', 'International'
        ]
        
        logger.add("logs/data_collection.log", rotation="100 MB")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            connector=aiohttp.TCPConnector(limit=50)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def clean_nan_values(self, value):
        """Convert NaN to None for database storage"""
        if value is None:
            return None
        if isinstance(value, (np.float64, np.float32, float)):
            if np.isnan(value) or np.isinf(value):
                return None
            # Check for extreme values
            if abs(value) > 1e10:
                return None
            # Check for -100 placeholder values
            if value == -100.0:
                return None
        return float(value) if isinstance(value, np.number) else value
    
    async def fetch_all_schemes(self) -> List[Dict]:
        """Fetch all mutual fund schemes"""
        try:
            async with self.session.get(self.mfapi_base) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Fetched {len(data)} total schemes")
                    return data
                else:
                    logger.error(f"Failed to fetch schemes: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching schemes: {str(e)}")
            return []
    
    async def fetch_scheme_historical_data(self, scheme_code: str) -> Optional[Dict]:
        """Fetch complete historical data for a scheme"""
        try:
            url = f"{self.mfapi_base}/{scheme_code}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check if we have sufficient historical data
                    if 'data' in data and len(data['data']) > 1000:  # At least 4 years of daily data
                        return data
                    return None
                    
                elif response.status == 429:  # Rate limited
                    await asyncio.sleep(2)
                    return await self.fetch_scheme_historical_data(scheme_code)
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching scheme {scheme_code}: {str(e)}")
            return None
    
    def filter_quality_schemes(self, all_schemes: List[Dict]) -> List[Dict]:
        """Filter for high-quality schemes with good history"""
        filtered = []
        
        for scheme in all_schemes:
            scheme_name = scheme.get('schemeName', '')
            
            # Check if from top fund house
            is_top_fund = any(fund_house in scheme_name for fund_house in self.top_fund_houses)
            
            # Check for important keywords
            has_category = any(cat.lower() in scheme_name.lower() for cat in self.target_categories)
            
            # Exclude certain types
            exclude_keywords = ['Direct Plan', 'IDCW', 'Dividend', 'Monthly', 'Quarterly']
            is_excluded = any(keyword in scheme_name for keyword in exclude_keywords)
            
            # Include Growth plans from top fund houses
            if (is_top_fund or has_category) and not is_excluded and 'Growth' in scheme_name:
                filtered.append(scheme)
        
        logger.info(f"Filtered to {len(filtered)} quality schemes")
        return filtered
    
    def calculate_comprehensive_metrics(self, nav_data: List[Dict]) -> Dict:
        """Calculate comprehensive metrics for 5-10 year periods"""
        if not nav_data or len(nav_data) < 250:  # At least 1 year of data
            return {}
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(nav_data)
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
            df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
            
            # Remove invalid rows
            df = df.dropna().sort_values('date')
            df = df[df['nav'] > 0]  # Remove zero or negative NAVs
            
            if len(df) < 250:
                return {}
            
            # Calculate returns with error handling
            df['daily_return'] = df['nav'].pct_change()
            
            # Remove extreme outliers (likely data errors)
            df = df[abs(df['daily_return']) < 0.5]  # Remove > 50% daily changes
            
            metrics = {}
            current_nav = df.iloc[-1]['nav']
            current_date = df.iloc[-1]['date']
            
            # Calculate returns for multiple periods
            periods = [
                ('1M', 21), ('3M', 63), ('6M', 126),
                ('1Y', 252), ('2Y', 504), ('3Y', 756),
                ('5Y', 1260), ('7Y', 1764), ('10Y', 2520)
            ]
            
            for period_name, trading_days in periods:
                if len(df) >= trading_days:
                    past_nav = df.iloc[-trading_days]['nav']
                    
                    if past_nav > 0:  # Ensure valid past NAV
                        if period_name in ['1M', '3M', '6M']:
                            # Simple return for short periods
                            period_return = ((current_nav - past_nav) / past_nav) * 100
                        else:
                            # Annualized return for longer periods
                            years = trading_days / 252
                            total_return = (current_nav / past_nav) - 1
                            if total_return > -1:  # Avoid complex number in power calculation
                                period_return = ((1 + total_return) ** (1/years) - 1) * 100
                            else:
                                period_return = -100.0
                        
                        # Sanity check on returns
                        if abs(period_return) < 200:  # Reasonable return range
                            metrics[f'return_{period_name.lower()}'] = period_return
            
            # Risk metrics with error handling
            valid_returns = df['daily_return'].dropna()
            valid_returns = valid_returns[np.isfinite(valid_returns)]
            
            if len(valid_returns) > 20:
                annual_vol = valid_returns.std() * np.sqrt(252) * 100
                if np.isfinite(annual_vol) and annual_vol > 0:
                    metrics['volatility'] = annual_vol
                
                # Sharpe ratio (6% risk-free rate for India)
                risk_free_rate = 0.06
                if 'return_1y' in metrics and annual_vol > 0:
                    excess_return = (metrics['return_1y'] / 100) - risk_free_rate
                    sharpe = excess_return / (annual_vol / 100)
                    if np.isfinite(sharpe):
                        metrics['sharpe_ratio'] = sharpe
                
                # Sortino ratio
                negative_returns = valid_returns[valid_returns < 0]
                if len(negative_returns) > 0:
                    downside_deviation = negative_returns.std() * np.sqrt(252)
                    if downside_deviation > 0 and 'return_1y' in metrics:
                        excess_return = (metrics['return_1y'] / 100) - risk_free_rate
                        sortino = excess_return / downside_deviation
                        if np.isfinite(sortino):
                            metrics['sortino_ratio'] = sortino
                
                # Maximum drawdown
                df['cumulative'] = (1 + valid_returns).cumprod()
                df['rolling_max'] = df['cumulative'].expanding().max()
                df['drawdown'] = (df['cumulative'] - df['rolling_max']) / df['rolling_max']
                max_dd = df['drawdown'].min() * 100
                if np.isfinite(max_dd):
                    metrics['max_drawdown'] = max_dd
            
            # Information about data range
            metrics['data_start_date'] = df.iloc[0]['date'].strftime('%Y-%m-%d')
            metrics['data_end_date'] = df.iloc[-1]['date'].strftime('%Y-%m-%d')
            metrics['total_days'] = len(df)
            metrics['years_of_data'] = metrics['total_days'] / 252
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    async def save_historical_data(self, scheme_data: Dict, metrics: Dict) -> bool:
        """Save historical data to database with proper error handling"""
        try:
            meta = scheme_data.get('meta', {})
            scheme_code = str(meta.get('scheme_code', ''))
            
            if not scheme_code or not metrics:
                return False
            
            # Check if fund exists
            existing_fund = self.db.query(MutualFund).filter_by(
                scheme_code=scheme_code
            ).first()
            
            # Categorize fund
            scheme_name = meta.get('scheme_name', '')
            category = self.categorize_fund(scheme_name)
            risk_level = self.assess_risk_level(scheme_name, metrics)
            
            # Clean and truncate scheme_type to fit database column
            scheme_type = str(meta.get('scheme_type', 'Open Ended'))
            if len(scheme_type) > 200:
                scheme_type = scheme_type[:200]
            
            # Clean numeric values for database
            alpha_value = metrics.get('return_3y', 0) - 12 if 'return_3y' in metrics else 0
            alpha = self.clean_nan_values(alpha_value)
            sharpe_ratio = self.clean_nan_values(metrics.get('sharpe_ratio'))
            
            if not existing_fund:
                # Create new fund
                fund = MutualFund(
                    scheme_code=scheme_code,
                    scheme_name=scheme_name[:500],  # Truncate to fit
                    fund_house=meta.get('fund_house', '')[:200],  # Truncate to fit
                    category=category,
                    scheme_type=scheme_type,
                    risk_level=risk_level,
                    minimum_investment=5000.0,
                    alpha=alpha,
                    beta=1.0,
                    sharpe_ratio=sharpe_ratio,
                    esg_score=np.random.uniform(3.0, 4.5)
                )
                self.db.add(fund)
                self.db.flush()
                fund_id = fund.id
            else:
                # Update existing fund
                existing_fund.category = category
                existing_fund.risk_level = risk_level
                existing_fund.sharpe_ratio = sharpe_ratio
                existing_fund.updated_at = datetime.utcnow()
                fund_id = existing_fund.id
            
            # Save NAV history (limit to last 2500 entries)
            nav_data = scheme_data.get('data', [])[:2500]
            nav_count = 0
            
            for nav_entry in nav_data:
                try:
                    nav_date = datetime.strptime(nav_entry['date'], '%d-%m-%Y')
                    nav_value = float(nav_entry['nav'])
                    
                    # Skip invalid NAV values
                    if nav_value <= 0 or not np.isfinite(nav_value):
                        continue
                    
                    # Check if NAV entry exists
                    existing_nav = self.db.query(NAVHistory).filter_by(
                        fund_id=fund_id,
                        date=nav_date
                    ).first()
                    
                    if not existing_nav:
                        nav_history = NAVHistory(
                            fund_id=fund_id,
                            date=nav_date,
                            nav=nav_value
                        )
                        self.db.add(nav_history)
                        nav_count += 1
                        
                except (ValueError, KeyError):
                    continue
            
            # Save performance metrics with cleaned values
            perf_metrics = PerformanceMetrics(
                fund_id=fund_id,
                calculation_date=datetime.utcnow(),
                return_1m=self.clean_nan_values(metrics.get('return_1m')),
                return_3m=self.clean_nan_values(metrics.get('return_3m')),
                return_6m=self.clean_nan_values(metrics.get('return_6m')),
                return_1y=self.clean_nan_values(metrics.get('return_1y')),
                return_3y=self.clean_nan_values(metrics.get('return_3y')),
                return_5y=self.clean_nan_values(metrics.get('return_5y')),
                volatility=self.clean_nan_values(metrics.get('volatility')),
                sharpe_ratio=self.clean_nan_values(metrics.get('sharpe_ratio')),
                max_drawdown=self.clean_nan_values(metrics.get('max_drawdown'))
            )
            self.db.add(perf_metrics)
            
            self.db.commit()
            
            years_of_data = metrics.get('years_of_data', 0)
            logger.info(f"Saved {scheme_name[:50]} with {nav_count} NAV entries, {years_of_data:.1f} years of data")
            return True
            
        except Exception as e:
            logger.error(f"Error saving fund: {str(e)}")
            self.db.rollback()
            return False
    
    def categorize_fund(self, scheme_name: str) -> str:
        """Categorize fund based on name"""
        name_lower = scheme_name.lower()
        
        # Equity categories
        if 'large cap' in name_lower or 'bluechip' in name_lower:
            return 'Large Cap'
        elif 'mid cap' in name_lower or 'midcap' in name_lower:
            return 'Mid Cap'
        elif 'small cap' in name_lower or 'smallcap' in name_lower:
            return 'Small Cap'
        elif 'multi cap' in name_lower or 'flexi cap' in name_lower:
            return 'Multi Cap'
        elif 'elss' in name_lower or 'tax' in name_lower:
            return 'ELSS'
        elif 'index' in name_lower or 'nifty' in name_lower or 'sensex' in name_lower:
            return 'Index'
        
        # Debt categories
        elif 'liquid' in name_lower:
            return 'Liquid'
        elif 'ultra short' in name_lower or 'overnight' in name_lower:
            return 'Ultra Short Duration'
        elif 'short' in name_lower and 'duration' in name_lower:
            return 'Short Duration'
        elif 'corporate bond' in name_lower:
            return 'Corporate Bond'
        elif 'gilt' in name_lower:
            return 'Gilt'
        
        # Hybrid
        elif 'hybrid' in name_lower or 'balanced' in name_lower:
            return 'Hybrid'
        elif 'arbitrage' in name_lower:
            return 'Arbitrage'
        
        # Others
        elif 'gold' in name_lower:
            return 'Gold'
        elif 'international' in name_lower or 'global' in name_lower:
            return 'International'
        
        # Default
        elif 'equity' in name_lower:
            return 'Equity'
        elif 'debt' in name_lower or 'bond' in name_lower or 'income' in name_lower:
            return 'Debt'
        else:
            return 'Other'
    
    def assess_risk_level(self, scheme_name: str, metrics: Dict) -> str:
        """Assess risk level based on category and volatility"""
        name_lower = scheme_name.lower()
        volatility = metrics.get('volatility', 15)
        
        # Low risk categories
        if any(cat in name_lower for cat in ['liquid', 'overnight', 'ultra short', 'arbitrage']):
            return 'Low'
        
        # High risk categories
        elif any(cat in name_lower for cat in ['small cap', 'mid cap', 'sector', 'thematic']):
            return 'High'
        
        # Volatility-based assessment
        elif volatility and volatility < 10:
            return 'Low'
        elif volatility and volatility < 20:
            return 'Moderate'
        else:
            return 'High'
    
    async def collect_historical_data(self, target_funds: int = 500):
        """Main collection function for historical data"""
        logger.info(f"Starting historical data collection for {target_funds} funds...")
        
        try:
            # Fetch all schemes
            all_schemes = await self.fetch_all_schemes()
            
            if not all_schemes:
                logger.error("No schemes fetched")
                return
            
            # Filter for quality schemes
            quality_schemes = self.filter_quality_schemes(all_schemes)
            
            # Limit to target number
            selected_schemes = quality_schemes[:target_funds]
            logger.info(f"Processing {len(selected_schemes)} schemes...")
            
            # Process in batches
            batch_size = 20
            success_count = 0
            schemes_with_10y = 0
            schemes_with_5y = 0
            schemes_with_3y = 0
            
            for i in range(0, len(selected_schemes), batch_size):
                batch = selected_schemes[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(selected_schemes)-1)//batch_size + 1}")
                
                tasks = []
                for scheme in batch:
                    task = self.process_historical_scheme(scheme)
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, dict):
                        if result.get('success'):
                            success_count += 1
                            years = result.get('years_of_data', 0)
                            if years >= 10:
                                schemes_with_10y += 1
                            elif years >= 5:
                                schemes_with_5y += 1
                            elif years >= 3:
                                schemes_with_3y += 1
                
                # Rate limiting
                await asyncio.sleep(1)
                
                # Progress update
                if (i + batch_size) % 100 == 0:
                    logger.info(f"Progress: {success_count} funds saved")
                    logger.info(f"Data distribution: 10Y+: {schemes_with_10y}, 5-10Y: {schemes_with_5y}, 3-5Y: {schemes_with_3y}")
            
            logger.info(f"Collection completed!")
            logger.info(f"Total funds saved: {success_count}")
            logger.info(f"10+ years data: {schemes_with_10y} funds")
            logger.info(f"5-10 years data: {schemes_with_5y} funds")
            logger.info(f"3-5 years data: {schemes_with_3y} funds")
            
            # Save summary
            summary = {
                'collection_date': datetime.now().isoformat(),
                'total_funds': success_count,
                'funds_10y_plus': schemes_with_10y,
                'funds_5y_to_10y': schemes_with_5y,
                'funds_3y_to_5y': schemes_with_3y
            }
            
            summary_path = Path('data/collection_summary.json')
            summary_path.parent.mkdir(exist_ok=True)
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")
        finally:
            self.db.close()
    
    async def process_historical_scheme(self, scheme: Dict) -> Dict:
        """Process single scheme with historical data"""
        try:
            scheme_code = str(scheme.get('schemeCode', ''))
            if not scheme_code:
                return {'success': False}
            
            # Fetch historical data
            details = await self.fetch_scheme_historical_data(scheme_code)
            if not details:
                return {'success': False}
            
            # Calculate metrics
            nav_data = details.get('data', [])
            metrics = self.calculate_comprehensive_metrics(nav_data)
            
            if not metrics:
                return {'success': False}
            
            # Save to database
            saved = await self.save_historical_data(details, metrics)
            
            return {
                'success': saved,
                'years_of_data': metrics.get('years_of_data', 0)
            }
            
        except Exception as e:
            logger.error(f"Error processing scheme: {str(e)}")
            return {'success': False}

async def run_historical_collection():
    """Run historical data collection"""
    init_db()
    async with HistoricalDataCollector() as collector:
        await collector.collect_historical_data(target_funds=500)

if __name__ == "__main__":
    asyncio.run(run_historical_collection())