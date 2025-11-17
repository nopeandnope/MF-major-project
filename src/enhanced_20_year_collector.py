"""
Enhanced 20-Year Historical Data Collector for Indian Mutual Funds
Collects comprehensive data from multiple sources including BSE, NSE, and MFAPI
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import sqlite3
import yfinance as yf
import time

class Enhanced20YearCollector:
    def __init__(self):
        self.mfapi_base = "https://api.mfapi.in/mf"
        self.session = None

        # Initialize database
        self.db_path = Path("data/historical_20y.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()

        # Extended list of top performing funds with 15+ year history
        self.priority_funds = [
            # Large Cap Leaders
            "119551",  # HDFC Top 100 Fund
            "119508",  # ICICI Prudential Focused Bluechip Equity Fund
            "119562",  # SBI Bluechip Fund
            "100016",  # UTI Equity Fund
            "119823",  # Mirae Asset Large Cap Fund

            # Mid Cap Champions
            "119560",  # HDFC Mid-Cap Opportunities Fund
            "119551",  # ICICI Prudential MidCap Fund
            "119836",  # Kotak Emerging Equity Fund
            "119555",  # DSP Midcap Fund

            # Multi Cap Excellence
            "119760",  # Canara Robeco Equity Diversified
            "119762",  # Franklin India Prima Fund
            "119578",  # Nippon India Growth Fund

            # Index Funds
            "120503",  # UTI Nifty Index Fund
            "119834",  # HDFC Index Fund - Sensex Plan

            # ELSS Tax Savers
            "119529",  # Axis Long Term Equity Fund
            "119591",  # HDFC TaxSaver
            "119596",  # ICICI Prudential Long Term Equity Fund

            # Debt Fund Leaders
            "119506",  # HDFC Corporate Bond Fund
            "119507",  # ICICI Prudential Corporate Bond Fund
            "119589",  # SBI Corporate Bond Fund
            "119512",  # HDFC Liquid Fund
            "119540",  # ICICI Prudential Liquid Fund

            # Hybrid Champions
            "119549",  # HDFC Balanced Advantage Fund
            "119563",  # ICICI Prudential Equity & Debt Fund
            "119564",  # SBI Equity Hybrid Fund

            # Sectoral & Thematic
            "119775",  # ICICI Prudential Technology Fund
            "119503",  # Reliance Banking Fund
            "119522",  # SBI Healthcare Opportunities Fund
        ]

        # Market indices for benchmark data
        self.benchmark_indices = {
            '^NSEI': 'NIFTY_50',
            '^BSESN': 'SENSEX',
            '^NSEBANK': 'BANK_NIFTY',
            # Note: Midcap and Smallcap indices may not be available on Yahoo Finance
            # Using alternative index symbols that are available
        }

        logger.add("logs/enhanced_data_collection.log", rotation="100 MB")

    def init_database(self):
        """Initialize comprehensive database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enhanced funds table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS funds (
                scheme_code TEXT PRIMARY KEY,
                scheme_name TEXT NOT NULL,
                fund_house TEXT,
                category TEXT,
                subcategory TEXT,
                launch_date DATE,
                aum REAL,
                expense_ratio REAL,
                exit_load REAL,
                risk_level TEXT,
                benchmark TEXT,
                fund_manager TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 20-year NAV history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nav_history_20y (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scheme_code TEXT,
                date DATE,
                nav REAL,
                units REAL DEFAULT NULL,
                cumulative_units REAL DEFAULT NULL,
                FOREIGN KEY (scheme_code) REFERENCES funds(scheme_code),
                UNIQUE(scheme_code, date)
            )
        """)

        # Market indices data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_indices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                date DATE,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                UNIQUE(symbol, date)
            )
        """)

        # Comprehensive performance metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics_20y (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scheme_code TEXT,
                calculation_date DATE,
                -- Returns across all periods
                return_1m REAL, return_3m REAL, return_6m REAL,
                return_1y REAL, return_2y REAL, return_3y REAL,
                return_5y REAL, return_7y REAL, return_10y REAL,
                return_15y REAL, return_20y REAL,
                -- Risk metrics
                volatility_1y REAL, volatility_3y REAL, volatility_5y REAL,
                beta_1y REAL, beta_3y REAL, beta_5y REAL,
                alpha_1y REAL, alpha_3y REAL, alpha_5y REAL,
                sharpe_1y REAL, sharpe_3y REAL, sharpe_5y REAL,
                sortino_1y REAL, sortino_3y REAL, sortino_5y REAL,
                -- Drawdown metrics
                max_drawdown_1y REAL, max_drawdown_3y REAL, max_drawdown_5y REAL,
                current_drawdown REAL,
                drawdown_duration_days INTEGER,
                -- Rolling performance
                best_1y_return REAL, worst_1y_return REAL,
                best_3y_return REAL, worst_3y_return REAL,
                positive_years_count INTEGER,
                total_years INTEGER,
                -- Correlation with benchmark
                correlation_benchmark REAL,
                tracking_error REAL,
                information_ratio REAL,
                -- Advanced metrics
                calmar_ratio REAL,
                treynor_ratio REAL,
                capture_ratio_up REAL,
                capture_ratio_down REAL,
                FOREIGN KEY (scheme_code) REFERENCES funds(scheme_code)
            )
        """)

        # Market regime analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_regimes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_date DATE,
                end_date DATE,
                regime_type TEXT, -- Bull, Bear, Sideways
                market_return REAL,
                volatility REAL,
                description TEXT
            )
        """)

        conn.commit()
        conn.close()

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120),
            connector=aiohttp.TCPConnector(limit=30, limit_per_host=10)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def collect_enhanced_fund_data(self, scheme_code: str) -> Optional[Dict]:
        """Collect comprehensive fund data with multiple attempts"""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                url = f"{self.mfapi_base}/{scheme_code}"

                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Validate data quality
                        if self.validate_fund_data(data):
                            return data
                        else:
                            logger.warning(f"Invalid data for scheme {scheme_code}")
                            return None

                    elif response.status == 429:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.info(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    else:
                        logger.warning(f"HTTP {response.status} for scheme {scheme_code}")
                        return None

            except asyncio.TimeoutError:
                logger.warning(f"Timeout for scheme {scheme_code}, attempt {attempt + 1}")
                await asyncio.sleep(1)
                continue

            except Exception as e:
                logger.error(f"Error fetching {scheme_code}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return None

        return None

    def validate_fund_data(self, data: Dict) -> bool:
        """Validate fund data quality and completeness"""
        if not data or 'meta' not in data or 'data' not in data:
            return False

        nav_data = data['data']
        if not nav_data or len(nav_data) < 252 * 5:  # At least 5 years
            return False

        # Check for data consistency
        try:
            for entry in nav_data[:10]:  # Check first 10 entries
                datetime.strptime(entry['date'], '%d-%m-%Y')
                float(entry['nav'])
        except (ValueError, KeyError):
            return False

        return True

    async def collect_market_indices_data(self):
        """Collect 20 years of market indices data"""
        logger.info("Collecting market indices data...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 20)  # 20 years

        conn = sqlite3.connect(self.db_path)

        for symbol, name in self.benchmark_indices.items():
            try:
                logger.info(f"Fetching {name} data...")
                ticker = yf.Ticker(symbol)

                # Fetch data in chunks to handle large datasets
                chunk_data = []
                current_start = start_date

                while current_start < end_date:
                    chunk_end = min(current_start + timedelta(days=365 * 2), end_date)

                    data = ticker.history(
                        start=current_start.strftime('%Y-%m-%d'),
                        end=chunk_end.strftime('%Y-%m-%d')
                    )

                    if not data.empty:
                        for date, row in data.iterrows():
                            chunk_data.append((
                                name, date.strftime('%Y-%m-%d'),
                                float(row['Open']), float(row['High']),
                                float(row['Low']), float(row['Close']),
                                int(row['Volume']) if not pd.isna(row['Volume']) else 0
                            ))

                    current_start = chunk_end
                    await asyncio.sleep(0.1)  # Rate limiting

                # Bulk insert
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR REPLACE INTO market_indices
                    (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, chunk_data)

                conn.commit()
                logger.info(f"Saved {len(chunk_data)} records for {name}")

            except Exception as e:
                logger.error(f"Error collecting {name}: {str(e)}")

        conn.close()

    def calculate_comprehensive_metrics(self, scheme_code: str, nav_data: List[Dict]) -> Dict:
        """Calculate comprehensive 20-year performance metrics"""
        if not nav_data or len(nav_data) < 252:  # At least 1 year
            return {}

        try:
            # Convert to DataFrame
            df = pd.DataFrame(nav_data)
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
            df['nav'] = pd.to_numeric(df['nav'], errors='coerce')

            # Clean and sort
            df = df.dropna().sort_values('date')
            df = df[df['nav'] > 0]

            if len(df) < 252:
                return {}

            # Calculate daily returns
            df['daily_return'] = df['nav'].pct_change()
            df = df.dropna()

            # Remove extreme outliers
            df = df[abs(df['daily_return']) < 0.3]  # Remove >30% daily changes

            metrics = {}

            # Calculate returns for all periods
            periods = {
                '1M': 21, '3M': 63, '6M': 126,
                '1Y': 252, '2Y': 504, '3Y': 756,
                '5Y': 1260, '7Y': 1764, '10Y': 2520,
                '15Y': 3780, '20Y': 5040
            }

            current_nav = df.iloc[-1]['nav']

            for period_name, days in periods.items():
                if len(df) >= days:
                    past_nav = df.iloc[-days]['nav']
                    if past_nav > 0:
                        if period_name in ['1M', '3M', '6M']:
                            period_return = ((current_nav - past_nav) / past_nav) * 100
                        else:
                            years = days / 252
                            total_return = (current_nav / past_nav) - 1
                            if total_return > -0.99:
                                period_return = ((1 + total_return) ** (1/years) - 1) * 100
                            else:
                                period_return = -50.0

                        if -100 < period_return < 100:  # Sanity check
                            metrics[f'return_{period_name.lower()}'] = round(period_return, 2)

            # Risk metrics for different periods
            for years in [1, 3, 5]:
                days = years * 252
                if len(df) >= days:
                    period_returns = df['daily_return'].iloc[-days:]

                    # Volatility
                    vol = period_returns.std() * np.sqrt(252) * 100
                    if np.isfinite(vol):
                        metrics[f'volatility_{years}y'] = round(vol, 2)

                    # Sharpe Ratio (assuming 6% risk-free rate)
                    if f'return_{years}y' in metrics and vol > 0:
                        excess_return = (metrics[f'return_{years}y'] / 100) - 0.06
                        sharpe = excess_return / (vol / 100)
                        if np.isfinite(sharpe):
                            metrics[f'sharpe_{years}y'] = round(sharpe, 2)

                    # Sortino Ratio
                    negative_returns = period_returns[period_returns < 0]
                    if len(negative_returns) > 10:
                        downside_vol = negative_returns.std() * np.sqrt(252) * 100
                        if downside_vol > 0 and f'return_{years}y' in metrics:
                            excess_return = (metrics[f'return_{years}y'] / 100) - 0.06
                            sortino = excess_return / (downside_vol / 100)
                            if np.isfinite(sortino):
                                metrics[f'sortino_{years}y'] = round(sortino, 2)

            # Drawdown analysis
            df['cumulative'] = (1 + df['daily_return']).cumprod()
            df['rolling_max'] = df['cumulative'].expanding().max()
            df['drawdown'] = (df['cumulative'] - df['rolling_max']) / df['rolling_max'] * 100

            # Max drawdown for different periods
            for years in [1, 3, 5]:
                days = years * 252
                if len(df) >= days:
                    period_dd = df['drawdown'].iloc[-days:].min()
                    if np.isfinite(period_dd):
                        metrics[f'max_drawdown_{years}y'] = round(period_dd, 2)

            # Current drawdown
            current_dd = df['drawdown'].iloc[-1]
            if np.isfinite(current_dd):
                metrics['current_drawdown'] = round(current_dd, 2)

            # Rolling performance analysis
            if len(df) >= 252 * 3:
                rolling_1y = df['daily_return'].rolling(252).apply(
                    lambda x: ((1 + x).prod() - 1) * 100
                )
                metrics['best_1y_return'] = round(rolling_1y.max(), 2)
                metrics['worst_1y_return'] = round(rolling_1y.min(), 2)

            # Data completeness
            metrics['data_start_date'] = df.iloc[0]['date'].strftime('%Y-%m-%d')
            metrics['data_end_date'] = df.iloc[-1]['date'].strftime('%Y-%m-%d')
            metrics['total_data_points'] = len(df)
            metrics['years_of_data'] = round(len(df) / 252, 1)

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics for {scheme_code}: {str(e)}")
            return {}

    async def save_comprehensive_data(self, scheme_code: str, data: Dict, metrics: Dict):
        """Save comprehensive fund data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            meta = data.get('meta', {})
            scheme_name = meta.get('scheme_name', '')
            fund_house = meta.get('fund_house', '')

            # Categorize and assess risk
            category = self.categorize_fund_enhanced(scheme_name)
            risk_level = self.assess_risk_comprehensive(scheme_name, metrics)

            # Save fund metadata
            cursor.execute("""
                INSERT OR REPLACE INTO funds
                (scheme_code, scheme_name, fund_house, category, risk_level)
                VALUES (?, ?, ?, ?, ?)
            """, (scheme_code, scheme_name, fund_house, category, risk_level))

            # Save NAV history (limit to last 20 years)
            nav_data = data.get('data', [])
            nav_entries = []

            for entry in nav_data:
                try:
                    date_str = entry['date']
                    nav_value = float(entry['nav'])

                    if nav_value > 0:
                        nav_entries.append((scheme_code, date_str, nav_value))

                except (ValueError, KeyError):
                    continue

            # Bulk insert NAV data
            cursor.executemany("""
                INSERT OR REPLACE INTO nav_history_20y (scheme_code, date, nav)
                VALUES (?, ?, ?)
            """, nav_entries)

            # Save comprehensive performance metrics
            if metrics:
                cursor.execute("""
                    INSERT OR REPLACE INTO performance_metrics_20y
                    (scheme_code, calculation_date, return_1m, return_3m, return_6m,
                     return_1y, return_2y, return_3y, return_5y, return_7y, return_10y,
                     return_15y, return_20y, volatility_1y, volatility_3y, volatility_5y,
                     sharpe_1y, sharpe_3y, sharpe_5y, max_drawdown_1y, max_drawdown_3y,
                     max_drawdown_5y, current_drawdown, best_1y_return, worst_1y_return)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    scheme_code, datetime.now().strftime('%Y-%m-%d'),
                    metrics.get('return_1m'), metrics.get('return_3m'), metrics.get('return_6m'),
                    metrics.get('return_1y'), metrics.get('return_2y'), metrics.get('return_3y'),
                    metrics.get('return_5y'), metrics.get('return_7y'), metrics.get('return_10y'),
                    metrics.get('return_15y'), metrics.get('return_20y'),
                    metrics.get('volatility_1y'), metrics.get('volatility_3y'), metrics.get('volatility_5y'),
                    metrics.get('sharpe_1y'), metrics.get('sharpe_3y'), metrics.get('sharpe_5y'),
                    metrics.get('max_drawdown_1y'), metrics.get('max_drawdown_3y'), metrics.get('max_drawdown_5y'),
                    metrics.get('current_drawdown'), metrics.get('best_1y_return'), metrics.get('worst_1y_return')
                ))

            conn.commit()

            years_data = metrics.get('years_of_data', 0)
            logger.info(f"Saved {scheme_name[:50]}... with {len(nav_entries)} NAV entries, {years_data:.1f} years")

        except Exception as e:
            logger.error(f"Error saving data for {scheme_code}: {str(e)}")
            conn.rollback()

        finally:
            conn.close()

    def categorize_fund_enhanced(self, scheme_name: str) -> str:
        """Enhanced fund categorization"""
        name_lower = scheme_name.lower()

        # Detailed equity categories
        if any(term in name_lower for term in ['large cap', 'blue chip', 'top 100', 'nifty 50']):
            return 'Large Cap Equity'
        elif any(term in name_lower for term in ['mid cap', 'midcap']):
            return 'Mid Cap Equity'
        elif any(term in name_lower for term in ['small cap', 'smallcap']):
            return 'Small Cap Equity'
        elif any(term in name_lower for term in ['multi cap', 'flexi cap', 'diversified']):
            return 'Multi Cap Equity'
        elif any(term in name_lower for term in ['elss', 'tax saver', '80c']):
            return 'ELSS'
        elif any(term in name_lower for term in ['index', 'nifty', 'sensex', 'etf']):
            return 'Index Fund'
        elif any(term in name_lower for term in ['focused', 'concentrated']):
            return 'Focused Equity'
        elif any(term in name_lower for term in ['value', 'contra']):
            return 'Value Equity'

        # Debt categories
        elif any(term in name_lower for term in ['liquid', 'money market']):
            return 'Liquid Fund'
        elif any(term in name_lower for term in ['ultra short', 'overnight']):
            return 'Ultra Short Duration'
        elif 'short' in name_lower and 'duration' in name_lower:
            return 'Short Duration'
        elif any(term in name_lower for term in ['medium', 'intermediate']):
            return 'Medium Duration'
        elif any(term in name_lower for term in ['corporate bond', 'credit']):
            return 'Corporate Bond'
        elif 'gilt' in name_lower:
            return 'Gilt Fund'
        elif 'dynamic bond' in name_lower:
            return 'Dynamic Bond'

        # Hybrid
        elif any(term in name_lower for term in ['balanced', 'hybrid', 'monthly income']):
            return 'Hybrid Fund'
        elif 'aggressive' in name_lower:
            return 'Aggressive Hybrid'
        elif 'conservative' in name_lower:
            return 'Conservative Hybrid'
        elif 'arbitrage' in name_lower:
            return 'Arbitrage Fund'

        # Thematic/Sectoral
        elif any(term in name_lower for term in ['banking', 'pharma', 'technology', 'infrastructure', 'consumption']):
            return 'Sectoral/Thematic'
        elif any(term in name_lower for term in ['international', 'global', 'overseas']):
            return 'International Fund'
        elif 'gold' in name_lower:
            return 'Gold Fund'

        # Default categorization
        elif 'equity' in name_lower:
            return 'Equity Fund'
        elif any(term in name_lower for term in ['debt', 'bond', 'income']):
            return 'Debt Fund'
        else:
            return 'Other'

    def assess_risk_comprehensive(self, scheme_name: str, metrics: Dict) -> str:
        """Comprehensive risk assessment based on category and historical metrics"""
        name_lower = scheme_name.lower()
        volatility_3y = metrics.get('volatility_3y', 15)
        max_drawdown_3y = metrics.get('max_drawdown_3y', -15)

        # Very Low Risk
        if any(cat in name_lower for cat in ['liquid', 'overnight', 'ultra short']):
            return 'Very Low'

        # Low Risk
        elif any(cat in name_lower for cat in ['short duration', 'corporate bond', 'gilt', 'arbitrage']):
            return 'Low'

        # Very High Risk
        elif any(cat in name_lower for cat in ['small cap', 'sectoral', 'thematic']) or \
             (volatility_3y and volatility_3y > 25) or (max_drawdown_3y and max_drawdown_3y < -35):
            return 'Very High'

        # High Risk
        elif any(cat in name_lower for cat in ['mid cap', 'international']) or \
             (volatility_3y and volatility_3y > 20) or (max_drawdown_3y and max_drawdown_3y < -25):
            return 'High'

        # Moderate Risk
        elif any(cat in name_lower for cat in ['large cap', 'index', 'hybrid', 'multi cap']) or \
             (volatility_3y and 10 < volatility_3y <= 20):
            return 'Moderate'

        # Moderately Low Risk
        elif any(cat in name_lower for cat in ['conservative', 'balanced advantage']) or \
             (volatility_3y and volatility_3y <= 10):
            return 'Moderately Low'

        else:
            return 'Moderate'  # Default

    async def collect_20_year_data(self, max_funds: int = 1000):
        """Main method to collect comprehensive 20-year data"""
        logger.info(f"Starting comprehensive 20-year data collection for up to {max_funds} funds...")

        start_time = datetime.now()

        try:
            # First, collect market indices data
            await self.collect_market_indices_data()

            # Process priority funds first
            all_schemes_to_process = self.priority_funds.copy()

            # Fetch additional schemes from API
            try:
                all_schemes_response = await self.collect_enhanced_fund_data("")  # Get all schemes
                if all_schemes_response and isinstance(all_schemes_response, list):
                    additional_schemes = [str(s.get('schemeCode', '')) for s in all_schemes_response[:max_funds]]
                    all_schemes_to_process.extend(additional_schemes)
            except:
                logger.warning("Could not fetch additional schemes, using priority list only")

            # Remove duplicates and limit
            unique_schemes = list(dict.fromkeys(all_schemes_to_process))[:max_funds]

            logger.info(f"Processing {len(unique_schemes)} funds...")

            # Process in batches
            batch_size = 15
            total_saved = 0
            funds_20y = 0
            funds_15y = 0
            funds_10y = 0
            funds_5y = 0

            for i in range(0, len(unique_schemes), batch_size):
                batch = unique_schemes[i:i + batch_size]
                batch_number = i // batch_size + 1
                total_batches = (len(unique_schemes) - 1) // batch_size + 1

                logger.info(f"Processing batch {batch_number}/{total_batches} with {len(batch)} funds")

                # Process batch concurrently
                tasks = []
                for scheme_code in batch:
                    if scheme_code:
                        task = self.process_single_fund_comprehensive(scheme_code)
                        tasks.append(task)

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for result in results:
                    if isinstance(result, dict) and result.get('success'):
                        total_saved += 1
                        years = result.get('years_of_data', 0)

                        if years >= 20:
                            funds_20y += 1
                        elif years >= 15:
                            funds_15y += 1
                        elif years >= 10:
                            funds_10y += 1
                        elif years >= 5:
                            funds_5y += 1

                # Progress update
                if batch_number % 5 == 0:
                    logger.info(f"Progress: {total_saved} funds saved")
                    logger.info(f"Data distribution: 20Y+: {funds_20y}, 15-20Y: {funds_15y}, 10-15Y: {funds_10y}, 5-10Y: {funds_5y}")

                # Rate limiting between batches
                await asyncio.sleep(2)

            end_time = datetime.now()
            duration = end_time - start_time

            # Final summary
            logger.info("=" * 60)
            logger.info("20-YEAR DATA COLLECTION COMPLETED")
            logger.info("=" * 60)
            logger.info(f"Duration: {duration}")
            logger.info(f"Total funds processed: {len(unique_schemes)}")
            logger.info(f"Successfully saved: {total_saved}")
            logger.info(f"Funds with 20+ years: {funds_20y}")
            logger.info(f"Funds with 15-20 years: {funds_15y}")
            logger.info(f"Funds with 10-15 years: {funds_10y}")
            logger.info(f"Funds with 5-10 years: {funds_5y}")
            logger.info("=" * 60)

            # Save collection summary
            summary = {
                'collection_completed': end_time.isoformat(),
                'duration_hours': duration.total_seconds() / 3600,
                'total_funds_processed': len(unique_schemes),
                'successfully_saved': total_saved,
                'funds_20y_plus': funds_20y,
                'funds_15_to_20y': funds_15y,
                'funds_10_to_15y': funds_10y,
                'funds_5_to_10y': funds_5y,
                'database_path': str(self.db_path)
            }

            summary_path = Path('data/20_year_collection_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Summary saved to: {summary_path}")

            # Return success if we saved any funds
            return total_saved > 0

        except Exception as e:
            logger.error(f"Critical error in 20-year data collection: {str(e)}")
            return False

    async def process_single_fund_comprehensive(self, scheme_code: str) -> Dict:
        """Process a single fund with comprehensive analysis"""
        try:
            # Fetch fund data
            data = await self.collect_enhanced_fund_data(scheme_code)
            if not data:
                return {'success': False, 'scheme_code': scheme_code, 'error': 'No data fetched'}

            # Calculate comprehensive metrics
            nav_data = data.get('data', [])
            metrics = self.calculate_comprehensive_metrics(scheme_code, nav_data)

            if not metrics:
                return {'success': False, 'scheme_code': scheme_code, 'error': 'No metrics calculated'}

            # Save comprehensive data
            await self.save_comprehensive_data(scheme_code, data, metrics)

            return {
                'success': True,
                'scheme_code': scheme_code,
                'years_of_data': metrics.get('years_of_data', 0),
                'fund_name': data.get('meta', {}).get('scheme_name', '')[:50]
            }

        except Exception as e:
            logger.error(f"Error processing fund {scheme_code}: {str(e)}")
            return {'success': False, 'scheme_code': scheme_code, 'error': str(e)}


async def run_20_year_collection():
    """Run the comprehensive 20-year data collection"""
    collector = Enhanced20YearCollector()

    async with collector:
        await collector.collect_20_year_data(max_funds=800)


if __name__ == "__main__":
    # Run the enhanced 20-year data collection
    asyncio.run(run_20_year_collection())