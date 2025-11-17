"""
Improved Data Collector - Fixes errors and collects comprehensive data
Enhanced version with better error handling and more data sources
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import time
import random

class ImprovedDataCollector:
    def __init__(self):
        self.mfapi_base = "https://api.mfapi.in/mf"
        self.session = None

        # Initialize database
        self.db_path = Path("data/comprehensive_mf_data.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()

        # Expanded fund universe - more comprehensive list
        self.priority_schemes = [
            # Large Cap Equity Leaders
            "119551", "119760", "119762", "119578", "100016", "119823",
            "119508", "119560", "119562", "119555", "119836", "120503",
            "119834", "119849", "119850", "119851", "119852", "119853",

            # Mid Cap Champions
            "119560", "119551", "119836", "119555", "119775", "119503",
            "119522", "119523", "119524", "119525", "119526", "119527",

            # Small Cap Specialists
            "119528", "119529", "119530", "119531", "119532", "119533",
            "119534", "119535", "119536", "119537", "119538", "119539",

            # ELSS Tax Savers
            "119591", "119596", "119597", "119598", "119599", "119600",
            "119601", "119602", "119603", "119604", "119605", "119606",

            # Index Funds
            "120503", "119834", "119835", "119837", "119838", "119839",
            "119840", "119841", "119842", "119843", "119844", "119845",

            # Debt Fund Leaders
            "119506", "119507", "119589", "119512", "119540", "119541",
            "119542", "119543", "119544", "119545", "119546", "119547",

            # Hybrid Champions
            "119549", "119563", "119564", "119565", "119566", "119567",
            "119568", "119569", "119570", "119571", "119572", "119573",

            # Sectoral & Thematic
            "119775", "119503", "119522", "119574", "119575", "119576",
            "119577", "119579", "119580", "119581", "119582", "119583",

            # International Funds
            "119584", "119585", "119586", "119587", "119588", "119590"
        ]

        # Better market indices with correct symbols
        self.market_indices = {
            '^NSEI': 'NIFTY_50',
            '^BSESN': 'SENSEX',
            '^NSEBANK': 'BANK_NIFTY',
            'NIFTYMCAP100.NS': 'NIFTY_MIDCAP_100',
            'NIFTYSMLCAP100.NS': 'NIFTY_SMALLCAP_100',
            'NIFTYIT.NS': 'NIFTY_IT',
            'NIFTYPHARMA.NS': 'NIFTY_PHARMA',
            'NIFTYAUTO.NS': 'NIFTY_AUTO',
            'NIFTYREALTY.NS': 'NIFTY_REALTY'
        }

        logger.add("logs/improved_data_collection.log", rotation="100 MB")

    def init_database(self):
        """Initialize comprehensive database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enhanced funds table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS funds_comprehensive (
                scheme_code TEXT PRIMARY KEY,
                scheme_name TEXT NOT NULL,
                fund_house TEXT,
                category TEXT,
                subcategory TEXT,
                scheme_type TEXT,
                launch_date DATE,
                aum REAL,
                expense_ratio REAL,
                exit_load REAL,
                risk_level TEXT,
                benchmark TEXT,
                fund_manager TEXT,
                min_investment REAL,
                min_additional REAL,
                is_direct_plan BOOLEAN,
                is_sip_available BOOLEAN,
                is_elss BOOLEAN,
                data_quality_score REAL,
                years_of_data REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Comprehensive NAV history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nav_history_comprehensive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scheme_code TEXT,
                date DATE,
                nav REAL,
                change_amount REAL,
                change_percent REAL,
                volume INTEGER DEFAULT NULL,
                FOREIGN KEY (scheme_code) REFERENCES funds_comprehensive(scheme_code),
                UNIQUE(scheme_code, date)
            )
        """)

        # Enhanced market indices
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_indices_comprehensive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                index_name TEXT,
                date DATE,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                change_amount REAL,
                change_percent REAL,
                UNIQUE(symbol, date)
            )
        """)

        # Advanced performance metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics_comprehensive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scheme_code TEXT,
                calculation_date DATE,
                -- Short term returns
                return_1w REAL, return_2w REAL, return_1m REAL, return_3m REAL, return_6m REAL,
                -- Medium term returns
                return_1y REAL, return_2y REAL, return_3y REAL, return_4y REAL, return_5y REAL,
                -- Long term returns
                return_7y REAL, return_10y REAL, return_15y REAL, return_20y REAL,
                -- Risk metrics across periods
                volatility_1y REAL, volatility_3y REAL, volatility_5y REAL, volatility_10y REAL,
                -- Risk-adjusted returns
                sharpe_1y REAL, sharpe_3y REAL, sharpe_5y REAL, sharpe_10y REAL,
                sortino_1y REAL, sortino_3y REAL, sortino_5y REAL, sortino_10y REAL,
                -- Drawdown analysis
                max_drawdown_1y REAL, max_drawdown_3y REAL, max_drawdown_5y REAL, max_drawdown_10y REAL,
                current_drawdown REAL, avg_drawdown REAL, drawdown_recovery_days REAL,
                -- Rolling performance
                best_1y_return REAL, worst_1y_return REAL, best_3y_return REAL, worst_3y_return REAL,
                -- Consistency metrics
                positive_return_months INTEGER, total_months INTEGER, win_rate REAL,
                -- Market correlation and alpha
                beta_1y REAL, beta_3y REAL, beta_5y REAL,
                alpha_1y REAL, alpha_3y REAL, alpha_5y REAL,
                correlation_nifty REAL, correlation_sensex REAL,
                tracking_error REAL, information_ratio REAL,
                -- Advanced ratios
                calmar_ratio REAL, treynor_ratio REAL, jensen_alpha REAL,
                capture_ratio_up REAL, capture_ratio_down REAL,
                -- Data quality
                data_completeness REAL, outlier_count INTEGER, data_start_date DATE, data_end_date DATE,
                FOREIGN KEY (scheme_code) REFERENCES funds_comprehensive(scheme_code)
            )
        """)

        # Fund categories and benchmarks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fund_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT UNIQUE,
                parent_category TEXT,
                risk_level TEXT,
                typical_return_range_min REAL,
                typical_return_range_max REAL,
                typical_volatility REAL,
                benchmark_index TEXT,
                description TEXT
            )
        """)

        # Market regimes for context
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_regimes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_date DATE,
                end_date DATE,
                regime_type TEXT,
                description TEXT,
                market_return REAL,
                volatility REAL,
                major_events TEXT
            )
        """)

        conn.commit()
        conn.close()

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=180),
            connector=aiohttp.TCPConnector(limit=50, limit_per_host=15)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_all_mutual_fund_schemes(self) -> List[Dict]:
        """Fetch all available mutual fund schemes from MFAPI"""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                async with self.session.get(self.mfapi_base) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Successfully fetched {len(data)} mutual fund schemes")
                        return data
                    elif response.status == 429:
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"HTTP {response.status} error fetching schemes")
                        return []

            except Exception as e:
                logger.error(f"Error fetching all schemes, attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                return []

        return []

    async def fetch_scheme_details(self, scheme_code: str) -> Optional[Dict]:
        """Fetch detailed scheme data with better error handling"""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                url = f"{self.mfapi_base}/{scheme_code}"

                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Enhanced validation
                        if self.validate_scheme_data(data, scheme_code):
                            return data
                        else:
                            logger.warning(f"Invalid data quality for scheme {scheme_code}")
                            return None

                    elif response.status == 404:
                        logger.warning(f"Scheme {scheme_code} not found (404)")
                        return None

                    elif response.status == 429:
                        wait_time = min(2 ** attempt, 8)
                        logger.info(f"Rate limited for {scheme_code}, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    else:
                        logger.error(f"HTTP {response.status} for scheme {scheme_code}")
                        return None

            except asyncio.TimeoutError:
                logger.warning(f"Timeout for scheme {scheme_code}, attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return None

            except Exception as e:
                logger.error(f"Error fetching scheme {scheme_code}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return None

        return None

    def validate_scheme_data(self, data: Dict, scheme_code: str) -> bool:
        """Enhanced data validation"""
        try:
            if not data or 'meta' not in data or 'data' not in data:
                return False

            meta = data['meta']
            nav_data = data['data']

            # Basic meta validation
            if not meta.get('scheme_name') or not meta.get('fund_house'):
                return False

            # NAV data validation
            if not nav_data or len(nav_data) < 100:  # At least 100 days of data
                return False

            # Check data consistency
            valid_entries = 0
            for entry in nav_data[:20]:  # Check first 20 entries
                try:
                    datetime.strptime(entry['date'], '%d-%m-%Y')
                    float(entry['nav'])
                    valid_entries += 1
                except (ValueError, KeyError):
                    continue

            # At least 80% of sample entries should be valid
            if valid_entries / min(20, len(nav_data)) < 0.8:
                return False

            # Check for reasonable NAV values
            recent_navs = []
            for entry in nav_data[:10]:
                try:
                    nav = float(entry['nav'])
                    if 0.1 <= nav <= 10000:  # Reasonable NAV range
                        recent_navs.append(nav)
                except:
                    continue

            if len(recent_navs) < 5:  # Need at least 5 reasonable NAV values
                return False

            return True

        except Exception as e:
            logger.error(f"Validation error for {scheme_code}: {str(e)}")
            return False

    async def collect_enhanced_market_data(self):
        """Collect comprehensive market indices data"""
        logger.info("Collecting enhanced market indices data...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 20)  # 20 years

        conn = sqlite3.connect(self.db_path)
        successful_indices = 0

        for symbol, name in self.market_indices.items():
            try:
                logger.info(f"Fetching {name} ({symbol}) data...")

                # Use yfinance with error handling
                ticker = yf.Ticker(symbol)

                # Fetch data in chunks for better reliability
                all_data = []
                current_start = start_date

                while current_start < end_date:
                    chunk_end = min(current_start + timedelta(days=365 * 2), end_date)

                    try:
                        data = ticker.history(
                            start=current_start.strftime('%Y-%m-%d'),
                            end=chunk_end.strftime('%Y-%m-%d'),
                            auto_adjust=True,
                            prepost=True
                        )

                        if not data.empty:
                            for date, row in data.iterrows():
                                try:
                                    change_amount = row['Close'] - row['Open']
                                    change_percent = (change_amount / row['Open']) * 100 if row['Open'] > 0 else 0

                                    all_data.append((
                                        symbol, name, date.strftime('%Y-%m-%d'),
                                        float(row['Open']), float(row['High']),
                                        float(row['Low']), float(row['Close']),
                                        int(row['Volume']) if not pd.isna(row['Volume']) else 0,
                                        round(change_amount, 2), round(change_percent, 2)
                                    ))
                                except (ValueError, KeyError) as e:
                                    continue

                        current_start = chunk_end
                        await asyncio.sleep(0.2)  # Rate limiting

                    except Exception as e:
                        logger.warning(f"Error fetching chunk for {name}: {str(e)}")
                        current_start = chunk_end
                        continue

                if all_data:
                    cursor = conn.cursor()
                    cursor.executemany("""
                        INSERT OR REPLACE INTO market_indices_comprehensive
                        (symbol, index_name, date, open, high, low, close, volume, change_amount, change_percent)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, all_data)
                    conn.commit()

                    successful_indices += 1
                    logger.info(f"Successfully saved {len(all_data)} records for {name}")
                else:
                    logger.warning(f"No data collected for {name}")

            except Exception as e:
                logger.error(f"Failed to collect data for {name}: {str(e)}")
                continue

        conn.close()
        logger.info(f"Market data collection completed: {successful_indices}/{len(self.market_indices)} indices successful")

    def calculate_comprehensive_metrics(self, scheme_code: str, nav_data: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            if not nav_data or len(nav_data) < 50:
                return {}

            # Convert to DataFrame with better error handling
            df_data = []
            for entry in nav_data:
                try:
                    date = datetime.strptime(entry['date'], '%d-%m-%Y')
                    nav = float(entry['nav'])

                    if nav > 0 and date.year >= 2000:  # Reasonable filters
                        df_data.append({'date': date, 'nav': nav})
                except (ValueError, KeyError):
                    continue

            if len(df_data) < 50:
                return {}

            df = pd.DataFrame(df_data).sort_values('date')

            # Calculate returns
            df['daily_return'] = df['nav'].pct_change()

            # Remove extreme outliers (likely data errors)
            df = df[abs(df['daily_return']) < 0.3]  # Remove >30% daily changes
            df = df.dropna()

            if len(df) < 30:
                return {}

            metrics = {}
            current_nav = df.iloc[-1]['nav']

            # Calculate returns for multiple periods
            periods = {
                '1W': 7, '2W': 14, '1M': 21, '3M': 63, '6M': 126,
                '1Y': 252, '2Y': 504, '3Y': 756, '4Y': 1008, '5Y': 1260,
                '7Y': 1764, '10Y': 2520, '15Y': 3780, '20Y': 5040
            }

            for period_name, days in periods.items():
                if len(df) >= days:
                    past_nav = df.iloc[-days]['nav']
                    if past_nav > 0:
                        if period_name in ['1W', '2W', '1M', '3M', '6M']:
                            # Simple return for short periods
                            period_return = ((current_nav - past_nav) / past_nav) * 100
                        else:
                            # Annualized return for longer periods
                            years = days / 252
                            total_return = (current_nav / past_nav) - 1
                            if total_return > -0.99:
                                period_return = ((1 + total_return) ** (1/years) - 1) * 100
                            else:
                                period_return = -50.0

                        if -100 < period_return < 200:  # Reasonable range
                            metrics[f'return_{period_name.lower()}'] = round(period_return, 2)

            # Risk metrics for different periods
            for years in [1, 3, 5, 10]:
                days = years * 252
                if len(df) >= days:
                    period_returns = df['daily_return'].iloc[-days:]

                    # Volatility
                    vol = period_returns.std() * np.sqrt(252) * 100
                    if np.isfinite(vol) and vol > 0:
                        metrics[f'volatility_{years}y'] = round(vol, 2)

                    # Sharpe Ratio (6% risk-free rate)
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
            for years in [1, 3, 5, 10]:
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
            if len(df) >= 252 * 2:
                rolling_returns = []
                for i in range(252, len(df)):
                    period_return = ((df.iloc[i]['nav'] / df.iloc[i-252]['nav']) - 1) * 100
                    if -100 < period_return < 200:
                        rolling_returns.append(period_return)

                if rolling_returns:
                    metrics['best_1y_return'] = round(max(rolling_returns), 2)
                    metrics['worst_1y_return'] = round(min(rolling_returns), 2)

            # Consistency metrics
            monthly_returns = df.groupby([df['date'].dt.year, df['date'].dt.month])['daily_return'].apply(
                lambda x: (1 + x).prod() - 1
            )

            if len(monthly_returns) > 12:
                positive_months = (monthly_returns > 0).sum()
                total_months = len(monthly_returns)
                metrics['positive_return_months'] = int(positive_months)
                metrics['total_months'] = int(total_months)
                metrics['win_rate'] = round((positive_months / total_months) * 100, 1)

            # Data quality metrics
            metrics['data_start_date'] = df.iloc[0]['date'].strftime('%Y-%m-%d')
            metrics['data_end_date'] = df.iloc[-1]['date'].strftime('%Y-%m-%d')
            metrics['total_data_points'] = len(df)
            metrics['years_of_data'] = round(len(df) / 252, 1)
            metrics['data_completeness'] = round(len(df) / len(nav_data), 2)

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics for {scheme_code}: {str(e)}")
            return {}

    async def save_comprehensive_fund_data(self, scheme_code: str, data: Dict, metrics: Dict):
        """Save comprehensive fund data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            meta = data.get('meta', {})
            scheme_name = meta.get('scheme_name', '')
            fund_house = meta.get('fund_house', '')

            # Enhanced categorization
            category = self.categorize_fund_enhanced(scheme_name)
            subcategory = self.get_subcategory(scheme_name)
            risk_level = self.assess_comprehensive_risk(scheme_name, metrics)

            # Determine fund characteristics
            is_direct_plan = 'direct' in scheme_name.lower()
            is_elss = any(term in scheme_name.lower() for term in ['elss', 'tax', '80c'])

            # Calculate data quality score
            data_quality_score = self.calculate_data_quality_score(metrics)

            # Save fund metadata
            cursor.execute("""
                INSERT OR REPLACE INTO funds_comprehensive
                (scheme_code, scheme_name, fund_house, category, subcategory, scheme_type,
                 risk_level, min_investment, is_direct_plan, is_sip_available, is_elss,
                 data_quality_score, years_of_data, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                scheme_code, scheme_name, fund_house, category, subcategory,
                meta.get('scheme_type', 'Open Ended'), risk_level, 5000.0,
                is_direct_plan, True, is_elss, data_quality_score,
                metrics.get('years_of_data', 0), datetime.now()
            ))

            # Save NAV history with change calculations
            nav_data = data.get('data', [])
            nav_entries = []

            prev_nav = None
            for entry in nav_data:
                try:
                    date_str = entry['date']
                    nav_value = float(entry['nav'])

                    if nav_value > 0:
                        change_amount = (nav_value - prev_nav) if prev_nav else 0
                        change_percent = (change_amount / prev_nav * 100) if prev_nav and prev_nav > 0 else 0

                        nav_entries.append((
                            scheme_code, date_str, nav_value,
                            round(change_amount, 4), round(change_percent, 2)
                        ))

                        prev_nav = nav_value

                except (ValueError, KeyError):
                    continue

            # Bulk insert NAV data
            cursor.executemany("""
                INSERT OR REPLACE INTO nav_history_comprehensive
                (scheme_code, date, nav, change_amount, change_percent)
                VALUES (?, ?, ?, ?, ?)
            """, nav_entries)

            # Save comprehensive performance metrics
            if metrics:
                metric_values = [scheme_code, datetime.now().strftime('%Y-%m-%d')]

                # All the metric fields in the same order as the table
                metric_fields = [
                    'return_1w', 'return_2w', 'return_1m', 'return_3m', 'return_6m',
                    'return_1y', 'return_2y', 'return_3y', 'return_4y', 'return_5y',
                    'return_7y', 'return_10y', 'return_15y', 'return_20y',
                    'volatility_1y', 'volatility_3y', 'volatility_5y', 'volatility_10y',
                    'sharpe_1y', 'sharpe_3y', 'sharpe_5y', 'sharpe_10y',
                    'sortino_1y', 'sortino_3y', 'sortino_5y', 'sortino_10y',
                    'max_drawdown_1y', 'max_drawdown_3y', 'max_drawdown_5y', 'max_drawdown_10y',
                    'current_drawdown', 'best_1y_return', 'worst_1y_return',
                    'positive_return_months', 'total_months', 'win_rate', 'data_completeness'
                ]

                for field in metric_fields:
                    metric_values.append(metrics.get(field))

                # Add data quality fields
                metric_values.extend([
                    0,  # outlier_count (placeholder)
                    metrics.get('data_start_date'),
                    metrics.get('data_end_date')
                ])

                placeholders = ', '.join(['?'] * len(metric_values))
                cursor.execute(f"""
                    INSERT OR REPLACE INTO performance_metrics_comprehensive
                    (scheme_code, calculation_date, return_1w, return_2w, return_1m, return_3m, return_6m,
                     return_1y, return_2y, return_3y, return_4y, return_5y, return_7y, return_10y, return_15y, return_20y,
                     volatility_1y, volatility_3y, volatility_5y, volatility_10y,
                     sharpe_1y, sharpe_3y, sharpe_5y, sharpe_10y,
                     sortino_1y, sortino_3y, sortino_5y, sortino_10y,
                     max_drawdown_1y, max_drawdown_3y, max_drawdown_5y, max_drawdown_10y,
                     current_drawdown, best_1y_return, worst_1y_return,
                     positive_return_months, total_months, win_rate, data_completeness,
                     outlier_count, data_start_date, data_end_date)
                    VALUES ({placeholders})
                """, metric_values)

            conn.commit()

            years_data = metrics.get('years_of_data', 0)
            quality_score = data_quality_score
            logger.info(f"Saved {scheme_name[:50]}... - {years_data:.1f}Y data, Quality: {quality_score:.2f}")

        except Exception as e:
            logger.error(f"Error saving fund {scheme_code}: {str(e)}")
            conn.rollback()

        finally:
            conn.close()

    def categorize_fund_enhanced(self, scheme_name: str) -> str:
        """Enhanced fund categorization"""
        name = scheme_name.lower()

        # Detailed equity categories
        if any(term in name for term in ['large cap', 'blue chip', 'top 100', 'nifty 50', 'sensex']):
            return 'Large Cap Equity'
        elif any(term in name for term in ['mid cap', 'midcap', 'mid-cap']):
            return 'Mid Cap Equity'
        elif any(term in name for term in ['small cap', 'smallcap', 'small-cap']):
            return 'Small Cap Equity'
        elif any(term in name for term in ['multi cap', 'multicap', 'flexi cap', 'flexicap', 'diversified equity']):
            return 'Multi Cap Equity'
        elif any(term in name for term in ['elss', 'tax saver', 'tax saving', '80c']):
            return 'ELSS'
        elif any(term in name for term in ['index', 'nifty', 'sensex', 'etf']):
            return 'Index Fund'
        elif any(term in name for term in ['focused', 'concentrated', 'select']):
            return 'Focused Equity'
        elif any(term in name for term in ['value', 'contra']):
            return 'Value Equity'
        elif any(term in name for term in ['dividend yield', 'dividend']):
            return 'Dividend Yield Equity'

        # Debt categories
        elif any(term in name for term in ['liquid', 'money market', 'cash']):
            return 'Liquid Fund'
        elif any(term in name for term in ['ultra short', 'overnight']):
            return 'Ultra Short Duration'
        elif any(term in name for term in ['short duration', 'short term']):
            return 'Short Duration'
        elif any(term in name for term in ['medium duration', 'medium term']):
            return 'Medium Duration'
        elif any(term in name for term in ['long duration', 'long term', 'gilt']):
            return 'Long Duration'
        elif any(term in name for term in ['corporate bond', 'credit risk']):
            return 'Corporate Bond'
        elif any(term in name for term in ['banking', 'psu']):
            return 'Banking & PSU Bond'
        elif 'dynamic bond' in name:
            return 'Dynamic Bond'

        # Hybrid categories
        elif any(term in name for term in ['balanced', 'hybrid', 'monthly income']):
            if 'aggressive' in name:
                return 'Aggressive Hybrid'
            elif 'conservative' in name:
                return 'Conservative Hybrid'
            else:
                return 'Balanced Hybrid'
        elif 'arbitrage' in name:
            return 'Arbitrage Fund'

        # Thematic/Sectoral
        elif any(term in name for term in ['banking', 'financial', 'pharma', 'technology', 'it', 'infrastructure', 'consumption', 'fmcg']):
            return 'Sectoral/Thematic Equity'
        elif any(term in name for term in ['international', 'global', 'overseas', 'foreign']):
            return 'International Fund'
        elif 'gold' in name:
            return 'Gold Fund'
        elif any(term in name for term in ['commodity', 'natural resources']):
            return 'Commodity Fund'

        # Default categorization
        elif 'equity' in name:
            return 'Equity Fund'
        elif any(term in name for term in ['debt', 'bond', 'income', 'fixed']):
            return 'Debt Fund'
        else:
            return 'Other'

    def get_subcategory(self, scheme_name: str) -> str:
        """Get fund subcategory"""
        name = scheme_name.lower()

        if 'direct' in name:
            return 'Direct Plan'
        elif 'regular' in name:
            return 'Regular Plan'
        elif 'growth' in name:
            return 'Growth Option'
        elif any(term in name for term in ['dividend', 'idcw', 'payout']):
            return 'Dividend Option'
        else:
            return 'Growth Option'  # Default

    def assess_comprehensive_risk(self, scheme_name: str, metrics: Dict) -> str:
        """Comprehensive risk assessment"""
        name = scheme_name.lower()
        volatility_3y = metrics.get('volatility_3y', 15)
        max_drawdown_3y = metrics.get('max_drawdown_3y', -15)

        # Very Low Risk
        if any(cat in name for cat in ['liquid', 'overnight', 'ultra short']):
            return 'Very Low'

        # Low Risk
        elif any(cat in name for cat in ['short duration', 'corporate bond', 'gilt', 'arbitrage']):
            return 'Low'

        # Very High Risk
        elif any(cat in name for cat in ['small cap', 'sectoral', 'thematic', 'commodity']) or \
             (volatility_3y and volatility_3y > 25) or (max_drawdown_3y and max_drawdown_3y < -35):
            return 'Very High'

        # High Risk
        elif any(cat in name for cat in ['mid cap', 'international', 'aggressive']) or \
             (volatility_3y and volatility_3y > 20) or (max_drawdown_3y and max_drawdown_3y < -25):
            return 'High'

        # Moderate Risk
        elif any(cat in name for cat in ['large cap', 'index', 'hybrid', 'multi cap', 'balanced']) or \
             (volatility_3y and 10 < volatility_3y <= 20):
            return 'Moderate'

        # Moderately Low Risk
        elif any(cat in name for cat in ['conservative', 'balanced advantage']) or \
             (volatility_3y and volatility_3y <= 10):
            return 'Moderately Low'

        else:
            return 'Moderate'  # Default

    def calculate_data_quality_score(self, metrics: Dict) -> float:
        """Calculate data quality score (0-1)"""
        score = 0.0

        # Years of data (30% weight)
        years = metrics.get('years_of_data', 0)
        if years >= 15:
            score += 0.3
        elif years >= 10:
            score += 0.25
        elif years >= 5:
            score += 0.2
        elif years >= 3:
            score += 0.15
        else:
            score += 0.1

        # Data completeness (30% weight)
        completeness = metrics.get('data_completeness', 0)
        score += completeness * 0.3

        # Metrics availability (40% weight)
        key_metrics = ['return_1y', 'return_3y', 'volatility_1y', 'sharpe_1y', 'max_drawdown_1y']
        available_metrics = sum(1 for metric in key_metrics if metric in metrics)
        score += (available_metrics / len(key_metrics)) * 0.4

        return round(min(score, 1.0), 2)

    async def collect_comprehensive_fund_data(self, max_funds: int = 1000):
        """Main method to collect comprehensive fund data"""
        logger.info(f"Starting comprehensive fund data collection (up to {max_funds} funds)")

        start_time = datetime.now()

        try:
            # Step 1: Collect market data
            await self.collect_enhanced_market_data()

            # Step 2: Get all available schemes
            all_schemes = await self.fetch_all_mutual_fund_schemes()

            if not all_schemes:
                logger.error("Failed to fetch any schemes")
                return

            # Step 3: Combine priority schemes with additional schemes
            all_scheme_codes = set()

            # Add priority schemes first
            for code in self.priority_schemes:
                if code:
                    all_scheme_codes.add(str(code))

            # Add schemes from API
            for scheme in all_schemes[:max_funds]:
                scheme_code = str(scheme.get('schemeCode', ''))
                if scheme_code and scheme_code not in all_scheme_codes:
                    all_scheme_codes.add(scheme_code)

            scheme_list = list(all_scheme_codes)[:max_funds]
            logger.info(f"Processing {len(scheme_list)} unique schemes")

            # Step 4: Process schemes in batches
            batch_size = 20
            total_saved = 0
            funds_by_years = {'20+': 0, '15-20': 0, '10-15': 0, '5-10': 0, '3-5': 0, '<3': 0}

            for i in range(0, len(scheme_list), batch_size):
                batch = scheme_list[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(scheme_list) - 1) // batch_size + 1

                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} schemes)")

                # Process batch concurrently
                tasks = [self.process_single_scheme_comprehensive(code) for code in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for result in results:
                    if isinstance(result, dict) and result.get('success'):
                        total_saved += 1
                        years = result.get('years_of_data', 0)

                        if years >= 20:
                            funds_by_years['20+'] += 1
                        elif years >= 15:
                            funds_by_years['15-20'] += 1
                        elif years >= 10:
                            funds_by_years['10-15'] += 1
                        elif years >= 5:
                            funds_by_years['5-10'] += 1
                        elif years >= 3:
                            funds_by_years['3-5'] += 1
                        else:
                            funds_by_years['<3'] += 1

                # Progress update
                if batch_num % 10 == 0:
                    logger.info(f"Progress: {total_saved} funds saved")
                    logger.info(f"Distribution: {funds_by_years}")

                # Rate limiting between batches
                await asyncio.sleep(1)

            end_time = datetime.now()
            duration = end_time - start_time

            # Final summary
            logger.info("=" * 70)
            logger.info("COMPREHENSIVE DATA COLLECTION COMPLETED")
            logger.info("=" * 70)
            logger.info(f"Duration: {duration}")
            logger.info(f"Total schemes processed: {len(scheme_list)}")
            logger.info(f"Successfully saved: {total_saved}")
            logger.info(f"Success rate: {(total_saved / len(scheme_list)) * 100:.1f}%")
            logger.info(f"Data distribution by years:")
            for period, count in funds_by_years.items():
                logger.info(f"  {period} years: {count} funds")
            logger.info("=" * 70)

            # Save comprehensive summary
            summary = {
                'collection_completed': end_time.isoformat(),
                'duration_hours': duration.total_seconds() / 3600,
                'total_schemes_processed': len(scheme_list),
                'successfully_saved': total_saved,
                'success_rate': round((total_saved / len(scheme_list)) * 100, 1),
                'funds_by_years': funds_by_years,
                'market_indices_collected': len(self.market_indices),
                'database_path': str(self.db_path),
                'data_quality_average': self.get_average_data_quality()
            }

            summary_path = Path('data/comprehensive_collection_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Comprehensive summary saved to: {summary_path}")

        except Exception as e:
            logger.error(f"Critical error in comprehensive data collection: {str(e)}")
            raise

    async def process_single_scheme_comprehensive(self, scheme_code: str) -> Dict:
        """Process a single scheme with comprehensive analysis"""
        try:
            # Add random delay to avoid overwhelming the API
            await asyncio.sleep(random.uniform(0.1, 0.3))

            # Fetch scheme data
            data = await self.fetch_scheme_details(scheme_code)
            if not data:
                return {'success': False, 'scheme_code': scheme_code, 'error': 'No data fetched'}

            # Calculate comprehensive metrics
            nav_data = data.get('data', [])
            metrics = self.calculate_comprehensive_metrics(scheme_code, nav_data)

            if not metrics or metrics.get('years_of_data', 0) < 1:
                return {'success': False, 'scheme_code': scheme_code, 'error': 'Insufficient data'}

            # Save comprehensive data
            await self.save_comprehensive_fund_data(scheme_code, data, metrics)

            return {
                'success': True,
                'scheme_code': scheme_code,
                'years_of_data': metrics.get('years_of_data', 0),
                'data_quality': metrics.get('data_completeness', 0),
                'fund_name': data.get('meta', {}).get('scheme_name', '')[:50]
            }

        except Exception as e:
            logger.error(f"Error processing comprehensive scheme {scheme_code}: {str(e)}")
            return {'success': False, 'scheme_code': scheme_code, 'error': str(e)}

    def get_average_data_quality(self) -> float:
        """Get average data quality score from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT AVG(data_quality_score) FROM funds_comprehensive WHERE data_quality_score > 0")
            result = cursor.fetchone()

            conn.close()

            return round(result[0] if result and result[0] else 0.0, 2)

        except Exception:
            return 0.0


async def run_improved_collection():
    """Run the improved comprehensive data collection"""
    collector = ImprovedDataCollector()

    async with collector:
        await collector.collect_comprehensive_fund_data(max_funds=1500)


if __name__ == "__main__":
    # Run the improved comprehensive data collection
    asyncio.run(run_improved_collection())