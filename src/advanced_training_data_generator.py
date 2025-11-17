"""
Advanced Training Data Generator for Financial LLM

Generates high-quality training data with rich context including:
- Recent performance metrics
- Risk metrics (volatility, Sharpe, drawdown)
- Market conditions and sentiment
- Economic indicators
- Historical patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import sqlite3
from pathlib import Path


class AdvancedTrainingDataGenerator:
    """
    Generate high-quality training data for financial LLM with rich contextual information
    """

    def __init__(self, db_path: str):
        """
        Initialize training data generator.

        Args:
            db_path: Path to SQLite database with historical NAV data
        """
        self.db_path = db_path
        self.conn = None
        self.nifty_data = None  # To be loaded if available
        self.vix_data = None  # To be loaded if available

    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)

    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def calculate_returns(
        self,
        nav_series: pd.Series,
        periods: List[int] = [1, 7, 30, 90, 365]
    ) -> Dict[str, float]:
        """
        Calculate returns for multiple periods.

        Args:
            nav_series: Series of NAV values (sorted by date)
            periods: List of periods in days

        Returns:
            Dictionary of returns for each period
        """
        returns = {}
        for period in periods:
            if len(nav_series) > period:
                returns[f'{period}day'] = round(
                    (nav_series.iloc[-1] / nav_series.iloc[-period] - 1) * 100,
                    2
                )
            else:
                returns[f'{period}day'] = None
        return returns

    def calculate_risk_metrics(self, nav_series: pd.Series) -> Dict[str, float]:
        """
        Calculate volatility, Sharpe ratio, and max drawdown.

        Args:
            nav_series: Series of NAV values

        Returns:
            Dictionary of risk metrics
        """
        # Calculate daily returns
        returns = nav_series.pct_change().dropna()

        if len(returns) < 30:
            return {
                'volatility': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0
            }

        # Annualized volatility
        volatility = returns.std() * np.sqrt(252) * 100

        # Sharpe ratio (assuming 6.5% risk-free rate for India)
        risk_free_rate = 0.065
        excess_returns = returns.mean() * 252 - risk_free_rate
        sharpe = excess_returns / (volatility / 100) if volatility > 0 else 0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        return {
            'volatility': round(volatility, 2),
            'sharpe': round(sharpe, 2),
            'max_drawdown': round(max_drawdown, 2)
        }

    def determine_market_phase(self, recent_return: float) -> str:
        """
        Classify market phase based on recent returns.

        Args:
            recent_return: 30-day return percentage

        Returns:
            Market phase classification
        """
        if recent_return > 5:
            return 'Bull Market'
        elif recent_return < -5:
            return 'Bear Market'
        else:
            return 'Sideways Market'

    def classify_fund_category(self, scheme_name: str) -> str:
        """
        Classify fund category from scheme name.

        Args:
            scheme_name: Name of the mutual fund scheme

        Returns:
            Fund category
        """
        name_lower = scheme_name.lower()

        if 'large cap' in name_lower or 'bluechip' in name_lower:
            return 'Large Cap Equity'
        elif 'mid cap' in name_lower:
            return 'Mid Cap Equity'
        elif 'small cap' in name_lower:
            return 'Small Cap Equity'
        elif 'multi cap' in name_lower or 'flexi cap' in name_lower:
            return 'Multi Cap'
        elif 'debt' in name_lower or 'bond' in name_lower:
            return 'Debt Fund'
        elif 'hybrid' in name_lower or 'balanced' in name_lower:
            return 'Hybrid Fund'
        elif 'elss' in name_lower or 'tax' in name_lower:
            return 'ELSS'
        elif 'sector' in name_lower or 'thematic' in name_lower:
            return 'Thematic/Sectoral'
        else:
            return 'Other'

    def generate_recommendation(
        self,
        actual_return: float,
        risk_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate investment recommendation based on actual future returns.

        Args:
            actual_return: Actual 3-month future return
            risk_metrics: Risk metrics dictionary

        Returns:
            Dictionary with recommendation and reasoning
        """
        # Determine recommendation
        if actual_return > 8:
            recommendation = "STRONG BUY"
            allocation = "30-40%"
        elif actual_return > 4:
            recommendation = "BUY"
            allocation = "20-30%"
        elif actual_return > -2:
            recommendation = "HOLD"
            allocation = "10-20%"
        else:
            recommendation = "SELL"
            allocation = "0-10%"

        # Determine risk level
        volatility = risk_metrics['volatility']
        if volatility > 20:
            risk_level = 'High'
        elif volatility > 12:
            risk_level = 'Moderate'
        else:
            risk_level = 'Low'

        return {
            'recommendation': recommendation,
            'allocation': allocation,
            'risk_level': risk_level,
            'expected_return': round(actual_return, 2)
        }

    def generate_training_example(
        self,
        scheme_code: str,
        as_of_date: datetime,
        lookback_days: int = 365
    ) -> Optional[Dict[str, Any]]:
        """
        Generate single rich training example for a specific fund and date.

        Args:
            scheme_code: Mutual fund scheme code
            as_of_date: Date to generate example for
            lookback_days: Days of historical data to use for context

        Returns:
            Dictionary with prompt, completion, and metadata
        """
        # Get fund data
        query = f"""
        SELECT date, nav
        FROM nav_history_20y
        WHERE scheme_code = ?
        AND date <= ?
        ORDER BY date DESC
        LIMIT ?
        """

        df = pd.read_sql_query(
            query,
            self.conn,
            params=(scheme_code, as_of_date.strftime('%Y-%m-%d'), lookback_days + 100)
        )

        if len(df) < 90:  # Need at least 3 months of data
            return None

        df = df.sort_values('date')
        df['date'] = pd.to_datetime(df['date'])

        # Filter to lookback period
        cutoff_date = as_of_date - timedelta(days=lookback_days)
        df = df[df['date'] >= cutoff_date]

        if len(df) < 90:
            return None

        # Get scheme name from funds table
        scheme_info = pd.read_sql_query(
            "SELECT scheme_name FROM funds WHERE scheme_code = ?",
            self.conn,
            params=(scheme_code,)
        )
        scheme_name = scheme_info['scheme_name'].iloc[0] if len(scheme_info) > 0 else f"Fund {scheme_code}"

        # Calculate metrics
        nav_series = df['nav']
        returns = self.calculate_returns(nav_series)
        risk_metrics = self.calculate_risk_metrics(nav_series)

        # Market phase
        market_phase = self.determine_market_phase(returns.get('30day', 0))

        # Fund category
        category = self.classify_fund_category(scheme_name)

        # Get actual future return (3 months ahead)
        # Get NAV approximately 90 days in the future
        future_date_min = as_of_date + timedelta(days=85)
        future_date_max = as_of_date + timedelta(days=95)

        future_query = f"""
        SELECT date, nav
        FROM nav_history_20y
        WHERE scheme_code = ?
        AND date > ?
        AND date <= ?
        ORDER BY date ASC
        LIMIT 1
        """

        future_df = pd.read_sql_query(
            future_query,
            self.conn,
            params=(scheme_code, as_of_date.strftime('%Y-%m-%d'), future_date_max.strftime('%Y-%m-%d'))
        )

        if len(future_df) == 0:
            # Try getting any future data point
            fallback_query = """
            SELECT date, nav
            FROM nav_history_20y
            WHERE scheme_code = ?
            AND date > ?
            ORDER BY date ASC
            LIMIT 1
            """
            future_df = pd.read_sql_query(
                fallback_query,
                self.conn,
                params=(scheme_code, as_of_date.strftime('%Y-%m-%d'))
            )

            if len(future_df) == 0:
                return None

        actual_future_nav = future_df['nav'].iloc[0]
        current_nav = nav_series.iloc[-1]
        actual_return = ((actual_future_nav / current_nav) - 1) * 100

        # Generate recommendation
        rec = self.generate_recommendation(actual_return, risk_metrics)

        # Create prompt
        prompt = f"""[CONTEXT]
Fund Name: {scheme_name}
Scheme Code: {scheme_code}
Category: {category}
Current NAV: ₹{current_nav:.2f}

[RECENT PERFORMANCE]
7-day Return: {returns.get('7day', 'N/A')}%
30-day Return: {returns.get('30day', 'N/A')}%
90-day Return: {returns.get('90day', 'N/A')}%
1-year Return: {returns.get('365day', 'N/A')}%

[RISK METRICS]
Volatility (annualized): {risk_metrics['volatility']}%
Sharpe Ratio: {risk_metrics['sharpe']}
Maximum Drawdown (1-year): {risk_metrics['max_drawdown']}%

[MARKET CONDITIONS]
Market Phase: {market_phase}
Date: {as_of_date.strftime('%Y-%m-%d')}

[QUESTION]
Based on current market conditions, fund performance, and risk metrics, predict the expected 3-month return and provide an investment recommendation.

[ANSWER]
"""

        # Create answer
        answer = f"""Expected 3-month return: {rec['expected_return']:.2f}%
Recommendation: {rec['recommendation']}
Risk Level: {rec['risk_level']}
Suggested Allocation: {rec['allocation']} for balanced portfolio

Reasoning:
1. Recent momentum: {returns.get('90day', 0):.1f}% in last 90 days indicates {"positive" if returns.get('90day', 0) > 0 else "negative"} trend
2. Market phase: {market_phase} suggests {"favorable" if market_phase == "Bull Market" else "cautious" if market_phase == "Bear Market" else "neutral"} conditions
3. Risk-adjusted returns (Sharpe: {risk_metrics['sharpe']:.2f}) indicate {"good" if risk_metrics['sharpe'] > 1 else "moderate" if risk_metrics['sharpe'] > 0.5 else "poor"} performance
4. Volatility ({risk_metrics['volatility']:.1f}%) is {"high" if risk_metrics['volatility'] > 20 else "moderate" if risk_metrics['volatility'] > 12 else "low"}
5. Fund category ({category}) provides {"stability" if "Large Cap" in category or "Debt" in category else "growth potential" if "Mid Cap" in category or "Small Cap" in category else "diversification"}

Risk Management:
- Set stop-loss at {-8 if rec['recommendation'] in ['BUY', 'STRONG BUY'] else -5}% from entry
- Recommended holding period: {"6-12 months" if rec['recommendation'] in ['BUY', 'STRONG BUY'] else "3-6 months"}
- Suitable for: {"Aggressive" if rec['risk_level'] == 'High' else "Moderate" if rec['risk_level'] == 'Moderate' else "Conservative"} investors
"""

        return {
            'prompt': prompt,
            'completion': answer,
            'metadata': {
                'scheme_code': scheme_code,
                'scheme_name': scheme_name,
                'as_of_date': as_of_date.strftime('%Y-%m-%d'),
                'actual_return': round(actual_return, 2),
                'recommendation': rec['recommendation'],
                'category': category
            }
        }

    def generate_dataset(
        self,
        num_examples: int = 10000,
        sample_schemes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate complete training dataset.

        Args:
            num_examples: Target number of training examples
            sample_schemes: List of specific schemes to sample (None = all schemes)

        Returns:
            List of training examples
        """
        self.connect()
        examples = []

        try:
            # Get list of schemes
            if sample_schemes is None:
                query = "SELECT DISTINCT scheme_code FROM nav_history_20y"
                schemes_df = pd.read_sql_query(query, self.conn)
                schemes = schemes_df['scheme_code'].tolist()
            else:
                schemes = sample_schemes

            print(f"Generating training data from {len(schemes)} schemes...")
            print(f"Target: {num_examples} examples")

            examples_per_scheme = max(1, num_examples // len(schemes))

            for idx, scheme_code in enumerate(schemes):
                if len(examples) >= num_examples:
                    break

                # Get date range for this scheme
                date_query = f"""
                SELECT MIN(date) as min_date, MAX(date) as max_date
                FROM nav_history_20y
                WHERE scheme_code = ?
                """
                dates_df = pd.read_sql_query(date_query, self.conn, params=(scheme_code,))

                if len(dates_df) == 0:
                    continue

                min_date_str = dates_df['min_date'].iloc[0]
                max_date_str = dates_df['max_date'].iloc[0]

                # Check if dates are valid
                if min_date_str is None or max_date_str is None or pd.isna(min_date_str) or pd.isna(max_date_str):
                    print(f"  Skipping scheme {scheme_code}: Invalid dates")
                    continue

                min_date = pd.to_datetime(min_date_str)
                max_date = pd.to_datetime(max_date_str)

                # Need at least 15 months of data (1 year history + 3 months future)
                if (max_date - min_date).days < 450:
                    continue

                # Sample random dates (excluding last 90 days for future prediction)
                sample_end = max_date - timedelta(days=90)
                sample_start = min_date + timedelta(days=365)

                if sample_end <= sample_start:
                    continue

                # Generate random dates
                date_range = (sample_end - sample_start).days
                num_samples = min(examples_per_scheme, date_range // 30)

                for _ in range(num_samples):
                    random_days = np.random.randint(0, date_range)
                    sample_date = sample_start + timedelta(days=random_days)

                    example = self.generate_training_example(scheme_code, sample_date)

                    if example:
                        examples.append(example)

                        if len(examples) >= num_examples:
                            break

                if (idx + 1) % 10 == 0:
                    print(f"  Progress: {len(examples)}/{num_examples} examples generated")

        finally:
            self.disconnect()

        print(f"\nGeneration complete! Total examples: {len(examples)}")
        return examples

    def save_dataset(
        self,
        examples: List[Dict[str, Any]],
        output_path: str = 'training_data_enhanced.jsonl'
    ):
        """
        Save training dataset in JSONL format.

        Args:
            examples: List of training examples
            output_path: Path to save JSONL file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')

        print(f"Training data saved to: {output_path}")
        print(f"Total examples: {len(examples)}")

        # Print statistics
        recommendations = [ex['metadata']['recommendation'] for ex in examples]
        print("\nRecommendation Distribution:")
        for rec in set(recommendations):
            count = recommendations.count(rec)
            pct = (count / len(recommendations)) * 100
            print(f"  {rec}: {count} ({pct:.1f}%)")


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("ADVANCED TRAINING DATA GENERATOR")
    print("=" * 80)

    # Path to database
    db_path = "../data/historical_20y.db"

    if Path(db_path).exists():
        generator = AdvancedTrainingDataGenerator(db_path)

        # Generate training data
        training_data = generator.generate_dataset(num_examples=1000)

        # Save to file
        generator.save_dataset(training_data, output_path="../data/training_data_enhanced.jsonl")

        # Show example
        if len(training_data) > 0:
            print("\n" + "=" * 80)
            print("SAMPLE TRAINING EXAMPLE")
            print("=" * 80)
            print(training_data[0]['prompt'])
            print(training_data[0]['completion'])
    else:
        print(f"Database not found: {db_path}")
        print("Please ensure historical data is collected first.")
