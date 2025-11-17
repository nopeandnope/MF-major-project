"""
Mutual Fund Backtesting Engine

Comprehensive backtesting framework for mutual fund strategies using the bt library.
Supports multiple timeframes, crisis scenario testing, and detailed performance metrics.
"""

import bt
import pandas as pd
import numpy as np
import quantstats as qs
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import sqlite3
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MutualFundBacktester:
    """
    Comprehensive backtesting engine for mutual fund strategies
    """

    def __init__(
        self,
        historical_data_path: Optional[str] = None,
        benchmark_data_path: Optional[str] = None,
        db_path: Optional[str] = None
    ):
        """
        Initialize backtester with historical data.

        Args:
            historical_data_path: Path to historical NAV data (CSV)
            benchmark_data_path: Path to benchmark data (e.g., Nifty 50)
            db_path: Path to SQLite database (alternative to CSV)
        """
        self.historical_data_path = historical_data_path
        self.db_path = db_path
        self.data = None
        self.benchmark = None
        self.results = {}

        # Load data
        if db_path:
            self.data = self.load_data_from_db(db_path)
        elif historical_data_path:
            self.data = self.load_data(historical_data_path)

        # Load benchmark if provided
        if benchmark_data_path:
            self.benchmark = self.load_benchmark(benchmark_data_path)

    def load_data_from_db(self, db_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load historical NAV data from SQLite database.

        Args:
            db_path: Path to database file

        Returns:
            Dictionary with 'prices' and 'returns' DataFrames
        """
        print(f"Loading data from database: {db_path}")
        conn = sqlite3.connect(db_path)

        # Load all NAV data (join with funds table to get scheme_name)
        query = """
        SELECT n.date, n.scheme_code, f.scheme_name, n.nav
        FROM nav_history_20y n
        LEFT JOIN funds f ON n.scheme_code = f.scheme_code
        ORDER BY n.date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)

        # Use scheme_name for better readability (fallback to scheme_code if name not available)
        df['identifier'] = df['scheme_name'].fillna(df['scheme_code'].astype(str))

        # Pivot to get funds as columns
        pivot_df = df.pivot(index='date', columns='identifier', values='nav')

        # Fill missing values forward (assuming NAV doesn't change on holidays)
        pivot_df = pivot_df.fillna(method='ffill')

        # Calculate returns
        returns_df = pivot_df.pct_change().fillna(0)

        print(f"Loaded {len(pivot_df.columns)} funds with {len(pivot_df)} days of data")
        print(f"Date range: {pivot_df.index.min()} to {pivot_df.index.max()}")

        return {
            'prices': pivot_df,
            'returns': returns_df
        }

    def load_data(self, path: str) -> Dict[str, pd.DataFrame]:
        """
        Load historical NAV data from CSV.

        Args:
            path: Path to CSV file

        Returns:
            Dictionary with 'prices' and 'returns' DataFrames
        """
        print(f"Loading data from CSV: {path}")
        df = pd.read_csv(path, parse_dates=['date'])

        # Use scheme_name if available, otherwise scheme_code
        identifier_col = 'scheme_name' if 'scheme_name' in df.columns else 'scheme_code'

        # Pivot to get funds as columns
        pivot_df = df.pivot(index='date', columns=identifier_col, values='nav')

        # Fill missing values forward
        pivot_df = pivot_df.fillna(method='ffill')

        # Calculate returns
        returns_df = pivot_df.pct_change().fillna(0)

        print(f"Loaded {len(pivot_df.columns)} funds with {len(pivot_df)} days of data")
        print(f"Date range: {pivot_df.index.min()} to {pivot_df.index.max()}")

        return {
            'prices': pivot_df,
            'returns': returns_df
        }

    def load_benchmark(self, path: str) -> pd.Series:
        """
        Load benchmark data (e.g., Nifty 50).

        Args:
            path: Path to benchmark CSV

        Returns:
            Series with benchmark prices
        """
        df = pd.read_csv(path, parse_dates=['date'], index_col='date')
        return df['close']

    def create_equal_weight_strategy(
        self,
        funds_list: List[str],
        rebalance_freq: str = 'quarterly',
        custom_data: Optional[pd.DataFrame] = None
    ) -> Any:
        """
        Create equal weight strategy with periodic rebalancing.

        Args:
            funds_list: List of fund names to include
            rebalance_freq: 'monthly', 'quarterly', 'yearly'
            custom_data: Optional custom data to use instead of self.data

        Returns:
            Configured Backtest object
        """
        # Select data for chosen funds
        if custom_data is not None:
            data = custom_data[funds_list]
        else:
            data = self.data['prices'][funds_list]

        # Define rebalancing frequency
        rebalance_algos = {
            'monthly': bt.algos.RunMonthly(),
            'quarterly': bt.algos.RunQuarterly(),
            'yearly': bt.algos.RunYearly()
        }

        # Create strategy
        strategy = bt.Strategy(
            f'EqualWeight_{rebalance_freq}',
            [
                rebalance_algos[rebalance_freq],
                bt.algos.SelectAll(),
                bt.algos.WeighEqually(),
                bt.algos.Rebalance()
            ]
        )

        return bt.Backtest(strategy, data)

    def create_model_based_strategy(
        self,
        model_predictions: pd.DataFrame,
        rebalance_freq: str = 'quarterly'
    ) -> Any:
        """
        Create strategy based on AI model predictions.

        Args:
            model_predictions: DataFrame with predicted returns/scores
            rebalance_freq: Rebalancing frequency

        Returns:
            Configured Backtest object
        """
        # Custom algo to weight based on model predictions
        class WeighByModelPrediction(bt.Algo):
            def __init__(self, predictions):
                super(WeighByModelPrediction, self).__init__()
                self.predictions = predictions

            def __call__(self, target):
                # Get predictions for current date
                current_date = target.now
                if current_date not in self.predictions.index:
                    return True

                pred = self.predictions.loc[current_date]

                # Normalize to weights (only positive predictions)
                pred_positive = pred[pred > 0]
                if len(pred_positive) == 0:
                    return True

                weights = pred_positive / pred_positive.sum()
                target.temp['weights'] = weights

                return True

        rebalance_algos = {
            'monthly': bt.algos.RunMonthly(),
            'quarterly': bt.algos.RunQuarterly(),
            'yearly': bt.algos.RunYearly()
        }

        strategy = bt.Strategy(
            f'ModelBased_{rebalance_freq}',
            [
                rebalance_algos[rebalance_freq],
                WeighByModelPrediction(model_predictions),
                bt.algos.Rebalance()
            ]
        )

        data = self.data['prices'][model_predictions.columns]
        return bt.Backtest(strategy, data)

    def run_timeframe_tests(
        self,
        strategy_name: str = 'equal_weight',
        num_funds: int = 10
    ) -> Dict[str, Any]:
        """
        Run backtests across multiple timeframes.

        Args:
            strategy_name: Type of strategy to use
            num_funds: Number of top funds to include

        Returns:
            Dictionary of results for each timeframe
        """
        results = {}

        # Define timeframes (end date is most recent)
        end_date = self.data['prices'].index.max()

        timeframes = {
            '3_months': end_date - timedelta(days=90),
            '6_months': end_date - timedelta(days=180),
            '1_year': end_date - timedelta(days=365),
            '2_years': end_date - timedelta(days=365*2),
            '3_years': end_date - timedelta(days=365*3),
            '5_years': end_date - timedelta(days=365*5),
            '10_years': end_date - timedelta(days=365*10),
            '15_years': end_date - timedelta(days=365*15)
        }

        # Select top funds by average return
        avg_returns = self.data['returns'].mean()
        top_funds = avg_returns.nlargest(num_funds).index.tolist()

        print(f"\nRunning timeframe tests with top {num_funds} funds...")
        print(f"Selected funds: {top_funds[:3]}..." if len(top_funds) > 3 else f"Selected funds: {top_funds}")

        for period_name, start_date in timeframes.items():
            try:
                # Filter data for this timeframe
                mask = (self.data['prices'].index >= start_date) & (self.data['prices'].index <= end_date)
                period_data = self.data['prices'][mask]

                if len(period_data) < 30:  # Skip if insufficient data
                    print(f"  {period_name}: Insufficient data (only {len(period_data)} days)")
                    continue

                # Filter to only include funds that have data for this period
                valid_funds = [f for f in top_funds if f in period_data.columns and period_data[f].notna().sum() > 30]

                if len(valid_funds) == 0:
                    print(f"  {period_name}: No valid funds with sufficient data")
                    continue

                # Create and run backtest with filtered period data
                backtest = self.create_equal_weight_strategy(
                    valid_funds,
                    rebalance_freq='quarterly',
                    custom_data=period_data
                )

                result = bt.run(backtest)
                results[period_name] = result
                print(f"  {period_name}: Completed ({len(period_data)} days, {len(valid_funds)} funds)")

            except Exception as e:
                print(f"  {period_name}: Failed - {str(e)}")
                continue

        return results

    def test_crisis_scenarios(
        self,
        num_funds: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Test strategy performance during market crashes.

        Args:
            num_funds: Number of top funds to include

        Returns:
            Dictionary of results for each crisis period
        """
        crisis_periods = {
            'covid_crash_and_recovery': {
                'start': '2019-01-01',
                'end': '2021-12-31',
                'description': 'COVID-19 crash (Feb-Mar 2020) and recovery'
            },
            'covid_crash_only': {
                'start': '2020-01-01',
                'end': '2020-06-30',
                'description': 'COVID-19 crash period only'
            },
            '2022_correction': {
                'start': '2022-01-01',
                'end': '2022-12-31',
                'description': '2022 market correction'
            },
            'demonetization': {
                'start': '2016-09-01',
                'end': '2017-03-31',
                'description': '2016 demonetization impact'
            }
        }

        results = {}
        avg_returns = self.data['returns'].mean()
        top_funds = avg_returns.nlargest(num_funds).index.tolist()

        print(f"\nTesting crisis scenarios with top {num_funds} funds...")

        for crisis_name, period_info in crisis_periods.items():
            try:
                start_date = pd.to_datetime(period_info['start'])
                end_date = pd.to_datetime(period_info['end'])

                # Filter data
                mask = (self.data['prices'].index >= start_date) & (self.data['prices'].index <= end_date)
                crisis_data = self.data['prices'][mask]

                if len(crisis_data) < 30:
                    print(f"  {crisis_name}: Insufficient data (only {len(crisis_data)} days)")
                    continue

                # Filter to only include funds that have data for this period
                valid_funds = [f for f in top_funds if f in crisis_data.columns and crisis_data[f].notna().sum() > 30]

                if len(valid_funds) == 0:
                    print(f"  {crisis_name}: No valid funds with sufficient data")
                    continue

                # Run backtest with filtered crisis data
                backtest = self.create_equal_weight_strategy(
                    valid_funds,
                    rebalance_freq='monthly',  # Monthly rebalancing during crisis
                    custom_data=crisis_data
                )

                result = bt.run(backtest)
                results[crisis_name] = {
                    'result': result,
                    'description': period_info['description']
                }
                print(f"  {crisis_name}: Completed ({len(crisis_data)} days, {len(valid_funds)} funds)")

            except Exception as e:
                print(f"  {crisis_name}: Failed - {str(e)}")
                continue

        return results

    def calculate_performance_metrics(self, backtest_result: Any) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Args:
            backtest_result: Result from bt.run()

        Returns:
            Dictionary of performance metrics
        """
        # Get the first (and typically only) strategy name
        strategy_name = list(backtest_result.prices.columns)[0]

        # Get prices and calculate returns for the strategy
        prices = backtest_result.prices[strategy_name]
        returns = prices.pct_change().fillna(0)

        # Basic metrics
        total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

        # Annualized return
        years = (prices.index[-1] - prices.index[0]).days / 365.25
        annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100

        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252) * 100

        # Sharpe Ratio (assuming 6.5% risk-free rate for India)
        risk_free_rate = 0.065
        sharpe_ratio = (annualized_return/100 - risk_free_rate) / (volatility/100) if volatility > 0 else 0

        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return/100 - risk_free_rate) / downside_std if downside_std > 0 else 0

        # Calmar Ratio
        calmar_ratio = (annualized_return/100) / abs(max_drawdown/100) if max_drawdown != 0 else 0

        # Win rate
        win_rate = (returns > 0).sum() / len(returns) * 100

        return {
            'total_return': round(total_return, 2),
            'annualized_return': round(annualized_return, 2),
            'volatility': round(volatility, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'calmar_ratio': round(calmar_ratio, 2),
            'win_rate': round(win_rate, 2),
            'total_days': len(returns),
            'start_date': str(prices.index[0].date()),
            'end_date': str(prices.index[-1].date())
        }

    def generate_html_report(
        self,
        backtest_result: Any,
        output_path: str = 'reports/backtest_report.html'
    ):
        """
        Generate comprehensive HTML report using quantstats.

        Args:
            backtest_result: Result from bt.run()
            output_path: Path to save HTML report
        """
        # Get the first strategy name and calculate returns
        strategy_name = list(backtest_result.prices.columns)[0]
        prices = backtest_result.prices[strategy_name]
        returns = prices.pct_change().fillna(0)

        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if self.benchmark is not None:
            # Align benchmark with strategy returns
            benchmark_aligned = self.benchmark.reindex(returns.index).pct_change().fillna(0)
            qs.reports.html(returns, benchmark_aligned, output=output_path, title="MF Backtest Report")
        else:
            qs.reports.html(returns, output=output_path, title="MF Backtest Report")

        print(f"Report generated: {output_path}")

    def generate_combined_report(
        self,
        timeframe_results: Dict[str, Any],
        crisis_results: Dict[str, Dict[str, Any]] = None,
        output_path: str = 'reports/backtest_comprehensive.html'
    ):
        """
        Generate a comprehensive combined report for all timeframes and crisis scenarios.

        Args:
            timeframe_results: Dictionary of timeframe results
            crisis_results: Dictionary of crisis scenario results
            output_path: Path to save HTML report
        """
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend

        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Collect all metrics
        timeframe_order = ['3_months', '6_months', '1_year', '2_years', '3_years', '5_years', '10_years', '15_years']
        all_metrics = []

        for period in timeframe_order:
            if period in timeframe_results:
                metrics = self.calculate_performance_metrics(timeframe_results[period])
                metrics['period'] = period.replace('_', ' ').title()
                metrics['period_key'] = period
                all_metrics.append(metrics)

        # Build HTML report
        html_content = self._build_comprehensive_html(all_metrics, timeframe_results, crisis_results)

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Comprehensive report generated: {output_path}")

    def _build_comprehensive_html(
        self,
        all_metrics: List[Dict],
        timeframe_results: Dict[str, Any],
        crisis_results: Dict[str, Dict[str, Any]] = None
    ) -> str:
        """Build comprehensive HTML report content."""

        # Create comparison charts
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO

        def fig_to_base64(fig):
            """Convert matplotlib figure to base64 string."""
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            return image_base64

        # Chart 1: Returns Comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Multi-Timeframe Performance Analysis', fontsize=16, fontweight='bold')

        periods = [m['period'] for m in all_metrics]
        total_returns = [m['total_return'] for m in all_metrics]
        ann_returns = [m['annualized_return'] for m in all_metrics]
        sharpe_ratios = [m['sharpe_ratio'] for m in all_metrics]
        max_drawdowns = [abs(m['max_drawdown']) for m in all_metrics]

        # Total Returns
        axes[0, 0].bar(periods, total_returns, color='#2E86AB')
        axes[0, 0].set_title('Total Returns by Timeframe', fontweight='bold')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Annualized Returns
        axes[0, 1].bar(periods, ann_returns, color='#A23B72')
        axes[0, 1].set_title('Annualized Returns by Timeframe', fontweight='bold')
        axes[0, 1].set_ylabel('Annualized Return (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)

        # Sharpe Ratios
        axes[1, 0].bar(periods, sharpe_ratios, color='#F18F01')
        axes[1, 0].set_title('Sharpe Ratio by Timeframe', fontweight='bold')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].axhline(y=1.0, color='green', linestyle='--', label='Good (>1.0)')
        axes[1, 0].axhline(y=2.0, color='darkgreen', linestyle='--', label='Excellent (>2.0)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Max Drawdowns
        axes[1, 1].bar(periods, max_drawdowns, color='#C73E1D')
        axes[1, 1].set_title('Maximum Drawdown by Timeframe', fontweight='bold')
        axes[1, 1].set_ylabel('Max Drawdown (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        chart_img = fig_to_base64(fig)

        # Build HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Backtesting Report - Mutual Fund Advisor</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .section {{
            padding: 40px;
            border-bottom: 2px solid #f0f0f0;
        }}
        .section:last-child {{
            border-bottom: none;
        }}
        .section-title {{
            font-size: 2em;
            color: #2E86AB;
            margin-bottom: 25px;
            border-left: 5px solid #A23B72;
            padding-left: 15px;
        }}
        .section-subtitle {{
            font-size: 1.4em;
            color: #555;
            margin-top: 30px;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2E86AB;
        }}
        .metric-value.positive {{
            color: #27AE60;
        }}
        .metric-value.negative {{
            color: #E74C3C;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        th {{
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 1px;
        }}
        td {{
            padding: 15px;
            border-bottom: 1px solid #e0e0e0;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        tr:nth-child(even) {{
            background-color: #fafafa;
        }}
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .explanation {{
            background: #FFF9E6;
            border-left: 4px solid #F18F01;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .explanation h4 {{
            color: #D68910;
            margin-bottom: 10px;
            font-size: 1.2em;
        }}
        .explanation p {{
            line-height: 1.8;
            color: #555;
        }}
        .info-box {{
            background: #E8F5E9;
            border: 2px solid #4CAF50;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
        }}
        .info-box h4 {{
            color: #2E7D32;
            margin-bottom: 10px;
        }}
        .warning-box {{
            background: #FFEBEE;
            border: 2px solid #F44336;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
        }}
        .warning-box h4 {{
            color: #C62828;
            margin-bottom: 10px;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-right: 8px;
        }}
        .badge-short {{
            background: #E3F2FD;
            color: #1976D2;
        }}
        .badge-medium {{
            background: #F3E5F5;
            color: #7B1FA2;
        }}
        .badge-long {{
            background: #E8F5E9;
            color: #388E3C;
        }}
        .footer {{
            background: #2C3E50;
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .period-highlight {{
            font-weight: bold;
            color: #2E86AB;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Comprehensive Backtesting Report</h1>
            <p>Multi-Timeframe Performance Analysis & Strategy Evaluation</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>

        <!-- Executive Summary -->
        <div class="section">
            <h2 class="section-title">📈 Executive Summary</h2>
            <p style="line-height: 1.8; margin-bottom: 20px;">
                This comprehensive report analyzes the backtesting results of an equal-weight mutual fund portfolio strategy
                across <strong>{len(all_metrics)} different timeframes</strong>, ranging from short-term (3 months) to long-term (15 years).
                The strategy employs quarterly rebalancing and equal weighting across selected top-performing mutual funds.
            </p>

            <div class="metrics-grid">
"""

        # Add summary metrics for best performing periods
        if len(all_metrics) > 0:
            best_return = max(all_metrics, key=lambda x: x['annualized_return'])
            best_sharpe = max(all_metrics, key=lambda x: x['sharpe_ratio'])
            lowest_drawdown = min(all_metrics, key=lambda x: abs(x['max_drawdown']))

            html += f"""
                <div class="metric-card">
                    <div class="metric-label">Best Annualized Return</div>
                    <div class="metric-value positive">{best_return['annualized_return']:.2f}%</div>
                    <p style="margin-top: 10px; color: #666; font-size: 0.9em;">{best_return['period']}</p>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Best Risk-Adjusted Return</div>
                    <div class="metric-value positive">{best_sharpe['sharpe_ratio']:.2f}</div>
                    <p style="margin-top: 10px; color: #666; font-size: 0.9em;">{best_sharpe['period']} (Sharpe Ratio)</p>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Lowest Drawdown</div>
                    <div class="metric-value positive">{lowest_drawdown['max_drawdown']:.2f}%</div>
                    <p style="margin-top: 10px; color: #666; font-size: 0.9em;">{lowest_drawdown['period']}</p>
                </div>
"""

        html += """
            </div>
        </div>

        <!-- Visual Analysis -->
        <div class="section">
            <h2 class="section-title">📊 Visual Performance Comparison</h2>
            <div class="chart-container">
"""
        html += f'<img src="data:image/png;base64,{chart_img}" alt="Performance Charts">'
        html += """
            </div>

            <div class="explanation">
                <h4>📖 Understanding the Charts</h4>
                <p><strong>Total Returns:</strong> Shows absolute returns over each period. Longer timeframes naturally show higher total returns due to compounding.</p>
                <p><strong>Annualized Returns:</strong> Normalizes returns to a per-year basis, allowing fair comparison across different timeframes.</p>
                <p><strong>Sharpe Ratio:</strong> Measures risk-adjusted returns. Values >1.0 are good, >2.0 are excellent. Higher is better.</p>
                <p><strong>Maximum Drawdown:</strong> Worst peak-to-trough decline. Lower is better as it indicates less risk.</p>
            </div>
        </div>

        <!-- Detailed Metrics Table -->
        <div class="section">
            <h2 class="section-title">📋 Detailed Performance Metrics</h2>

            <h3 class="section-subtitle">
                <span class="badge badge-short">SHORT TERM</span> Performance (3-6 Months)
            </h3>
            <table>
                <thead>
                    <tr>
                        <th>Period</th>
                        <th>Total Return</th>
                        <th>Ann. Return</th>
                        <th>Volatility</th>
                        <th>Sharpe Ratio</th>
                        <th>Max Drawdown</th>
                        <th>Sortino Ratio</th>
                        <th>Win Rate</th>
                    </tr>
                </thead>
                <tbody>
"""

        # Short term rows
        for m in all_metrics:
            if m['period_key'] in ['3_months', '6_months']:
                html += f"""
                    <tr>
                        <td class="period-highlight">{m['period']}</td>
                        <td>{m['total_return']:.2f}%</td>
                        <td>{m['annualized_return']:.2f}%</td>
                        <td>{m['volatility']:.2f}%</td>
                        <td>{m['sharpe_ratio']:.2f}</td>
                        <td>{m['max_drawdown']:.2f}%</td>
                        <td>{m['sortino_ratio']:.2f}</td>
                        <td>{m['win_rate']:.2f}%</td>
                    </tr>
"""

        html += """
                </tbody>
            </table>

            <h3 class="section-subtitle">
                <span class="badge badge-medium">MEDIUM TERM</span> Performance (1-3 Years)
            </h3>
            <table>
                <thead>
                    <tr>
                        <th>Period</th>
                        <th>Total Return</th>
                        <th>Ann. Return</th>
                        <th>Volatility</th>
                        <th>Sharpe Ratio</th>
                        <th>Max Drawdown</th>
                        <th>Sortino Ratio</th>
                        <th>Win Rate</th>
                    </tr>
                </thead>
                <tbody>
"""

        # Medium term rows
        for m in all_metrics:
            if m['period_key'] in ['1_year', '2_years', '3_years']:
                html += f"""
                    <tr>
                        <td class="period-highlight">{m['period']}</td>
                        <td>{m['total_return']:.2f}%</td>
                        <td>{m['annualized_return']:.2f}%</td>
                        <td>{m['volatility']:.2f}%</td>
                        <td>{m['sharpe_ratio']:.2f}</td>
                        <td>{m['max_drawdown']:.2f}%</td>
                        <td>{m['sortino_ratio']:.2f}</td>
                        <td>{m['win_rate']:.2f}%</td>
                    </tr>
"""

        html += """
                </tbody>
            </table>

            <h3 class="section-subtitle">
                <span class="badge badge-long">LONG TERM</span> Performance (5-15 Years)
            </h3>
            <table>
                <thead>
                    <tr>
                        <th>Period</th>
                        <th>Total Return</th>
                        <th>Ann. Return</th>
                        <th>Volatility</th>
                        <th>Sharpe Ratio</th>
                        <th>Max Drawdown</th>
                        <th>Sortino Ratio</th>
                        <th>Win Rate</th>
                    </tr>
                </thead>
                <tbody>
"""

        # Long term rows
        for m in all_metrics:
            if m['period_key'] in ['5_years', '10_years', '15_years']:
                html += f"""
                    <tr>
                        <td class="period-highlight">{m['period']}</td>
                        <td>{m['total_return']:.2f}%</td>
                        <td>{m['annualized_return']:.2f}%</td>
                        <td>{m['volatility']:.2f}%</td>
                        <td>{m['sharpe_ratio']:.2f}</td>
                        <td>{m['max_drawdown']:.2f}%</td>
                        <td>{m['sortino_ratio']:.2f}</td>
                        <td>{m['win_rate']:.2f}%</td>
                    </tr>
"""

        html += """
                </tbody>
            </table>
        </div>

        <!-- Crisis Scenarios -->
"""

        if crisis_results and len(crisis_results) > 0:
            html += """
        <div class="section">
            <h2 class="section-title">⚠️ Crisis Scenario Analysis</h2>
            <p style="line-height: 1.8; margin-bottom: 20px;">
                Testing how the strategy performs during market crashes and corrections provides insight into downside risk management.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Crisis Scenario</th>
                        <th>Description</th>
                        <th>Total Return</th>
                        <th>Max Drawdown</th>
                        <th>Sharpe Ratio</th>
                        <th>Days</th>
                    </tr>
                </thead>
                <tbody>
"""
            for crisis_name, crisis_info in crisis_results.items():
                metrics = self.calculate_performance_metrics(crisis_info['result'])
                html += f"""
                    <tr>
                        <td class="period-highlight">{crisis_name.replace('_', ' ').title()}</td>
                        <td>{crisis_info['description']}</td>
                        <td>{metrics['total_return']:.2f}%</td>
                        <td>{metrics['max_drawdown']:.2f}%</td>
                        <td>{metrics['sharpe_ratio']:.2f}</td>
                        <td>{metrics['total_days']}</td>
                    </tr>
"""
            html += """
                </tbody>
            </table>
        </div>
"""

        # Metrics Explanation Section
        html += """
        <!-- Metrics Explanation -->
        <div class="section">
            <h2 class="section-title">📚 Understanding Key Metrics</h2>

            <div class="explanation">
                <h4>📊 Total Return</h4>
                <p>The absolute percentage gain or loss from the beginning to the end of the period.
                Formula: ((Final Value - Initial Value) / Initial Value) × 100.
                This shows the raw performance but doesn't account for the time taken.</p>
            </div>

            <div class="explanation">
                <h4>📈 Annualized Return</h4>
                <p>The geometric average amount of money earned by an investment each year over a given time period.
                This normalizes returns across different timeframes, making them comparable.
                Formula: ((1 + Total Return)^(1/Years) - 1) × 100.
                <strong>Good:</strong> >12% for equity funds, <strong>Excellent:</strong> >15%</p>
            </div>

            <div class="explanation">
                <h4>📉 Volatility (Standard Deviation)</h4>
                <p>Measures the dispersion of returns around the mean, indicating investment risk.
                Higher volatility means more unpredictable returns.
                Annualized by multiplying daily std dev by √252 (trading days).
                <strong>Low:</strong> <15%, <strong>Medium:</strong> 15-25%, <strong>High:</strong> >25%</p>
            </div>

            <div class="explanation">
                <h4>⚡ Sharpe Ratio</h4>
                <p>Measures risk-adjusted returns by comparing excess returns to volatility.
                Formula: (Return - Risk-Free Rate) / Volatility. Uses 6.5% risk-free rate (Indian context).
                <strong>Interpretation:</strong> <1.0 = Poor, 1.0-2.0 = Good, 2.0-3.0 = Very Good, >3.0 = Excellent</p>
            </div>

            <div class="explanation">
                <h4>💥 Maximum Drawdown</h4>
                <p>The largest peak-to-trough decline during the investment period.
                Shows worst-case loss an investor would have experienced.
                <strong>Low Risk:</strong> <10%, <strong>Medium Risk:</strong> 10-20%, <strong>High Risk:</strong> >20%</p>
            </div>

            <div class="explanation">
                <h4>🎯 Sortino Ratio</h4>
                <p>Similar to Sharpe Ratio but only considers downside volatility (negative returns).
                Better measure for strategies with asymmetric return distributions.
                Higher values indicate better risk-adjusted returns with focus on downside protection.</p>
            </div>

            <div class="explanation">
                <h4>📊 Calmar Ratio</h4>
                <p>Annualized return divided by maximum drawdown.
                Measures return per unit of worst-case risk.
                <strong>Good:</strong> >3.0, <strong>Excellent:</strong> >5.0</p>
            </div>

            <div class="explanation">
                <h4>🎲 Win Rate</h4>
                <p>Percentage of days with positive returns.
                Higher win rate indicates more consistent performance.
                <strong>Typical:</strong> 50-60% for equity portfolios</p>
            </div>
        </div>

        <!-- Key Insights -->
        <div class="section">
            <h2 class="section-title">💡 Key Insights & Recommendations</h2>

            <div class="info-box">
                <h4>✅ Strengths Identified</h4>
                <ul style="line-height: 1.8; margin-left: 20px; color: #555;">
"""

        # Add dynamic insights
        if len(all_metrics) > 0:
            avg_sharpe = sum(m['sharpe_ratio'] for m in all_metrics) / len(all_metrics)
            avg_win_rate = sum(m['win_rate'] for m in all_metrics) / len(all_metrics)

            if avg_sharpe > 1.0:
                html += f"<li><strong>Strong Risk-Adjusted Returns:</strong> Average Sharpe Ratio of {avg_sharpe:.2f} indicates good risk-adjusted performance across timeframes.</li>"

            if avg_win_rate > 55:
                html += f"<li><strong>Consistent Positive Performance:</strong> Average win rate of {avg_win_rate:.1f}% shows regular positive returns.</li>"

            long_term = [m for m in all_metrics if m['period_key'] in ['5_years', '10_years', '15_years']]
            if long_term:
                avg_lt_return = sum(m['annualized_return'] for m in long_term) / len(long_term)
                if avg_lt_return > 12:
                    html += f"<li><strong>Excellent Long-Term Performance:</strong> Long-term annualized returns averaging {avg_lt_return:.2f}% demonstrate wealth creation potential.</li>"

        html += """
                </ul>
            </div>

            <div class="warning-box">
                <h4>⚠️ Risk Considerations</h4>
                <ul style="line-height: 1.8; margin-left: 20px; color: #555;">
"""

        # Add risk warnings
        if len(all_metrics) > 0:
            max_dd = min(m['max_drawdown'] for m in all_metrics)
            if max_dd < -20:
                html += f"<li><strong>Significant Drawdown Risk:</strong> Maximum drawdown of {max_dd:.2f}% observed. Investors should be prepared for temporary losses.</li>"

            short_term = [m for m in all_metrics if m['period_key'] in ['3_months', '6_months']]
            if short_term:
                for m in short_term:
                    if m['volatility'] > 20:
                        html += f"<li><strong>High Short-Term Volatility:</strong> {m['period']} shows {m['volatility']:.2f}% volatility, indicating unstable short-term returns.</li>"

        html += """
                    <li><strong>Past Performance Disclaimer:</strong> Historical returns do not guarantee future performance. Market conditions change over time.</li>
                    <li><strong>Equal-Weight Limitation:</strong> This strategy uses equal weighting. More sophisticated weighting schemes might improve performance.</li>
                </ul>
            </div>

            <div class="info-box">
                <h4>🎯 Strategic Recommendations</h4>
                <ul style="line-height: 1.8; margin-left: 20px; color: #555;">
                    <li><strong>Investment Horizon:</strong> Based on the analysis, a minimum 3-5 year investment horizon is recommended for optimal returns.</li>
                    <li><strong>Rebalancing:</strong> Quarterly rebalancing helps maintain target allocations and can enhance returns through systematic profit-taking.</li>
                    <li><strong>Diversification:</strong> The equal-weight approach ensures no single fund dominates the portfolio, reducing concentration risk.</li>
                    <li><strong>Monitoring:</strong> Regular review (quarterly) is recommended to ensure funds continue meeting performance criteria.</li>
                    <li><strong>Risk Management:</strong> Consider implementing stop-loss or trailing stop mechanisms for additional downside protection.</li>
                </ul>
            </div>
        </div>

        <!-- Strategy Details -->
        <div class="section">
            <h2 class="section-title">⚙️ Strategy Methodology</h2>

            <div style="background: #F5F5F5; padding: 25px; border-radius: 8px; line-height: 1.8;">
                <h4 style="color: #2E86AB; margin-bottom: 15px;">Portfolio Construction</h4>
                <ul style="margin-left: 20px; color: #555;">
                    <li><strong>Universe Selection:</strong> Top-performing mutual funds based on historical average returns</li>
                    <li><strong>Weighting Scheme:</strong> Equal weight allocation across all selected funds</li>
                    <li><strong>Rebalancing:</strong> Quarterly rebalancing to maintain equal weights</li>
                    <li><strong>Execution:</strong> Simulated using actual historical NAV data</li>
                </ul>

                <h4 style="color: #2E86AB; margin-top: 25px; margin-bottom: 15px;">Assumptions & Constraints</h4>
                <ul style="margin-left: 20px; color: #555;">
                    <li><strong>Transaction Costs:</strong> Not included in current simulation (conservative approach)</li>
                    <li><strong>Taxes:</strong> Not considered (would vary by investor type and holding period)</li>
                    <li><strong>Liquidity:</strong> Assumes sufficient liquidity for rebalancing</li>
                    <li><strong>Slippage:</strong> Not modeled (NAV execution assumed)</li>
                    <li><strong>Survivorship Bias:</strong> Uses currently available funds; historical delistings may affect results</li>
                </ul>

                <h4 style="color: #2E86AB; margin-top: 25px; margin-bottom: 15px;">Risk-Free Rate</h4>
                <p style="color: #555;">
                    A risk-free rate of <strong>6.5%</strong> is used for Sharpe and Sortino ratio calculations,
                    representing typical Indian government securities yields. This serves as the benchmark for
                    evaluating risk-adjusted returns.
                </p>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <h3 style="margin-bottom: 15px;">📊 Mutual Fund Advisor - Backtesting Engine</h3>
            <p>This report is generated automatically from historical mutual fund data.</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                Report Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
            </p>
            <p style="margin-top: 20px; font-size: 0.85em; opacity: 0.8;">
                ⚠️ <strong>Disclaimer:</strong> This analysis is for educational purposes only and does not constitute investment advice.
                Past performance does not guarantee future results. Consult with a qualified financial advisor before making investment decisions.
            </p>
        </div>
    </div>
</body>
</html>
"""

        return html

    def compare_strategies(
        self,
        strategies_dict: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Compare multiple strategies side-by-side.

        Args:
            strategies_dict: Dict of {strategy_name: backtest_result}

        Returns:
            DataFrame with comparison metrics
        """
        comparison = {}

        for name, result in strategies_dict.items():
            metrics = self.calculate_performance_metrics(result)
            comparison[name] = metrics

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison).T

        # Add rankings
        comparison_df['rank_sharpe'] = comparison_df['sharpe_ratio'].rank(ascending=False)
        comparison_df['rank_return'] = comparison_df['annualized_return'].rank(ascending=False)
        comparison_df['rank_drawdown'] = comparison_df['max_drawdown'].rank(ascending=True)

        return comparison_df


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("MUTUAL FUND BACKTESTING ENGINE")
    print("=" * 80)

    # Initialize backtester (replace with your actual database path)
    db_path = "../data/historical_20y.db"

    if Path(db_path).exists():
        backtester = MutualFundBacktester(db_path=db_path)

        # Run timeframe tests
        print("\n" + "=" * 80)
        print("RUNNING MULTI-TIMEFRAME BACKTESTS")
        print("=" * 80)
        timeframe_results = backtester.run_timeframe_tests(num_funds=10)

        # Print results
        for period, result in timeframe_results.items():
            metrics = backtester.calculate_performance_metrics(result)
            print(f"\n{period.upper().replace('_', ' ')} Performance:")
            print(f"  Total Return: {metrics['total_return']}%")
            print(f"  Annualized Return: {metrics['annualized_return']}%")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']}")
            print(f"  Max Drawdown: {metrics['max_drawdown']}%")

        # Generate HTML report for 1-year period if available
        if '1_year' in timeframe_results:
            backtester.generate_html_report(
                timeframe_results['1_year'],
                output_path='../reports/1year_backtest.html'
            )

        print("\n" + "=" * 80)
    else:
        print(f"Database not found: {db_path}")
        print("Please ensure historical data is collected first.")
