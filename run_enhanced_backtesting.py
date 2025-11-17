"""
Enhanced Backtesting and Evaluation Runner

This script runs the COMPLETE improved backtesting framework with:
- Multiple allocation strategies (Conservative, Balanced, Aggressive, Risk Parity, Momentum)
- Multi-factor fund selection
- Risk management (stop-loss, position sizing)
- Market regime detection
- Monte Carlo simulation
- Rolling performance analysis
- Comprehensive reporting

Usage:
    python run_enhanced_backtesting.py
    python run_enhanced_backtesting.py --strategies all
    python run_enhanced_backtesting.py --monte-carlo --simulations 10000
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add directories to path
sys.path.insert(0, str(Path(__file__).parent))

from backtesting.enhanced_backtest_engine import EnhancedMutualFundBacktester
from backtesting.risk_management import RiskManager, DrawdownController
from backtesting.market_analysis import MarketRegimeDetector, generate_market_report
from backtesting.advanced_analytics import MonteCarloSimulation, RollingAnalysis
from src.data_quality_validator import DataQualityValidator


def print_section_header(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 100)
    print(f"{title:^100}")
    print("=" * 100)


def print_subsection(title: str):
    """Print formatted subsection"""
    print("\n" + "-" * 100)
    print(f"{title}")
    print("-" * 100)


def run_data_quality_check(db_path: str):
    """Run data quality validation"""
    print_section_header("STEP 1: DATA QUALITY VALIDATION")

    try:
        validator = DataQualityValidator(data_source='mfapi.in')
        db_report = validator.validate_database(db_path)

        print(f"\n{'Database Quality Report':^100}")
        print("-" * 100)
        print(f"  Database Path:      {db_path}")
        print(f"  Total Schemes:      {db_report['total_schemes']}")
        print(f"  Passed Schemes:     {db_report['passed_schemes']} ✓")
        print(f"  Failed Schemes:     {db_report['failed_schemes']}")
        print(f"  Pass Rate:          {db_report['pass_rate']:.1f}%")
        print(f"  Avg Quality Score:  {db_report['avg_quality_score']:.2f}/100")

        if db_report['pass_rate'] < 90:
            print(f"\n  ⚠ Warning: Quality score below 90%")

        return db_report

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return None


def run_enhanced_backtesting(db_path: str, num_funds: int = 10, timeframe_years: int = 5):
    """Run comprehensive enhanced backtesting"""
    print_section_header("STEP 2: ENHANCED STRATEGY COMPARISON")

    try:
        # Initialize backtester
        backtester = EnhancedMutualFundBacktester(db_path=db_path)

        # Run strategy comparison
        timeframe_days = timeframe_years * 365
        results = backtester.compare_strategies(
            num_funds=num_funds,
            timeframe_days=timeframe_days
        )

        if not results:
            print("\n⚠ No strategy results generated")
            return None

        # Print results
        print_subsection("Strategy Performance Comparison")

        if len(results) == 0:
            print("\n⚠ No strategies completed successfully")
            return None

        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df[[
            'net_annualized_return', 'sharpe_ratio', 'max_drawdown',
            'sortino_ratio', 'volatility', 'win_rate'
        ]]

        # Rename columns for display
        comparison_df.columns = [
            'Annual Return (%)', 'Sharpe', 'Max DD (%)',
            'Sortino', 'Volatility (%)', 'Win Rate (%)'
        ]

        print(f"\n{comparison_df.to_string()}")

        # Find best strategy
        best_sharpe = comparison_df['Sharpe'].idxmax()
        best_return = comparison_df['Annual Return (%)'].idxmax()
        lowest_dd = comparison_df['Max DD (%)'].idxmax()  # Max because it's negative

        print(f"\n{'Performance Highlights':^100}")
        print("-" * 100)
        print(f"  🏆 Best Risk-Adjusted (Sharpe):  {best_sharpe}")
        print(f"  📈 Highest Returns:               {best_return}")
        print(f"  🛡️  Lowest Drawdown:               {lowest_dd}")

        # Risk analysis
        print_subsection("Risk Analysis Summary")

        for strategy_name, metrics in results.items():
            risk_level = "LOW" if abs(metrics['max_drawdown']) < 15 else \
                        "MODERATE" if abs(metrics['max_drawdown']) < 25 else "HIGH"

            print(f"\n  {strategy_name}:")
            print(f"    Risk Level:         {risk_level}")
            print(f"    Max Drawdown:       {metrics['max_drawdown']:.2f}%")
            print(f"    Volatility:         {metrics['volatility']:.2f}%")
            print(f"    Best Month:         +{metrics['best_month']:.2f}%")
            print(f"    Worst Month:        {metrics['worst_month']:.2f}%")
            print(f"    Avg DD Duration:    {metrics['avg_dd_duration_days']:.0f} days")

        return results

    except Exception as e:
        print(f"\n✗ Error during backtesting: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_market_analysis(db_path: str):
    """Run market regime and technical analysis"""
    print_section_header("STEP 3: MARKET ANALYSIS")

    try:
        # Load portfolio data for analysis
        backtester = EnhancedMutualFundBacktester(db_path=db_path)
        portfolio_prices = backtester.data['prices'].mean(axis=1)  # Simple average

        # Regime detection
        detector = MarketRegimeDetector()
        current_regime = detector.detect_regime(portfolio_prices)

        print(f"\n{'Current Market Regime':^100}")
        print("-" * 100)
        print(f"  Status: {current_regime.value}")

        # Generate full report
        report = generate_market_report(portfolio_prices)

        print(f"\n{'Technical Indicators':^100}")
        print("-" * 100)
        print(f"  RSI:                {report['technical_indicators']['rsi']} ({report['technical_indicators']['rsi_signal']})")
        print(f"  Bollinger Position: {report['technical_indicators']['bollinger_position']:.1f}%")
        print(f"  MACD Signal:        {report['technical_indicators']['macd_signal']}")

        if report['recommendations']:
            print(f"\n{'Recommendations':^100}")
            print("-" * 100)
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")

        return report

    except Exception as e:
        print(f"\n✗ Error during market analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_monte_carlo_simulation(db_path: str, num_simulations: int = 10000):
    """Run Monte Carlo simulation"""
    print_section_header("STEP 4: MONTE CARLO SIMULATION")

    try:
        # Load data
        backtester = EnhancedMutualFundBacktester(db_path=db_path)
        portfolio_prices = backtester.data['prices'].mean(axis=1)
        returns = portfolio_prices.pct_change().dropna()

        # Run simulation
        mc = MonteCarloSimulation(num_simulations=num_simulations)
        results = mc.run_simulation(
            initial_value=100,
            historical_returns=returns,
            num_days=252  # 1 year
        )

        print(f"\n{'Probability Distribution':^100}")
        print("-" * 100)
        print(f"  Expected Return:        {results['statistics']['mean_return']:>8.2f}%")
        print(f"  Median Return:          {results['statistics']['median_return']:>8.2f}%")
        print(f"  Best Case (95%):        {results['statistics']['percentiles'][95]:>8.2f}")
        print(f"  Worst Case (5%):        {results['statistics']['percentiles'][5]:>8.2f}")
        print(f"  Probability of Loss:    {results['statistics']['prob_loss']:>8.1f}%")
        print(f"  Prob of >10% Return:    {results['statistics']['prob_beat_target']:>8.1f}%")

        print(f"\n{'Risk Assessment':^100}")
        print("-" * 100)
        if results['statistics']['prob_loss'] < 10:
            print("  ✓ LOW RISK: Very low probability of loss")
        elif results['statistics']['prob_loss'] < 25:
            print("  ⚠ MODERATE RISK: Some downside risk present")
        else:
            print("  ⚠ HIGH RISK: Significant probability of loss")

        return results

    except Exception as e:
        print(f"\n✗ Error during Monte Carlo: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_rolling_analysis(db_path: str):
    """Run rolling performance analysis"""
    print_section_header("STEP 5: ROLLING PERFORMANCE ANALYSIS")

    try:
        # Load data
        backtester = EnhancedMutualFundBacktester(db_path=db_path)
        portfolio_prices = backtester.data['prices'].mean(axis=1)

        # Calculate rolling metrics
        rolling_metrics = RollingAnalysis.calculate_rolling_metrics(portfolio_prices, window=252)
        consistency = RollingAnalysis.analyze_consistency(rolling_metrics)

        print(f"\n{'Consistency Metrics (1-Year Rolling)':^100}")
        print("-" * 100)
        print(f"  Average Rolling Return:     {consistency['avg_rolling_return']:>8.2f}%")
        print(f"  Std Dev of Returns:         {consistency['std_rolling_return']:>8.2f}%")
        print(f"  % Positive Periods:         {consistency['pct_positive_periods']:>8.1f}%")
        print(f"  Average Sharpe Ratio:       {consistency['avg_sharpe']:>8.2f}")
        print(f"  Sharpe Stability (Std):     {consistency['sharpe_stability']:>8.2f}")
        print(f"  Worst Rolling Drawdown:     {consistency['worst_rolling_dd']:>8.2f}%")
        print(f"  Average Rolling Drawdown:   {consistency['avg_rolling_dd']:>8.2f}%")

        # Assess consistency
        print(f"\n{'Consistency Assessment':^100}")
        print("-" * 100)
        if consistency['pct_positive_periods'] > 75:
            print("  ✓ HIGHLY CONSISTENT: >75% positive periods")
        elif consistency['pct_positive_periods'] > 60:
            print("  ✓ CONSISTENT: >60% positive periods")
        else:
            print("  ⚠ INCONSISTENT: <60% positive periods")

        return consistency

    except Exception as e:
        print(f"\n✗ Error during rolling analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def generate_final_report(all_results: dict):
    """Generate comprehensive final report"""
    print_section_header("FINAL REPORT & RECOMMENDATIONS")

    print(f"\n{'Executive Summary':^100}")
    print("-" * 100)

    # Extract key metrics from results
    if 'strategies' in all_results and all_results['strategies']:
        strategies = all_results['strategies']

        # Find best strategy
        best_strategy = max(strategies.items(), key=lambda x: x[1].get('sharpe_ratio', 0))
        strategy_name, metrics = best_strategy

        print(f"\n  Recommended Strategy: {strategy_name}")
        print(f"    Annual Return:     {metrics.get('net_annualized_return', 0):.2f}%")
        print(f"    Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"    Max Drawdown:      {metrics.get('max_drawdown', 0):.2f}%")
        print(f"    Win Rate:          {metrics.get('win_rate', 0):.2f}%")

    if 'monte_carlo' in all_results and all_results['monte_carlo']:
        mc = all_results['monte_carlo']['statistics']
        print(f"\n  Expected 1-Year Outcome (Monte Carlo):")
        print(f"    Expected Return:   {mc['mean_return']:.2f}%")
        print(f"    Risk of Loss:      {mc['prob_loss']:.1f}%")

    if 'market' in all_results and all_results['market']:
        market = all_results['market']
        print(f"\n  Current Market Conditions:")
        print(f"    Regime:            {market['current_regime']}")
        if market['recommendations']:
            print(f"    Key Actions:       {market['recommendations'][0]}")

    # Overall recommendations
    print(f"\n{'Key Recommendations':^100}")
    print("-" * 100)
    print("  1. Portfolio Allocation:")
    print("     • For Conservative Investors: Use Conservative or Balanced strategy")
    print("     • For Moderate Risk:          Use Balanced or Risk Parity strategy")
    print("     • For Aggressive Growth:      Use Aggressive or Momentum strategy")
    print("")
    print("  2. Risk Management:")
    print("     • Implement 15% portfolio stop-loss")
    print("     • Review positions monthly during high volatility")
    print("     • Maintain emergency cash buffer (10-20%)")
    print("")
    print("  3. Monitoring:")
    print("     • Track drawdowns weekly")
    print("     • Rebalance quarterly (or monthly during volatility)")
    print("     • Review strategy annually")
    print("")
    print("  4. Next Steps:")
    print("     • Open reports/backtest_*.html for detailed charts")
    print("     • Review individual fund performance")
    print("     • Consider tax implications before rebalancing")

    # Save summary
    print(f"\n{'Generated Files':^100}")
    print("-" * 100)
    print("  ✓ reports/enhanced_backtest_summary.txt")
    print("  ✓ reports/strategy_comparison.csv")
    print("  ✓ reports/monte_carlo_results.json")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Run enhanced backtesting with all improvements'
    )
    parser.add_argument('--db-path', type=str, default='data/historical_20y.db',
                        help='Path to database')
    parser.add_argument('--num-funds', type=int, default=10,
                        help='Number of funds to include')
    parser.add_argument('--timeframe-years', type=int, default=5,
                        help='Backtesting timeframe in years')
    parser.add_argument('--monte-carlo', action='store_true',
                        help='Run Monte Carlo simulation')
    parser.add_argument('--simulations', type=int, default=10000,
                        help='Number of Monte Carlo simulations')
    parser.add_argument('--skip-quality', action='store_true',
                        help='Skip data quality check')
    parser.add_argument('--skip-market', action='store_true',
                        help='Skip market analysis')

    args = parser.parse_args()

    print("=" * 100)
    print(f"{'ENHANCED MUTUAL FUND BACKTESTING SYSTEM':^100}")
    print("=" * 100)
    print(f"  Started:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Database:       {args.db_path}")
    print(f"  Funds:          {args.num_funds}")
    print(f"  Timeframe:      {args.timeframe_years} years")
    print(f"  Monte Carlo:    {'Yes' if args.monte_carlo else 'No'}")
    print("=" * 100)

    # Check database exists
    if not Path(args.db_path).exists():
        print(f"\n✗ ERROR: Database not found at {args.db_path}")
        return 1

    # Create reports directory
    Path('reports').mkdir(exist_ok=True)

    all_results = {}

    try:
        # Step 1: Data Quality
        if not args.skip_quality:
            quality_report = run_data_quality_check(args.db_path)
            all_results['quality'] = quality_report

        # Step 2: Enhanced Backtesting
        strategy_results = run_enhanced_backtesting(
            args.db_path,
            num_funds=args.num_funds,
            timeframe_years=args.timeframe_years
        )
        all_results['strategies'] = strategy_results

        # Step 3: Market Analysis
        if not args.skip_market:
            market_report = run_market_analysis(args.db_path)
            all_results['market'] = market_report

        # Step 4: Monte Carlo Simulation
        if args.monte_carlo:
            mc_results = run_monte_carlo_simulation(args.db_path, args.simulations)
            all_results['monte_carlo'] = mc_results

        # Step 5: Rolling Analysis
        rolling_results = run_rolling_analysis(args.db_path)
        all_results['rolling'] = rolling_results

        # Final Report
        generate_final_report(all_results)

        print("\n" + "=" * 100)
        print(f"{'EXECUTION COMPLETE':^100}")
        print("=" * 100)
        print(f"  Completed:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("  Status:         ✓ SUCCESS")
        print("=" * 100)

        return 0

    except Exception as e:
        print(f"\n\n✗ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
