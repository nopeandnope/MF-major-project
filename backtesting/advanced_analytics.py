"""
Advanced Analytics Module

Implements:
- Monte Carlo simulation
- Walk-forward optimization
- Rolling performance analysis
- Scenario analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class MonteCarloSimulation:
    """
    Monte Carlo simulation for portfolio outcomes
    """

    def __init__(self, num_simulations: int = 10000):
        """
        Args:
            num_simulations: Number of simulation runs
        """
        self.num_simulations = num_simulations

    def simulate_returns(
        self,
        historical_returns: pd.Series,
        num_days: int = 252
    ) -> np.ndarray:
        """
        Simulate future returns based on historical distribution.

        Args:
            historical_returns: Historical return series
            num_days: Number of days to simulate

        Returns:
            Array of shape (num_simulations, num_days)
        """
        mean_return = historical_returns.mean()
        std_return = historical_returns.std()

        # Generate random returns
        simulated = np.random.normal(
            mean_return,
            std_return,
            size=(self.num_simulations, num_days)
        )

        return simulated

    def run_simulation(
        self,
        initial_value: float,
        historical_returns: pd.Series,
        num_days: int = 252
    ) -> Dict[str, any]:
        """
        Run complete Monte Carlo simulation.

        Args:
            initial_value: Starting portfolio value
            historical_returns: Historical returns
            num_days: Days to simulate (252 = 1 year)

        Returns:
            Simulation results with statistics
        """
        print(f"Running {self.num_simulations:,} Monte Carlo simulations for {num_days} days...")

        # Simulate returns
        simulated_returns = self.simulate_returns(historical_returns, num_days)

        # Calculate portfolio paths
        portfolio_paths = initial_value * np.cumprod(1 + simulated_returns, axis=1)

        # Final values
        final_values = portfolio_paths[:, -1]

        # Calculate statistics
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        percentile_values = np.percentile(final_values, percentiles)

        # Returns
        final_returns = (final_values / initial_value - 1) * 100

        # Probability of loss
        prob_loss = (final_returns < 0).sum() / self.num_simulations * 100

        # Probability of beating target (e.g., 10% annual return)
        target_return = 10.0
        prob_beat_target = (final_returns > target_return).sum() / self.num_simulations * 100

        results = {
            'initial_value': initial_value,
            'num_simulations': self.num_simulations,
            'num_days': num_days,
            'portfolio_paths': portfolio_paths,
            'final_values': final_values,
            'final_returns': final_returns,
            'statistics': {
                'mean_final_value': np.mean(final_values),
                'median_final_value': np.median(final_values),
                'std_final_value': np.std(final_values),
                'mean_return': np.mean(final_returns),
                'median_return': np.median(final_returns),
                'percentiles': dict(zip(percentiles, percentile_values)),
                'prob_loss': prob_loss,
                'prob_beat_target': prob_beat_target,
                'best_case': final_returns.max(),
                'worst_case': final_returns.min()
            }
        }

        print(f"\nSimulation Complete!")
        print(f"  Expected Return: {results['statistics']['mean_return']:.2f}%")
        print(f"  Median Return: {results['statistics']['median_return']:.2f}%")
        print(f"  Probability of Loss: {prob_loss:.1f}%")
        print(f"  Probability of >10% Return: {prob_beat_target:.1f}%")
        print(f"  95th Percentile: {percentile_values[-1]:.2f}")
        print(f"  5th Percentile: {percentile_values[0]:.2f}")

        return results


class RollingAnalysis:
    """
    Rolling window performance analysis
    """

    @staticmethod
    def calculate_rolling_metrics(
        prices: pd.Series,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.

        Args:
            prices: Price series
            window: Rolling window size (252 = 1 year)

        Returns:
            DataFrame with rolling metrics
        """
        returns = prices.pct_change().fillna(0)

        rolling_returns = []
        rolling_sharpe = []
        rolling_vol = []
        rolling_max_dd = []
        dates = []

        for i in range(window, len(prices)):
            subset_prices = prices.iloc[i-window:i+1]
            subset_returns = returns.iloc[i-window:i+1]

            # Rolling return
            period_return = (subset_prices.iloc[-1] / subset_prices.iloc[0] - 1) * 100
            rolling_returns.append(period_return)

            # Rolling volatility
            vol = subset_returns.std() * np.sqrt(252) * 100
            rolling_vol.append(vol)

            # Rolling Sharpe
            annual_return = period_return
            sharpe = (annual_return - 6.5) / vol if vol > 0 else 0
            rolling_sharpe.append(sharpe)

            # Rolling max drawdown
            cum_returns = (1 + subset_returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_dd = drawdown.min() * 100
            rolling_max_dd.append(max_dd)

            dates.append(prices.index[i])

        results = pd.DataFrame({
            'rolling_return': rolling_returns,
            'rolling_sharpe': rolling_sharpe,
            'rolling_volatility': rolling_vol,
            'rolling_max_dd': rolling_max_dd
        }, index=dates)

        return results

    @staticmethod
    def analyze_consistency(
        rolling_metrics: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Analyze consistency of performance.

        Args:
            rolling_metrics: DataFrame from calculate_rolling_metrics

        Returns:
            Consistency metrics
        """
        return {
            'avg_rolling_return': rolling_metrics['rolling_return'].mean(),
            'std_rolling_return': rolling_metrics['rolling_return'].std(),
            'pct_positive_periods': (rolling_metrics['rolling_return'] > 0).sum() / len(rolling_metrics) * 100,
            'avg_sharpe': rolling_metrics['rolling_sharpe'].mean(),
            'sharpe_stability': rolling_metrics['rolling_sharpe'].std(),
            'worst_rolling_dd': rolling_metrics['rolling_max_dd'].min(),
            'avg_rolling_dd': rolling_metrics['rolling_max_dd'].mean()
        }


class WalkForwardOptimizer:
    """
    Walk-forward optimization to avoid overfitting
    """

    def __init__(
        self,
        train_window: int = 252 * 3,  # 3 years
        test_window: int = 252,       # 1 year
        step_size: int = 126           # 6 months
    ):
        """
        Args:
            train_window: Training window size
            test_window: Testing window size
            step_size: Step size for walk-forward
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

    def split_data(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data into walk-forward windows.

        Returns:
            List of (train_data, test_data) tuples
        """
        splits = []
        total_length = len(data)

        start_idx = 0
        while start_idx + self.train_window + self.test_window <= total_length:
            train_end = start_idx + self.train_window
            test_end = train_end + self.test_window

            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[train_end:test_end]

            splits.append((train_data, test_data))

            start_idx += self.step_size

        return splits

    def optimize_parameters(
        self,
        train_data: pd.DataFrame,
        parameter_grid: Dict[str, List]
    ) -> Dict[str, any]:
        """
        Optimize strategy parameters on training data.

        Args:
            train_data: Training data
            parameter_grid: Grid of parameters to test

        Returns:
            Best parameters
        """
        # Placeholder for optimization logic
        # In practice, would test different parameter combinations
        # and select best based on Sharpe ratio or other metric

        best_params = {key: values[0] for key, values in parameter_grid.items()}
        return best_params

    def backtest_with_parameters(
        self,
        test_data: pd.DataFrame,
        parameters: Dict[str, any]
    ) -> Dict[str, float]:
        """
        Backtest strategy on test data with given parameters.

        Args:
            test_data: Test data
            parameters: Strategy parameters

        Returns:
            Performance metrics
        """
        # Placeholder for backtesting logic
        # In practice, would run strategy with parameters
        # and calculate performance

        returns = test_data.pct_change().mean(axis=1)
        total_return = (1 + returns).prod() - 1

        return {
            'return': total_return * 100,
            'sharpe': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        }

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        parameter_grid: Dict[str, List] = None
    ) -> Dict[str, any]:
        """
        Run complete walk-forward optimization.

        Args:
            data: Full dataset
            parameter_grid: Parameters to optimize

        Returns:
            Walk-forward results
        """
        if parameter_grid is None:
            parameter_grid = {
                'rebalance_freq': ['monthly', 'quarterly'],
                'num_funds': [5, 10, 15]
            }

        print(f"\nRunning Walk-Forward Optimization:")
        print(f"  Train Window: {self.train_window} days")
        print(f"  Test Window: {self.test_window} days")
        print(f"  Step Size: {self.step_size} days")

        splits = self.split_data(data)
        print(f"  Number of Windows: {len(splits)}")

        results = []

        for i, (train, test) in enumerate(splits, 1):
            print(f"\n  Window {i}/{len(splits)}")
            print(f"    Train: {train.index[0].date()} to {train.index[-1].date()}")
            print(f"    Test:  {test.index[0].date()} to {test.index[-1].date()}")

            # Optimize on train
            best_params = self.optimize_parameters(train, parameter_grid)

            # Test on test
            test_results = self.backtest_with_parameters(test, best_params)

            results.append({
                'window': i,
                'train_start': train.index[0],
                'train_end': train.index[-1],
                'test_start': test.index[0],
                'test_end': test.index[-1],
                'parameters': best_params,
                'test_return': test_results['return'],
                'test_sharpe': test_results['sharpe']
            })

        # Aggregate results
        avg_return = np.mean([r['test_return'] for r in results])
        avg_sharpe = np.mean([r['test_sharpe'] for r in results])
        consistency = np.std([r['test_return'] for r in results])

        print(f"\n  Walk-Forward Results:")
        print(f"    Average Test Return: {avg_return:.2f}%")
        print(f"    Average Test Sharpe: {avg_sharpe:.2f}")
        print(f"    Consistency (Std): {consistency:.2f}")

        return {
            'windows': results,
            'average_return': avg_return,
            'average_sharpe': avg_sharpe,
            'consistency': consistency
        }


if __name__ == "__main__":
    print("Advanced Analytics Module")
    print("=" * 80)

    # Generate sample data
    dates = pd.date_range(start='2015-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.0006, 0.015, len(dates))
    prices = pd.Series((1 + returns).cumprod() * 100, index=dates)

    # Monte Carlo
    print("\n" + "=" * 80)
    print("MONTE CARLO SIMULATION")
    print("=" * 80)
    mc = MonteCarloSimulation(num_simulations=1000)
    mc_results = mc.run_simulation(
        initial_value=100,
        historical_returns=prices.pct_change().dropna(),
        num_days=252
    )

    # Rolling Analysis
    print("\n" + "=" * 80)
    print("ROLLING ANALYSIS")
    print("=" * 80)
    rolling_metrics = RollingAnalysis.calculate_rolling_metrics(prices, window=252)
    consistency = RollingAnalysis.analyze_consistency(rolling_metrics)

    print("\nConsistency Metrics:")
    for key, value in consistency.items():
        print(f"  {key}: {value:.2f}")

    # Walk-Forward
    print("\n" + "=" * 80)
    print("WALK-FORWARD OPTIMIZATION")
    print("=" * 80)
    multi_asset_data = pd.DataFrame({
        'fund1': prices,
        'fund2': prices * 0.95,
        'fund3': prices * 1.05
    })

    wf = WalkForwardOptimizer(train_window=252*2, test_window=252, step_size=126)
    wf_results = wf.run_walk_forward(multi_asset_data)
