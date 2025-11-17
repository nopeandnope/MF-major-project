"""
Market Analysis Module

Implements:
- Market regime detection (Bull/Bear/Sideways)
- Benchmark comparison
- Market indicators analysis
- Economic cycle detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum


class MarketRegime(Enum):
    """Market regime classification"""
    BULL = "Bull Market"
    BEAR = "Bear Market"
    SIDEWAYS = "Sideways Market"
    HIGH_VOLATILITY = "High Volatility"
    RECOVERY = "Recovery Phase"


class MarketRegimeDetector:
    """
    Detect market regimes using multiple indicators
    """

    def __init__(
        self,
        ma_short: int = 50,
        ma_long: int = 200,
        vol_threshold: float = 20.0
    ):
        """
        Args:
            ma_short: Short-term moving average period
            ma_long: Long-term moving average period
            vol_threshold: Volatility threshold for high-vol regime
        """
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.vol_threshold = vol_threshold

    def detect_regime(
        self,
        prices: pd.Series,
        returns: pd.Series = None
    ) -> MarketRegime:
        """
        Detect current market regime.

        Args:
            prices: Price series (e.g., Nifty 50 or portfolio)
            returns: Optional return series

        Returns:
            MarketRegime enum
        """
        if returns is None:
            returns = prices.pct_change()

        # Calculate moving averages
        ma_short = prices.rolling(window=self.ma_short).mean()
        ma_long = prices.rolling(window=self.ma_long).mean()

        # Calculate volatility
        volatility = returns.rolling(window=60).std() * np.sqrt(252) * 100

        # Get recent values
        current_price = prices.iloc[-1]
        current_ma_short = ma_short.iloc[-1]
        current_ma_long = ma_long.iloc[-1]
        current_vol = volatility.iloc[-1]

        # Recent trend (3-month return)
        if len(returns) >= 60:
            recent_return = ((prices.iloc[-1] / prices.iloc[-60]) - 1) * 100
        else:
            recent_return = 0

        # Regime detection logic
        if current_vol > self.vol_threshold:
            return MarketRegime.HIGH_VOLATILITY

        # Bull market: Price above both MAs, short MA > long MA
        if (current_price > current_ma_short and
            current_price > current_ma_long and
            current_ma_short > current_ma_long):
            return MarketRegime.BULL

        # Bear market: Price below both MAs, short MA < long MA
        if (current_price < current_ma_short and
            current_price < current_ma_long and
            current_ma_short < current_ma_long):
            return MarketRegime.BEAR

        # Recovery: Price crossed above MA recently
        if len(ma_short) >= 10:
            prev_cross = ma_short.iloc[-10] < ma_long.iloc[-10]
            curr_cross = ma_short.iloc[-1] > ma_long.iloc[-1]
            if prev_cross and curr_cross and recent_return > 5:
                return MarketRegime.RECOVERY

        # Default: Sideways
        return MarketRegime.SIDEWAYS

    def get_regime_history(
        self,
        prices: pd.Series,
        window: int = 252
    ) -> pd.Series:
        """
        Get historical regime classification.

        Args:
            prices: Price series
            window: Lookback window for regime detection

        Returns:
            Series with regime for each date
        """
        regimes = []
        dates = []

        for i in range(window, len(prices)):
            subset = prices.iloc[:i+1]
            regime = self.detect_regime(subset)
            regimes.append(regime.value)
            dates.append(prices.index[i])

        return pd.Series(regimes, index=dates)


class BenchmarkComparator:
    """
    Compare portfolio performance against benchmarks
    """

    def __init__(self, benchmark_prices: Dict[str, pd.Series]):
        """
        Args:
            benchmark_prices: Dictionary of benchmark name -> price series
                Example: {'Nifty50': nifty_prices, 'Nifty500': nifty500_prices}
        """
        self.benchmark_prices = benchmark_prices

    def calculate_alpha_beta(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """
        Calculate alpha and beta vs benchmark.

        Returns:
            (alpha, beta) tuple
        """
        # Align series
        aligned = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if len(aligned) < 30:
            return 0.0, 1.0

        # Calculate beta (covariance / variance)
        covariance = aligned['portfolio'].cov(aligned['benchmark'])
        variance = aligned['benchmark'].var()
        beta = covariance / variance if variance != 0 else 1.0

        # Calculate alpha (annualized)
        portfolio_return = aligned['portfolio'].mean() * 252
        benchmark_return = aligned['benchmark'].mean() * 252
        alpha = portfolio_return - (beta * benchmark_return)

        return alpha * 100, beta

    def calculate_tracking_error(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate tracking error (annualized std of excess returns).
        """
        aligned = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if len(aligned) < 30:
            return 0.0

        excess_returns = aligned['portfolio'] - aligned['benchmark']
        tracking_error = excess_returns.std() * np.sqrt(252) * 100

        return tracking_error

    def calculate_information_ratio(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate information ratio (alpha / tracking error).
        """
        alpha, beta = self.calculate_alpha_beta(portfolio_returns, benchmark_returns)
        tracking_error = self.calculate_tracking_error(portfolio_returns, benchmark_returns)

        if tracking_error == 0:
            return 0.0

        return alpha / tracking_error

    def compare_performance(
        self,
        portfolio_prices: pd.Series,
        benchmark_name: str = None
    ) -> Dict[str, any]:
        """
        Comprehensive comparison vs benchmark(s).

        Args:
            portfolio_prices: Portfolio price series
            benchmark_name: Specific benchmark to compare (None = all)

        Returns:
            Dictionary with comparison metrics
        """
        portfolio_returns = portfolio_prices.pct_change().dropna()

        comparisons = {}

        benchmarks_to_compare = (
            {benchmark_name: self.benchmark_prices[benchmark_name]}
            if benchmark_name
            else self.benchmark_prices
        )

        for bench_name, bench_prices in benchmarks_to_compare.items():
            # Align dates
            common_dates = portfolio_prices.index.intersection(bench_prices.index)

            if len(common_dates) < 30:
                continue

            port_aligned = portfolio_prices.loc[common_dates]
            bench_aligned = bench_prices.loc[common_dates]

            port_returns = port_aligned.pct_change().dropna()
            bench_returns = bench_aligned.pct_change().dropna()

            # Calculate metrics
            alpha, beta = self.calculate_alpha_beta(port_returns, bench_returns)
            tracking_error = self.calculate_tracking_error(port_returns, bench_returns)
            info_ratio = self.calculate_information_ratio(port_returns, bench_returns)

            # Total returns
            port_total_return = ((port_aligned.iloc[-1] / port_aligned.iloc[0]) - 1) * 100
            bench_total_return = ((bench_aligned.iloc[-1] / bench_aligned.iloc[0]) - 1) * 100
            excess_return = port_total_return - bench_total_return

            # Annualized
            years = (common_dates[-1] - common_dates[0]).days / 365.25
            port_annual = ((1 + port_total_return/100) ** (1/years) - 1) * 100
            bench_annual = ((1 + bench_total_return/100) ** (1/years) - 1) * 100

            # Win rate (days portfolio beats benchmark)
            win_days = (port_returns > bench_returns).sum()
            win_rate = (win_days / len(port_returns)) * 100

            comparisons[bench_name] = {
                'alpha': round(alpha, 2),
                'beta': round(beta, 2),
                'tracking_error': round(tracking_error, 2),
                'information_ratio': round(info_ratio, 2),
                'portfolio_total_return': round(port_total_return, 2),
                'benchmark_total_return': round(bench_total_return, 2),
                'excess_return': round(excess_return, 2),
                'portfolio_annual_return': round(port_annual, 2),
                'benchmark_annual_return': round(bench_annual, 2),
                'win_rate': round(win_rate, 2),
                'days_compared': len(common_dates)
            }

        return comparisons


class MarketIndicators:
    """
    Calculate various market indicators
    """

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.

        Returns:
            RSI series (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Returns:
            (upper_band, middle_band, lower_band)
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    @staticmethod
    def calculate_macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD indicator.

        Returns:
            (macd_line, signal_line, histogram)
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def detect_support_resistance(
        prices: pd.Series,
        window: int = 20,
        threshold_pct: float = 2.0
    ) -> Dict[str, List[float]]:
        """
        Detect support and resistance levels.

        Args:
            prices: Price series
            window: Window for local min/max
            threshold_pct: Percentage threshold for level clustering

        Returns:
            Dictionary with support and resistance levels
        """
        # Find local minima (support)
        local_min = prices.rolling(window=window, center=True).min() == prices
        support_levels = prices[local_min].tolist()

        # Find local maxima (resistance)
        local_max = prices.rolling(window=window, center=True).max() == prices
        resistance_levels = prices[local_max].tolist()

        # Cluster nearby levels
        def cluster_levels(levels, threshold_pct):
            if not levels:
                return []
            levels = sorted(levels)
            clusters = [[levels[0]]]

            for level in levels[1:]:
                if abs(level - np.mean(clusters[-1])) / np.mean(clusters[-1]) * 100 <= threshold_pct:
                    clusters[-1].append(level)
                else:
                    clusters.append([level])

            return [np.mean(cluster) for cluster in clusters]

        support_clustered = cluster_levels(support_levels, threshold_pct)
        resistance_clustered = cluster_levels(resistance_levels, threshold_pct)

        return {
            'support_levels': support_clustered[-5:],  # Last 5 support levels
            'resistance_levels': resistance_clustered[-5:]  # Last 5 resistance levels
        }


def generate_market_report(
    portfolio_prices: pd.Series,
    benchmark_prices: Dict[str, pd.Series] = None
) -> Dict[str, any]:
    """
    Generate comprehensive market analysis report.

    Args:
        portfolio_prices: Portfolio price series
        benchmark_prices: Dictionary of benchmark prices

    Returns:
        Comprehensive market report
    """
    # Regime detection
    detector = MarketRegimeDetector()
    current_regime = detector.detect_regime(portfolio_prices)

    # Technical indicators
    rsi = MarketIndicators.calculate_rsi(portfolio_prices).iloc[-1]
    upper_bb, middle_bb, lower_bb = MarketIndicators.calculate_bollinger_bands(portfolio_prices)
    macd, signal, hist = MarketIndicators.calculate_macd(portfolio_prices)

    # Current position in Bollinger Bands
    current_price = portfolio_prices.iloc[-1]
    bb_position = ((current_price - lower_bb.iloc[-1]) /
                   (upper_bb.iloc[-1] - lower_bb.iloc[-1])) * 100 if len(upper_bb) > 0 else 50

    # Support/Resistance
    levels = MarketIndicators.detect_support_resistance(portfolio_prices)

    report = {
        'timestamp': datetime.now(),
        'current_regime': current_regime.value,
        'technical_indicators': {
            'rsi': round(rsi, 2) if not pd.isna(rsi) else None,
            'rsi_signal': 'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral',
            'bollinger_position': round(bb_position, 1),
            'macd_histogram': round(hist.iloc[-1], 2) if len(hist) > 0 else None,
            'macd_signal': 'Bullish' if hist.iloc[-1] > 0 else 'Bearish'
        },
        'support_resistance': levels,
        'recommendations': []
    }

    # Generate recommendations
    if current_regime == MarketRegime.BEAR:
        report['recommendations'].append("Reduce equity exposure, shift to defensive")
    elif current_regime == MarketRegime.BULL:
        report['recommendations'].append("Good environment for equity, maintain allocation")
    elif current_regime == MarketRegime.HIGH_VOLATILITY:
        report['recommendations'].append("High volatility detected, consider hedging")

    if rsi > 70:
        report['recommendations'].append("RSI overbought, consider taking profits")
    elif rsi < 30:
        report['recommendations'].append("RSI oversold, potential buying opportunity")

    # Benchmark comparison
    if benchmark_prices:
        comparator = BenchmarkComparator(benchmark_prices)
        report['benchmark_comparison'] = comparator.compare_performance(portfolio_prices)

    return report


if __name__ == "__main__":
    print("Market Analysis Module")
    print("=" * 80)

    # Generate sample data
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)

    # Simulate portfolio and benchmark
    port_returns = np.random.normal(0.0006, 0.015, len(dates))
    bench_returns = np.random.normal(0.0004, 0.012, len(dates))

    portfolio_prices = pd.Series((1 + port_returns).cumprod() * 100, index=dates)
    benchmark_prices = pd.Series((1 + bench_returns).cumprod() * 100, index=dates)

    # Market regime detection
    detector = MarketRegimeDetector()
    regime = detector.detect_regime(portfolio_prices)
    print(f"\nCurrent Market Regime: {regime.value}")

    # Benchmark comparison
    comparator = BenchmarkComparator({'Benchmark': benchmark_prices})
    comparison = comparator.compare_performance(portfolio_prices)

    print("\nBenchmark Comparison:")
    for key, value in comparison['Benchmark'].items():
        print(f"  {key}: {value}")

    # Full report
    report = generate_market_report(
        portfolio_prices,
        {'Benchmark': benchmark_prices}
    )

    print("\nMarket Report Generated:")
    print(f"  Regime: {report['current_regime']}")
    print(f"  RSI: {report['technical_indicators']['rsi']} ({report['technical_indicators']['rsi_signal']})")
    print(f"  Recommendations: {len(report['recommendations'])}")
