"""
Risk Management Module

Implements:
- Stop-loss mechanisms
- Position sizing
- Drawdown controls
- Volatility targeting
- Risk parity
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


class RiskManager:
    """
    Comprehensive risk management system for portfolio
    """

    def __init__(
        self,
        stop_loss_pct: float = 15.0,
        max_drawdown_limit: float = 20.0,
        target_volatility: float = 15.0,
        position_size_method: str = 'equal_weight'
    ):
        """
        Initialize risk manager.

        Args:
            stop_loss_pct: Portfolio stop-loss percentage
            max_drawdown_limit: Maximum acceptable drawdown
            target_volatility: Target annual volatility
            position_size_method: 'equal_weight', 'risk_parity', 'volatility_target'
        """
        self.stop_loss_pct = stop_loss_pct
        self.max_drawdown_limit = max_drawdown_limit
        self.target_volatility = target_volatility
        self.position_size_method = position_size_method

        # Tracking
        self.peak_value = None
        self.stop_loss_triggered = False
        self.drawdown_alerts = []

    def check_stop_loss(
        self,
        current_value: float,
        peak_value: float
    ) -> Tuple[bool, float]:
        """
        Check if stop-loss is triggered.

        Returns:
            (is_triggered, current_drawdown_pct)
        """
        drawdown = ((current_value - peak_value) / peak_value) * 100

        if drawdown <= -self.stop_loss_pct:
            return True, drawdown

        return False, drawdown

    def calculate_position_sizes(
        self,
        prices: pd.DataFrame,
        method: str = None
    ) -> pd.Series:
        """
        Calculate position sizes for each fund.

        Args:
            prices: DataFrame of fund prices
            method: Sizing method override

        Returns:
            Series of position weights
        """
        method = method or self.position_size_method

        if method == 'equal_weight':
            return self._equal_weight(prices)
        elif method == 'risk_parity':
            return self._risk_parity(prices)
        elif method == 'volatility_target':
            return self._volatility_target(prices)
        elif method == 'min_variance':
            return self._min_variance(prices)
        else:
            return self._equal_weight(prices)

    def _equal_weight(self, prices: pd.DataFrame) -> pd.Series:
        """Equal weight allocation"""
        n_funds = len(prices.columns)
        return pd.Series({col: 1.0 / n_funds for col in prices.columns})

    def _risk_parity(self, prices: pd.DataFrame, lookback: int = 60) -> pd.Series:
        """
        Risk parity: Weight inversely to volatility.
        """
        returns = prices.pct_change().iloc[-lookback:]
        volatilities = returns.std()

        # Inverse volatility
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()

        return weights

    def _volatility_target(self, prices: pd.DataFrame, lookback: int = 60) -> pd.Series:
        """
        Target volatility: Scale positions to hit target volatility.
        """
        returns = prices.pct_change().iloc[-lookback:]

        # Portfolio volatility with equal weights
        equal_weights = pd.Series([1.0/len(prices.columns)] * len(prices.columns),
                                   index=prices.columns)
        portfolio_returns = (returns * equal_weights).sum(axis=1)
        current_vol = portfolio_returns.std() * np.sqrt(252) * 100

        # Scale factor to hit target
        scale_factor = self.target_volatility / current_vol if current_vol > 0 else 1.0

        # Apply scale (cap at 1.5x leverage)
        scale_factor = min(scale_factor, 1.5)

        weights = equal_weights * scale_factor
        weights = weights / weights.sum()  # Normalize

        return weights

    def _min_variance(self, prices: pd.DataFrame, lookback: int = 60) -> pd.Series:
        """
        Minimum variance portfolio.
        """
        returns = prices.pct_change().iloc[-lookback:]

        # Covariance matrix
        cov_matrix = returns.cov()

        # Minimum variance weights (inverse of covariance matrix sum)
        try:
            inv_cov = np.linalg.inv(cov_matrix.values)
            ones = np.ones(len(cov_matrix))
            weights = inv_cov @ ones
            weights = weights / weights.sum()
            weights = pd.Series(weights, index=cov_matrix.index)
            weights = weights.clip(lower=0)  # No short selling
            weights = weights / weights.sum()  # Re-normalize
        except:
            # Fallback to equal weight if matrix is singular
            weights = self._equal_weight(prices)

        return weights

    def calculate_portfolio_metrics(
        self,
        prices: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate real-time portfolio metrics.

        Returns:
            Dictionary with metrics
        """
        returns = prices.pct_change().fillna(0)

        # Current drawdown
        peak = prices.expanding().max().iloc[-1]
        current_drawdown = ((prices.iloc[-1] - peak) / peak) * 100

        # Volatility (last 60 days)
        recent_vol = returns.iloc[-60:].std() * np.sqrt(252) * 100

        # Sharpe (last 252 days)
        if len(returns) >= 252:
            recent_returns = returns.iloc[-252:]
            annual_return = recent_returns.mean() * 252 * 100
            annual_vol = recent_returns.std() * np.sqrt(252) * 100
            sharpe = (annual_return - 6.5) / annual_vol if annual_vol > 0 else 0
        else:
            sharpe = 0

        return {
            'current_value': prices.iloc[-1],
            'peak_value': peak,
            'current_drawdown': current_drawdown,
            'recent_volatility': recent_vol,
            'recent_sharpe': sharpe
        }

    def assess_risk_level(
        self,
        portfolio_metrics: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Assess current risk level and recommend actions.

        Returns:
            Dictionary with risk assessment
        """
        drawdown = portfolio_metrics['current_drawdown']
        volatility = portfolio_metrics['recent_volatility']

        risk_level = 'LOW'
        action = 'HOLD'
        reason = []

        # Drawdown assessment
        if drawdown <= -self.stop_loss_pct:
            risk_level = 'CRITICAL'
            action = 'SELL_ALL'
            reason.append(f'Stop-loss triggered at {drawdown:.1f}%')
        elif drawdown <= -self.max_drawdown_limit * 0.75:
            risk_level = 'HIGH'
            action = 'REDUCE_EXPOSURE'
            reason.append(f'Approaching max drawdown limit ({drawdown:.1f}%)')
        elif drawdown <= -10:
            risk_level = 'MODERATE'
            action = 'MONITOR'
            reason.append(f'Moderate drawdown ({drawdown:.1f}%)')

        # Volatility assessment
        if volatility > self.target_volatility * 1.5:
            if risk_level == 'LOW':
                risk_level = 'MODERATE'
            action = 'REDUCE_EXPOSURE'
            reason.append(f'High volatility ({volatility:.1f}% vs target {self.target_volatility}%)')

        # Sharpe assessment
        if portfolio_metrics['recent_sharpe'] < 0:
            if risk_level == 'LOW':
                risk_level = 'MODERATE'
            reason.append('Negative risk-adjusted returns')

        return {
            'risk_level': risk_level,
            'action': action,
            'reasons': reason,
            'metrics': portfolio_metrics
        }

    def generate_risk_report(
        self,
        portfolio_prices: pd.Series,
        fund_prices: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Generate comprehensive risk report.

        Args:
            portfolio_prices: Time series of portfolio values
            fund_prices: DataFrame of individual fund prices

        Returns:
            Risk report dictionary
        """
        # Portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(portfolio_prices)

        # Risk assessment
        risk_assessment = self.assess_risk_level(portfolio_metrics)

        # Position analysis
        current_positions = self.calculate_position_sizes(fund_prices, method='risk_parity')

        # Individual fund risks
        fund_returns = fund_prices.pct_change().iloc[-60:]
        fund_volatilities = fund_returns.std() * np.sqrt(252) * 100

        # Correlation analysis
        correlation_matrix = fund_returns.corr()
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()

        # Concentration risk
        max_position = current_positions.max()
        concentration_risk = 'HIGH' if max_position > 0.25 else 'MODERATE' if max_position > 0.15 else 'LOW'

        return {
            'timestamp': datetime.now(),
            'portfolio_metrics': portfolio_metrics,
            'risk_assessment': risk_assessment,
            'recommended_positions': current_positions.to_dict(),
            'fund_volatilities': fund_volatilities.to_dict(),
            'average_correlation': avg_correlation,
            'concentration_risk': concentration_risk,
            'max_position': max_position
        }


class DrawdownController:
    """
    Monitor and control portfolio drawdowns
    """

    def __init__(self, alert_levels: List[float] = [-5, -10, -15, -20]):
        """
        Args:
            alert_levels: Drawdown levels to trigger alerts
        """
        self.alert_levels = alert_levels
        self.peak_value = None
        self.alerts_triggered = {level: False for level in alert_levels}

    def update(self, current_value: float) -> List[Dict]:
        """
        Update with current portfolio value.

        Returns:
            List of triggered alerts
        """
        if self.peak_value is None:
            self.peak_value = current_value

        # Update peak
        if current_value > self.peak_value:
            self.peak_value = current_value
            # Reset alerts on new peak
            self.alerts_triggered = {level: False for level in self.alert_levels}
            return []

        # Calculate drawdown
        drawdown = ((current_value - self.peak_value) / self.peak_value) * 100

        # Check alerts
        new_alerts = []
        for level in self.alert_levels:
            if drawdown <= level and not self.alerts_triggered[level]:
                self.alerts_triggered[level] = True
                new_alerts.append({
                    'level': level,
                    'current_drawdown': drawdown,
                    'peak_value': self.peak_value,
                    'current_value': current_value,
                    'timestamp': datetime.now()
                })

        return new_alerts


class VolatilityScaling:
    """
    Dynamically scale portfolio based on volatility
    """

    def __init__(self, target_vol: float = 15.0, lookback: int = 60):
        """
        Args:
            target_vol: Target annual volatility percentage
            lookback: Days to calculate volatility
        """
        self.target_vol = target_vol
        self.lookback = lookback

    def calculate_scale_factor(self, returns: pd.Series) -> float:
        """
        Calculate scaling factor to hit target volatility.

        Args:
            returns: Time series of portfolio returns

        Returns:
            Scale factor (1.0 = no change, <1.0 = reduce, >1.0 = increase)
        """
        if len(returns) < self.lookback:
            return 1.0

        recent_returns = returns.iloc[-self.lookback:]
        current_vol = recent_returns.std() * np.sqrt(252) * 100

        if current_vol == 0:
            return 1.0

        scale_factor = self.target_vol / current_vol

        # Cap scaling between 0.5x and 1.5x
        scale_factor = max(0.5, min(1.5, scale_factor))

        return scale_factor

    def get_volatility_adjusted_weights(
        self,
        current_weights: pd.Series,
        returns: pd.Series
    ) -> pd.Series:
        """
        Adjust position weights based on volatility.

        Args:
            current_weights: Current position weights
            returns: Portfolio returns series

        Returns:
            Adjusted weights
        """
        scale_factor = self.calculate_scale_factor(returns)

        adjusted_weights = current_weights * scale_factor

        # The rest goes to cash (implicit)
        total_weight = adjusted_weights.sum()

        if total_weight > 1.0:
            adjusted_weights = adjusted_weights / total_weight

        return adjusted_weights


if __name__ == "__main__":
    print("Risk Management Module")
    print("=" * 80)

    # Example usage
    rm = RiskManager(stop_loss_pct=15.0, max_drawdown_limit=20.0, target_volatility=15.0)

    # Simulate portfolio value
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, len(dates))
    prices = pd.Series((1 + returns).cumprod() * 100, index=dates)

    # Simulate a drawdown
    prices.iloc[500:600] *= 0.85

    # Calculate metrics
    metrics = rm.calculate_portfolio_metrics(prices)
    print("\nPortfolio Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")

    # Risk assessment
    assessment = rm.assess_risk_level(metrics)
    print(f"\nRisk Assessment:")
    print(f"  Level: {assessment['risk_level']}")
    print(f"  Action: {assessment['action']}")
    print(f"  Reasons: {', '.join(assessment['reasons'])}")
