"""
Visualization Module

Create interactive charts and dashboards for backtesting results.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional
from pathlib import Path


class BacktestVisualizer:
    """
    Create comprehensive visualizations for backtest results
    """

    def __init__(self, output_dir: str = 'reports'):
        """
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_strategy_comparison(
        self,
        strategy_results: Dict[str, Dict],
        save_path: str = None
    ) -> go.Figure:
        """
        Create comparison chart for multiple strategies.

        Args:
            strategy_results: Dictionary of strategy results
            save_path: Path to save HTML file

        Returns:
            Plotly figure
        """
        # Prepare data
        strategies = list(strategy_results.keys())
        returns = [metrics['net_annualized_return'] for metrics in strategy_results.values()]
        sharpes = [metrics['sharpe_ratio'] for metrics in strategy_results.values()]
        drawdowns = [abs(metrics['max_drawdown']) for metrics in strategy_results.values()]

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Annualized Returns',
                'Sharpe Ratio',
                'Maximum Drawdown',
                'Risk-Return Profile'
            ),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )

        # Returns bar chart
        fig.add_trace(
            go.Bar(x=strategies, y=returns, name='Returns',
                   marker_color='lightblue'),
            row=1, col=1
        )

        # Sharpe bar chart
        fig.add_trace(
            go.Bar(x=strategies, y=sharpes, name='Sharpe',
                   marker_color='lightgreen'),
            row=1, col=2
        )

        # Drawdown bar chart
        fig.add_trace(
            go.Bar(x=strategies, y=drawdowns, name='Max DD',
                   marker_color='lightcoral'),
            row=2, col=1
        )

        # Risk-Return scatter
        fig.add_trace(
            go.Scatter(
                x=drawdowns, y=returns,
                mode='markers+text',
                text=strategies,
                textposition='top center',
                marker=dict(size=15, color=sharpes, colorscale='Viridis',
                           showscale=True, colorbar=dict(title="Sharpe")),
                name='Strategies'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_xaxes(title_text="Strategy", row=1, col=1)
        fig.update_xaxes(title_text="Strategy", row=1, col=2)
        fig.update_xaxes(title_text="Strategy", row=2, col=1)
        fig.update_xaxes(title_text="Max Drawdown (%)", row=2, col=2)

        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=2)

        fig.update_layout(
            title_text="Strategy Performance Comparison",
            showlegend=False,
            height=800,
            template='plotly_white'
        )

        # Save if path provided
        if save_path:
            fig.write_html(str(self.output_dir / save_path))
            print(f"  ✓ Saved: {save_path}")

        return fig

    def plot_monte_carlo_distribution(
        self,
        final_returns: np.ndarray,
        save_path: str = None
    ) -> go.Figure:
        """
        Plot Monte Carlo return distribution.

        Args:
            final_returns: Array of final returns from simulation
            save_path: Path to save HTML

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=final_returns,
            nbinsx=50,
            name='Distribution',
            marker_color='lightblue',
            opacity=0.7
        ))

        # Add percentile lines
        percentiles = [5, 50, 95]
        colors = ['red', 'green', 'red']
        for p, color in zip(percentiles, colors):
            value = np.percentile(final_returns, p)
            fig.add_vline(
                x=value,
                line_dash="dash",
                line_color=color,
                annotation_text=f"{p}th: {value:.1f}%",
                annotation_position="top"
            )

        fig.update_layout(
            title="Monte Carlo Return Distribution (1 Year)",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            template='plotly_white',
            showlegend=False
        )

        if save_path:
            fig.write_html(str(self.output_dir / save_path))
            print(f"  ✓ Saved: {save_path}")

        return fig

    def plot_rolling_metrics(
        self,
        rolling_metrics: pd.DataFrame,
        save_path: str = None
    ) -> go.Figure:
        """
        Plot rolling performance metrics.

        Args:
            rolling_metrics: DataFrame with rolling metrics
            save_path: Path to save HTML

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Rolling Returns',
                'Rolling Sharpe Ratio',
                'Rolling Volatility',
                'Rolling Max Drawdown'
            )
        )

        # Rolling returns
        fig.add_trace(
            go.Scatter(
                x=rolling_metrics.index,
                y=rolling_metrics['rolling_return'],
                mode='lines',
                name='Returns',
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=rolling_metrics.index,
                y=rolling_metrics['rolling_sharpe'],
                mode='lines',
                name='Sharpe',
                line=dict(color='green')
            ),
            row=1, col=2
        )

        # Rolling volatility
        fig.add_trace(
            go.Scatter(
                x=rolling_metrics.index,
                y=rolling_metrics['rolling_volatility'],
                mode='lines',
                name='Volatility',
                line=dict(color='orange')
            ),
            row=2, col=1
        )

        # Rolling drawdown
        fig.add_trace(
            go.Scatter(
                x=rolling_metrics.index,
                y=rolling_metrics['rolling_max_dd'],
                mode='lines',
                name='Max DD',
                line=dict(color='red')
            ),
            row=2, col=2
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)

        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe", row=1, col=2)
        fig.update_yaxes(title_text="Vol (%)", row=2, col=1)
        fig.update_yaxes(title_text="DD (%)", row=2, col=2)

        fig.update_layout(
            title_text="Rolling Performance Metrics (1-Year Windows)",
            showlegend=False,
            height=700,
            template='plotly_white'
        )

        if save_path:
            fig.write_html(str(self.output_dir / save_path))
            print(f"  ✓ Saved: {save_path}")

        return fig

    def create_dashboard(
        self,
        strategy_results: Dict[str, Dict],
        rolling_metrics: pd.DataFrame,
        monte_carlo_returns: np.ndarray = None
    ) -> go.Figure:
        """
        Create comprehensive dashboard.

        Args:
            strategy_results: Strategy comparison results
            rolling_metrics: Rolling metrics DataFrame
            monte_carlo_returns: Monte Carlo returns array

        Returns:
            Plotly figure
        """
        print("\nGenerating Interactive Dashboard...")

        # Create individual plots
        self.plot_strategy_comparison(strategy_results, 'strategy_comparison.html')
        self.plot_rolling_metrics(rolling_metrics, 'rolling_metrics.html')

        if monte_carlo_returns is not None:
            self.plot_monte_carlo_distribution(monte_carlo_returns, 'monte_carlo.html')

        print("\n  ✓ Dashboard components created successfully!")
        print(f"\n  View reports at: {self.output_dir}/")

        return None


if __name__ == "__main__":
    print("Visualization Module Test")
    print("=" * 80)

    # Generate sample data
    sample_results = {
        'Conservative': {
            'net_annualized_return': 12.5,
            'sharpe_ratio': 1.2,
            'max_drawdown': -10.5
        },
        'Balanced': {
            'net_annualized_return': 15.8,
            'sharpe_ratio': 1.4,
            'max_drawdown': -15.2
        },
        'Aggressive': {
            'net_annualized_return': 19.3,
            'sharpe_ratio': 1.1,
            'max_drawdown': -22.8
        }
    }

    viz = BacktestVisualizer()
    viz.plot_strategy_comparison(sample_results, 'test_comparison.html')

    print("\nTest visualization created!")
