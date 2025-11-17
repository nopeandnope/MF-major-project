"""
Model Performance Evaluator

Comprehensive evaluation framework for assessing model prediction accuracy
including financial metrics, statistical measures, and backtesting integration.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
from pathlib import Path


class ModelPerformanceEvaluator:
    """
    Comprehensive evaluation of model predictions with financial metrics
    """

    def __init__(self, risk_free_rate: float = 0.065):
        """
        Initialize evaluator.

        Args:
            risk_free_rate: Annual risk-free rate (default 6.5% for India)
        """
        self.risk_free_rate = risk_free_rate
        self.predictions = []
        self.evaluation_history = []

    def add_prediction(
        self,
        predicted: float,
        actual: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a prediction-actual pair for evaluation.

        Args:
            predicted: Model's predicted return (%)
            actual: Actual return (%)
            metadata: Additional information (scheme_code, date, etc.)
        """
        self.predictions.append({
            'predicted': predicted,
            'actual': actual,
            'metadata': metadata or {}
        })

    def clear_predictions(self):
        """Clear all stored predictions"""
        self.predictions = []

    def calculate_prediction_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive prediction accuracy metrics.

        Returns:
            Dictionary with accuracy metrics
        """
        if len(self.predictions) == 0:
            raise ValueError("No predictions available for evaluation")

        y_pred = np.array([p['predicted'] for p in self.predictions])
        y_true = np.array([p['actual'] for p in self.predictions])

        # Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Mean Absolute Error
        mae = mean_absolute_error(y_true, y_pred)

        # Mean Absolute Percentage Error (handle division by zero)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.inf

        # R² Score
        r2 = r2_score(y_true, y_pred)

        # Directional Accuracy
        directional_accuracy = self.calculate_directional_accuracy(y_pred, y_true)

        # Spearman Correlation (ranking)
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)

        # Pearson Correlation
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)

        # Prediction bias (average error)
        bias = np.mean(y_pred - y_true)

        return {
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'mape': round(mape, 2),
            'r2_score': round(r2, 4),
            'directional_accuracy': round(directional_accuracy, 2),
            'spearman_correlation': round(spearman_corr, 4),
            'spearman_p_value': round(spearman_p, 4),
            'pearson_correlation': round(pearson_corr, 4),
            'pearson_p_value': round(pearson_p, 4),
            'prediction_bias': round(bias, 4),
            'num_predictions': len(self.predictions)
        }

    def calculate_directional_accuracy(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Calculate percentage of times model correctly predicted direction.

        Args:
            y_pred: Predicted returns
            y_true: Actual returns

        Returns:
            Directional accuracy percentage
        """
        pred_direction = np.sign(y_pred)
        true_direction = np.sign(y_true)

        correct = np.sum(pred_direction == true_direction)
        total = len(y_pred)

        return (correct / total) * 100 if total > 0 else 0

    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sharpe ratio from returns.

        Args:
            returns: Array of returns (as decimals, not percentages)

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
        sharpe = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)
        return sharpe

    def calculate_information_ratio(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """
        Calculate Information ratio vs benchmark.

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns

        Returns:
            Annualized Information ratio
        """
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns)

        if tracking_error == 0:
            return 0.0

        return (np.mean(excess_returns) / tracking_error) * np.sqrt(252)

    def quintile_analysis(self) -> Dict[str, Any]:
        """
        Perform quintile analysis - compare top 20% predictions vs bottom 20%.

        Returns:
            Dictionary with quintile analysis results
        """
        if len(self.predictions) < 10:
            return {
                'error': 'Insufficient predictions for quintile analysis (need at least 10)'
            }

        # Sort predictions by predicted return
        sorted_preds = sorted(
            self.predictions,
            key=lambda x: x['predicted'],
            reverse=True
        )

        quintile_size = len(sorted_preds) // 5

        # Top quintile (highest predicted returns)
        top_quintile = sorted_preds[:quintile_size]
        top_actual_returns = [p['actual'] for p in top_quintile]
        top_predicted_returns = [p['predicted'] for p in top_quintile]

        # Bottom quintile (lowest predicted returns)
        bottom_quintile = sorted_preds[-quintile_size:]
        bottom_actual_returns = [p['actual'] for p in bottom_quintile]
        bottom_predicted_returns = [p['predicted'] for p in bottom_quintile]

        # Calculate metrics
        top_avg_actual = np.mean(top_actual_returns)
        bottom_avg_actual = np.mean(bottom_actual_returns)
        spread = top_avg_actual - bottom_avg_actual

        # Sharpe ratios (convert % to decimal)
        top_sharpe = self.calculate_sharpe_ratio(np.array(top_actual_returns) / 100)
        bottom_sharpe = self.calculate_sharpe_ratio(np.array(bottom_actual_returns) / 100)

        return {
            'top_quintile_avg_actual': round(top_avg_actual, 2),
            'top_quintile_avg_predicted': round(np.mean(top_predicted_returns), 2),
            'bottom_quintile_avg_actual': round(bottom_avg_actual, 2),
            'bottom_quintile_avg_predicted': round(np.mean(bottom_predicted_returns), 2),
            'quintile_spread': round(spread, 2),
            'top_quintile_sharpe': round(top_sharpe, 2),
            'bottom_quintile_sharpe': round(bottom_sharpe, 2),
            'top_quintile_size': len(top_quintile),
            'bottom_quintile_size': len(bottom_quintile)
        }

    def category_performance(self) -> pd.DataFrame:
        """
        Analyze performance by fund category.

        Returns:
            DataFrame with metrics by category
        """
        # Extract categories from metadata
        categories = {}

        for pred in self.predictions:
            category = pred['metadata'].get('category', 'Unknown')

            if category not in categories:
                categories[category] = {
                    'predictions': [],
                    'actuals': []
                }

            categories[category]['predictions'].append(pred['predicted'])
            categories[category]['actuals'].append(pred['actual'])

        # Calculate metrics for each category
        results = []
        for category, data in categories.items():
            y_pred = np.array(data['predictions'])
            y_true = np.array(data['actuals'])

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            dir_acc = self.calculate_directional_accuracy(y_pred, y_true)

            results.append({
                'category': category,
                'count': len(y_pred),
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'directional_accuracy': round(dir_acc, 2),
                'avg_predicted': round(np.mean(y_pred), 2),
                'avg_actual': round(np.mean(y_true), 2)
            })

        return pd.DataFrame(results)

    def generate_evaluation_report(
        self,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.

        Args:
            save_path: Optional path to save report as JSON

        Returns:
            Dictionary with complete evaluation results
        """
        print("=" * 80)
        print("MODEL PERFORMANCE EVALUATION REPORT")
        print("=" * 80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Predictions: {len(self.predictions)}")
        print("=" * 80)

        # 1. Prediction Accuracy Metrics
        metrics = self.calculate_prediction_metrics()
        print("\n1. PREDICTION ACCURACY METRICS")
        print("-" * 80)
        print(f"RMSE:                    {metrics['rmse']:.4f}%")
        print(f"MAE:                     {metrics['mae']:.4f}%")
        print(f"MAPE:                    {metrics['mape']:.2f}%")
        print(f"R² Score:                {metrics['r2_score']:.4f}")
        print(f"Directional Accuracy:    {metrics['directional_accuracy']:.2f}%")
        print(f"Spearman Correlation:    {metrics['spearman_correlation']:.4f}")
        print(f"Pearson Correlation:     {metrics['pearson_correlation']:.4f}")
        print(f"Prediction Bias:         {metrics['prediction_bias']:.4f}%")

        # 2. Performance Benchmarks
        print("\n2. PERFORMANCE EVALUATION")
        print("-" * 80)

        # RMSE benchmarks
        if metrics['rmse'] < 3:
            print("[OK] RMSE: EXCELLENT (< 3%)")
        elif metrics['rmse'] < 5:
            print("[OK] RMSE: GOOD (< 5%)")
        elif metrics['rmse'] < 8:
            print("[ACCEPTABLE] RMSE: ACCEPTABLE (< 8%)")
        else:
            print("[NEEDS IMPROVEMENT] RMSE: NEEDS IMPROVEMENT (> 8%)")

        # Directional accuracy benchmarks
        if metrics['directional_accuracy'] > 70:
            print("[OK] Directional Accuracy: EXCELLENT (> 70%)")
        elif metrics['directional_accuracy'] > 65:
            print("[OK] Directional Accuracy: GOOD (> 65%)")
        elif metrics['directional_accuracy'] > 55:
            print("[ACCEPTABLE] Directional Accuracy: ACCEPTABLE (> 55%)")
        else:
            print("[NEEDS IMPROVEMENT] Directional Accuracy: NEEDS IMPROVEMENT (< 55%)")

        # R² benchmarks
        if metrics['r2_score'] > 0.7:
            print("[OK] R² Score: EXCELLENT (> 0.7)")
        elif metrics['r2_score'] > 0.5:
            print("[OK] R² Score: GOOD (> 0.5)")
        elif metrics['r2_score'] > 0.3:
            print("[ACCEPTABLE] R² Score: ACCEPTABLE (> 0.3)")
        else:
            print("[NEEDS IMPROVEMENT] R² Score: NEEDS IMPROVEMENT (< 0.3)")

        # 3. Quintile Analysis
        quintile_results = self.quintile_analysis()
        if 'error' not in quintile_results:
            print("\n3. QUINTILE ANALYSIS (Top 20% vs Bottom 20%)")
            print("-" * 80)
            print(f"Top Quintile Avg Actual Return:    {quintile_results['top_quintile_avg_actual']:.2f}%")
            print(f"Bottom Quintile Avg Actual Return: {quintile_results['bottom_quintile_avg_actual']:.2f}%")
            print(f"Spread:                            {quintile_results['quintile_spread']:.2f}%")
            print(f"Top Quintile Sharpe:               {quintile_results['top_quintile_sharpe']:.2f}")
            print(f"Bottom Quintile Sharpe:            {quintile_results['bottom_quintile_sharpe']:.2f}")

        # 4. Category Performance
        category_df = self.category_performance()
        if len(category_df) > 0:
            print("\n4. PERFORMANCE BY FUND CATEGORY")
            print("-" * 80)
            print(category_df.to_string(index=False))

        print("\n" + "=" * 80)

        # Compile full report
        report = {
            'timestamp': datetime.now().isoformat(),
            'num_predictions': len(self.predictions),
            'accuracy_metrics': metrics,
            'quintile_analysis': quintile_results,
            'category_performance': category_df.to_dict('records') if len(category_df) > 0 else []
        }

        # Save if path provided
        if save_path:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            print(f"\nReport saved to: {save_path}")

        self.evaluation_history.append(report)
        return report

    def plot_prediction_scatter(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Create scatter plot of predicted vs actual returns.

        Args:
            save_path: Optional path to save plot
            show: Whether to display plot
        """
        y_pred = np.array([p['predicted'] for p in self.predictions])
        y_true = np.array([p['actual'] for p in self.predictions])

        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)

        # Add diagonal line (perfect predictions)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        plt.xlabel('Actual Return (%)', fontsize=12)
        plt.ylabel('Predicted Return (%)', fontsize=12)
        plt.title('Model Predictions vs Actual Returns', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add metrics text
        metrics = self.calculate_prediction_metrics()
        textstr = f"RMSE: {metrics['rmse']:.2f}%\nR²: {metrics['r2_score']:.3f}\nDir. Acc: {metrics['directional_accuracy']:.1f}%"
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("MODEL PERFORMANCE EVALUATOR - DEMO")
    print("=" * 80)

    # Create evaluator
    evaluator = ModelPerformanceEvaluator()

    # Add sample predictions (replace with actual model predictions)
    np.random.seed(42)
    for i in range(100):
        actual = np.random.normal(5, 10)  # Mean 5%, std 10%
        predicted = actual + np.random.normal(0, 3)  # Add prediction error

        evaluator.add_prediction(
            predicted=predicted,
            actual=actual,
            metadata={
                'scheme_code': f'FUND{i:03d}',
                'category': np.random.choice(['Large Cap', 'Mid Cap', 'Small Cap'])
            }
        )

    # Generate report
    report = evaluator.generate_evaluation_report(
        save_path='../reports/model_evaluation_report.json'
    )

    # Create scatter plot
    evaluator.plot_prediction_scatter(
        save_path='../reports/prediction_scatter.png',
        show=False
    )

    print("\nEvaluation complete!")
