"""
Walk-Forward Validation for Time Series Models

Implements proper time series cross-validation to avoid data leakage.
Uses expanding or rolling window approach for training and testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Callable, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
import sqlite3
from pathlib import Path
import json


class WalkForwardValidator:
    """
    Time series walk-forward validation framework

    Prevents data leakage by ensuring test data always comes after training data
    """

    def __init__(
        self,
        db_path: str,
        n_splits: int = 5,
        test_size: int = 90,  # days
        gap: int = 0  # days gap between train and test
    ):
        """
        Initialize walk-forward validator.

        Args:
            db_path: Path to historical data database
            n_splits: Number of train/test splits
            test_size: Size of test set in days
            gap: Gap between train and test sets (prevents look-ahead bias)
        """
        self.db_path = db_path
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.results = []

    def create_time_splits(
        self,
        data: pd.DataFrame,
        date_column: str = 'date'
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Create time-based train/test splits.

        Args:
            data: DataFrame with time series data
            date_column: Name of date column

        Returns:
            List of (train_indices, test_indices) tuples
        """
        dates = pd.to_datetime(data[date_column]).sort_values()
        min_date = dates.min()
        max_date = dates.max()
        total_days = (max_date - min_date).days

        # Calculate split points
        test_days = self.test_size
        gap_days = self.gap

        # Minimum training days (at least 1 year)
        min_train_days = 365

        # Calculate available days for splits
        available_days = total_days - min_train_days - test_days - gap_days

        if available_days < 0:
            raise ValueError(f"Insufficient data: need at least {min_train_days + test_days + gap_days} days")

        # Calculate step size for rolling window
        step_size = available_days // (self.n_splits - 1) if self.n_splits > 1 else 0

        splits = []
        for i in range(self.n_splits):
            # Training end date
            train_end = min_date + timedelta(days=min_train_days + i * step_size)

            # Test start date (after gap)
            test_start = train_end + timedelta(days=gap_days)

            # Test end date
            test_end = test_start + timedelta(days=test_days)

            # Make sure test_end doesn't exceed max_date
            if test_end > max_date:
                break

            # Get indices for this split
            train_mask = (dates <= train_end)
            test_mask = (dates >= test_start) & (dates < test_end)

            train_indices = dates[train_mask].index
            test_indices = dates[test_mask].index

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        return splits

    def validate_model(
        self,
        model_predict_fn: Callable,
        scheme_codes: Optional[List[str]] = None,
        max_schemes: int = 50
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation on model predictions.

        Args:
            model_predict_fn: Function that takes (scheme_code, date) and returns predicted return
            scheme_codes: List of specific schemes to validate (None = all schemes)
            max_schemes: Maximum number of schemes to validate

        Returns:
            Dictionary with validation results
        """
        print("=" * 80)
        print("WALK-FORWARD VALIDATION")
        print("=" * 80)
        print(f"Number of splits: {self.n_splits}")
        print(f"Test size: {self.test_size} days")
        print(f"Gap: {self.gap} days")
        print("=" * 80)

        conn = sqlite3.connect(self.db_path)

        # Get list of schemes
        if scheme_codes is None:
            query = f"""
            SELECT DISTINCT scheme_code
            FROM nav_data
            LIMIT {max_schemes}
            """
            schemes_df = pd.read_sql_query(query, conn)
            scheme_codes = schemes_df['scheme_code'].tolist()

        all_predictions = []
        split_results = []

        for split_idx in range(self.n_splits):
            print(f"\nProcessing split {split_idx + 1}/{self.n_splits}...")

            split_predictions = []

            for scheme_code in scheme_codes:
                try:
                    # Load data for this scheme
                    query = f"""
                    SELECT date, nav
                    FROM nav_data
                    WHERE scheme_code = ?
                    ORDER BY date
                    """
                    df = pd.read_sql_query(query, conn, params=(scheme_code,))

                    if len(df) < 500:  # Need sufficient data
                        continue

                    df['date'] = pd.to_datetime(df['date'])

                    # Create splits for this scheme
                    try:
                        splits = self.create_time_splits(df)
                        if split_idx >= len(splits):
                            continue

                        train_idx, test_idx = splits[split_idx]
                    except ValueError:
                        continue

                    # Get test period data
                    test_data = df.loc[test_idx]

                    if len(test_data) < 10:
                        continue

                    # For each test date, predict and compare
                    for idx, row in test_data.iterrows():
                        test_date = row['date']
                        current_nav = row['nav']

                        # Get actual future return (30 days ahead)
                        future_date = test_date + timedelta(days=30)
                        future_data = df[
                            (df['date'] >= future_date) &
                            (df['date'] <= future_date + timedelta(days=7))
                        ]

                        if len(future_data) == 0:
                            continue

                        future_nav = future_data['nav'].iloc[0]
                        actual_return = ((future_nav / current_nav) - 1) * 100

                        # Get model prediction
                        try:
                            predicted_return = model_predict_fn(scheme_code, test_date)

                            if predicted_return is not None:
                                prediction_record = {
                                    'split': split_idx,
                                    'scheme_code': scheme_code,
                                    'date': test_date,
                                    'predicted': predicted_return,
                                    'actual': actual_return
                                }

                                split_predictions.append(prediction_record)
                                all_predictions.append(prediction_record)

                        except Exception as e:
                            continue

                except Exception as e:
                    print(f"  Error with scheme {scheme_code}: {str(e)}")
                    continue

            # Calculate metrics for this split
            if len(split_predictions) > 0:
                split_df = pd.DataFrame(split_predictions)
                y_pred = split_df['predicted'].values
                y_true = split_df['actual'].values

                # Calculate metrics
                rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
                mae = np.mean(np.abs(y_pred - y_true))

                # Directional accuracy
                dir_correct = np.sum(np.sign(y_pred) == np.sign(y_true))
                dir_acc = (dir_correct / len(y_pred)) * 100

                split_result = {
                    'split': split_idx,
                    'num_predictions': len(split_predictions),
                    'rmse': round(rmse, 4),
                    'mae': round(mae, 4),
                    'directional_accuracy': round(dir_acc, 2)
                }

                split_results.append(split_result)

                print(f"  Split {split_idx + 1} Results:")
                print(f"    Predictions: {len(split_predictions)}")
                print(f"    RMSE: {rmse:.4f}%")
                print(f"    MAE: {mae:.4f}%")
                print(f"    Directional Accuracy: {dir_acc:.2f}%")

        conn.close()

        # Calculate overall metrics
        if len(all_predictions) > 0:
            all_df = pd.DataFrame(all_predictions)
            y_pred = all_df['predicted'].values
            y_true = all_df['actual'].values

            overall_rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
            overall_mae = np.mean(np.abs(y_pred - y_true))
            overall_dir_acc = (np.sum(np.sign(y_pred) == np.sign(y_true)) / len(y_pred)) * 100

            # Calculate metrics by split
            split_rmses = [s['rmse'] for s in split_results]
            split_maes = [s['mae'] for s in split_results]
            split_dirs = [s['directional_accuracy'] for s in split_results]

            overall_results = {
                'total_predictions': len(all_predictions),
                'n_splits': len(split_results),
                'overall_rmse': round(overall_rmse, 4),
                'overall_mae': round(overall_mae, 4),
                'overall_directional_accuracy': round(overall_dir_acc, 2),
                'avg_split_rmse': round(np.mean(split_rmses), 4),
                'std_split_rmse': round(np.std(split_rmses), 4),
                'avg_split_mae': round(np.mean(split_maes), 4),
                'std_split_mae': round(np.std(split_maes), 4),
                'avg_split_directional_accuracy': round(np.mean(split_dirs), 2),
                'std_split_directional_accuracy': round(np.std(split_dirs), 2),
                'split_results': split_results,
                'predictions': all_predictions[:100]  # Save first 100 for inspection
            }

            print("\n" + "=" * 80)
            print("OVERALL VALIDATION RESULTS")
            print("=" * 80)
            print(f"Total Predictions: {overall_results['total_predictions']}")
            print(f"Number of Splits: {overall_results['n_splits']}")
            print(f"\nOverall RMSE: {overall_results['overall_rmse']:.4f}%")
            print(f"Overall MAE: {overall_results['overall_mae']:.4f}%")
            print(f"Overall Directional Accuracy: {overall_results['overall_directional_accuracy']:.2f}%")
            print(f"\nAverage Split RMSE: {overall_results['avg_split_rmse']:.4f}% (±{overall_results['std_split_rmse']:.4f}%)")
            print(f"Average Split MAE: {overall_results['avg_split_mae']:.4f}% (±{overall_results['std_split_mae']:.4f}%)")
            print(f"Average Split Dir. Acc: {overall_results['avg_split_directional_accuracy']:.2f}% (±{overall_results['std_split_directional_accuracy']:.2f}%)")
            print("=" * 80)

            return overall_results
        else:
            print("\nNo predictions generated!")
            return {'error': 'No predictions generated'}

    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save validation results to JSON file.

        Args:
            results: Results dictionary from validate_model()
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("WALK-FORWARD VALIDATION - DEMO")
    print("=" * 80)

    # Path to database
    db_path = "../data/historical_20y.db"

    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        print("Please ensure historical data is collected first.")
        exit(1)

    # Create dummy prediction function (replace with actual model)
    def dummy_predict_fn(scheme_code: str, date: datetime) -> float:
        """
        Dummy prediction function that returns random predictions.
        Replace this with your actual model prediction function.
        """
        # In reality, you would:
        # 1. Load model
        # 2. Get features for this scheme/date
        # 3. Make prediction
        # 4. Return predicted return

        # For demo, return random prediction
        return np.random.normal(5, 10)  # Mean 5%, std 10%

    # Create validator
    validator = WalkForwardValidator(
        db_path=db_path,
        n_splits=3,
        test_size=90,
        gap=0
    )

    # Run validation
    results = validator.validate_model(
        model_predict_fn=dummy_predict_fn,
        max_schemes=10  # Test on 10 schemes only for demo
    )

    # Save results
    if 'error' not in results:
        validator.save_results(
            results,
            output_path='../reports/walk_forward_validation_results.json'
        )

    print("\nValidation complete!")
