"""
Comprehensive Backtesting and Model Evaluation Runner

This script runs the complete Phase 1 and Phase 2 implementation:
- Data quality validation
- Multi-timeframe backtesting
- Crisis scenario testing
- Training data generation
- Model performance evaluation

Run this script to execute all backtesting and evaluation tasks.
"""

import sys
from pathlib import Path
import argparse

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_quality_validator import DataQualityValidator
from backtesting.mf_backtest_engine import MutualFundBacktester
from src.advanced_training_data_generator import AdvancedTrainingDataGenerator
from evaluation.model_evaluator import ModelPerformanceEvaluator
import pandas as pd
import numpy as np
from datetime import datetime


def check_database_exists(db_path: str) -> bool:
    """Check if historical database exists"""
    path = Path(db_path)
    if not path.exists():
        print(f"ERROR: Database not found at {db_path}")
        print("\nPlease run the historical data collector first:")
        print("  python src/historical_data_collector.py")
        return False
    return True


def run_data_quality_validation(db_path: str):
    """Run data quality validation on the database"""
    print("\n" + "=" * 80)
    print("STEP 1: DATA QUALITY VALIDATION")
    print("=" * 80)

    try:
        validator = DataQualityValidator(data_source='mfapi.in')

        # Validate database
        print("Validating database quality...")
        db_report = validator.validate_database(db_path)

        print(f"\n{'Database Analysis':^80}")
        print("-" * 80)
        print(f"Database Path:      {db_path}")
        print(f"Total Schemes:      {db_report['total_schemes']}")
        print(f"Passed Schemes:     {db_report['passed_schemes']} [OK]")
        print(f"Failed Schemes:     {db_report['failed_schemes']}")
        print(f"Pass Rate:          {db_report['pass_rate']:.1f}%")
        print(f"Avg Quality Score:  {db_report['avg_quality_score']:.2f}/100")

        if db_report['failed_schemes'] > 0:
            print(f"\n[!] Warning: {db_report['failed_schemes']} schemes failed quality check")
            print("Failed Schemes (showing first 5):")
            for scheme in db_report['failed_scheme_details'][:5]:
                print(f"  • {scheme['scheme_code']}: Score {scheme['score']}/100")

        # Save report
        validator.export_quality_report(
            db_report,
            'reports/data_quality_report.json'
        )
        print("\n[OK] Quality report saved to: reports/data_quality_report.json")

        return db_report

    except Exception as e:
        print(f"\n[X] Error during data quality validation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_backtesting(db_path: str, num_funds: int = 10):
    """Run comprehensive backtesting"""
    print("\n" + "=" * 80)
    print("STEP 2: BACKTESTING FRAMEWORK")
    print("=" * 80)

    try:
        # Initialize backtester
        print(f"Initializing backtester with {num_funds} top funds...")
        backtester = MutualFundBacktester(db_path=db_path)

        # 2A: Multi-timeframe testing
        print("\n2A. MULTI-TIMEFRAME BACKTESTS")
        print("-" * 80)

        timeframe_results = backtester.run_timeframe_tests(num_funds=num_funds)

        # Print summary - organized by term length
        summary_data = []

        # Define display order: short -> medium -> long term
        timeframe_order = ['3_months', '6_months', '1_year', '2_years', '3_years', '5_years', '10_years', '15_years']

        for period in timeframe_order:
            if period in timeframe_results:
                result = timeframe_results[period]
                metrics = backtester.calculate_performance_metrics(result)

                # Create readable period name
                period_display = period.replace('_', ' ').title()

                summary_data.append({
                    'Period': period_display,
                    'Return (%)': metrics['total_return'],
                    'Ann. Return (%)': metrics['annualized_return'],
                    'Sharpe': metrics['sharpe_ratio'],
                    'Max DD (%)': metrics['max_drawdown'],
                    'Days': metrics['total_days']
                })

        if len(summary_data) > 0:
            print(f"\n{'Timeframe Performance Summary':^80}")
            print("-" * 80)
            summary_df = pd.DataFrame(summary_data)
            print(summary_df.to_string(index=False))

            # Generate COMBINED comprehensive report
            print(f"\n[OK] Generating comprehensive combined report...")
            try:
                backtester.generate_combined_report(
                    timeframe_results,
                    crisis_results=None,  # Will add crisis results later
                    output_path='reports/backtest_comprehensive.html'
                )
                print(f"  [OK] Comprehensive report generated: reports/backtest_comprehensive.html")
            except Exception as e:
                print(f"  [X] Failed to generate comprehensive report: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("\n[!] Warning: No timeframe results generated")

        # 2B: Crisis scenario testing
        print("\n2B. CRISIS SCENARIO TESTING")
        print("-" * 80)

        crisis_results = backtester.test_crisis_scenarios(num_funds=num_funds)

        crisis_data = []
        for crisis_name, crisis_info in crisis_results.items():
            metrics = backtester.calculate_performance_metrics(crisis_info['result'])
            crisis_data.append({
                'Scenario': crisis_name.replace('_', ' ').title(),
                'Period': crisis_info['description'][:50],
                'Return (%)': metrics['total_return'],
                'Max DD (%)': metrics['max_drawdown'],
                'Sharpe': metrics['sharpe_ratio'],
                'Days': metrics['total_days']
            })

        if len(crisis_data) > 0:
            print(f"\n{'Crisis Scenario Performance':^80}")
            print("-" * 80)
            crisis_df = pd.DataFrame(crisis_data)
            print(crisis_df.to_string(index=False))

            # Re-generate comprehensive report with crisis results
            if len(timeframe_results) > 0:
                print(f"\n[OK] Updating comprehensive report with crisis scenarios...")
                try:
                    backtester.generate_combined_report(
                        timeframe_results,
                        crisis_results=crisis_results,
                        output_path='reports/backtest_comprehensive.html'
                    )
                    print(f"  [OK] Comprehensive report updated with crisis analysis")
                except Exception as e:
                    print(f"  [X] Failed to update report: {str(e)}")
        else:
            print("\n[!] Warning: No crisis scenario results generated")

        return timeframe_results, crisis_results

    except Exception as e:
        print(f"\n[X] Error during backtesting: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, {}


def run_training_data_generation(db_path: str, num_examples: int = 1000):
    """Generate enhanced training data"""
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING DATA GENERATION")
    print("=" * 80)

    try:
        generator = AdvancedTrainingDataGenerator(db_path)

        print(f"Target: {num_examples} training examples")
        print("This may take several minutes...\n")

        training_data = generator.generate_dataset(num_examples=num_examples)

        if len(training_data) > 0:
            # Save dataset
            output_path = 'data/training_data_enhanced.jsonl'
            generator.save_dataset(training_data, output_path=output_path)

            print(f"\n[OK] Successfully generated {len(training_data)} training examples")
            print(f"[OK] Data saved to: {output_path}")

            # Show sample
            print("\n" + "-" * 80)
            print(f"{'SAMPLE TRAINING EXAMPLE':^80}")
            print("-" * 80)
            if len(training_data) > 0:
                sample = training_data[0]
                print("\nPrompt Preview:")
                print(sample['prompt'][:400] + "..." if len(sample['prompt']) > 400 else sample['prompt'])
                print("\nCompletion Preview:")
                print(sample['completion'][:250] + "..." if len(sample['completion']) > 250 else sample['completion'])

            return training_data
        else:
            print("\n[!] WARNING: No training examples generated!")
            print("Possible reasons:")
            print("  • Insufficient historical data (need at least 15 months per fund)")
            print("  • Date range issues in database")
            print("  • All funds missing future data for predictions")
            return []

    except Exception as e:
        print(f"\n[X] Error during training data generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def run_model_evaluation():
    """Run model performance evaluation (demo with synthetic data)"""
    print("\n" + "=" * 80)
    print("STEP 4: MODEL PERFORMANCE EVALUATION")
    print("=" * 80)
    print("NOTE: Using synthetic predictions for demonstration.")
    print("      Replace with actual model predictions in production.")
    print("-" * 80)

    evaluator = ModelPerformanceEvaluator()

    # Generate synthetic predictions for demonstration
    np.random.seed(42)
    categories = ['Large Cap', 'Mid Cap', 'Small Cap', 'Multi Cap', 'Debt']

    for i in range(200):
        actual = np.random.normal(6, 12)  # Mean 6%, std 12%
        # Model predictions with some accuracy
        predicted = actual * 0.7 + np.random.normal(0, 4)

        evaluator.add_prediction(
            predicted=predicted,
            actual=actual,
            metadata={
                'scheme_code': f'FUND{i:03d}',
                'category': np.random.choice(categories),
                'date': f'2024-{(i % 12) + 1:02d}-01'
            }
        )

    # Generate evaluation report
    report = evaluator.generate_evaluation_report(
        save_path='reports/model_evaluation_report.json'
    )

    # Generate scatter plot
    try:
        evaluator.plot_prediction_scatter(
            save_path='reports/prediction_scatter.png',
            show=False
        )
    except Exception as e:
        print(f"Could not generate scatter plot: {e}")

    return report


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Run comprehensive backtesting and evaluation'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='data/historical_20y.db',
        help='Path to historical database'
    )
    parser.add_argument(
        '--num-funds',
        type=int,
        default=10,
        help='Number of funds to use in backtesting'
    )
    parser.add_argument(
        '--training-examples',
        type=int,
        default=1000,
        help='Number of training examples to generate'
    )
    parser.add_argument(
        '--skip-quality',
        action='store_true',
        help='Skip data quality validation'
    )
    parser.add_argument(
        '--skip-backtest',
        action='store_true',
        help='Skip backtesting'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training data generation'
    )
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip model evaluation'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MUTUAL FUND ADVISOR - PHASE 1 & 2 IMPLEMENTATION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database: {args.db_path}")
    print(f"Number of funds: {args.num_funds}")
    print(f"Training examples: {args.training_examples}")
    print("=" * 80)

    # Check database exists
    if not check_database_exists(args.db_path):
        return

    # Create reports directory
    Path('reports').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)

    results = {}

    try:
        # Step 1: Data Quality Validation
        if not args.skip_quality:
            results['quality'] = run_data_quality_validation(args.db_path)
        else:
            print("\nSkipping data quality validation...")

        # Step 2: Backtesting
        if not args.skip_backtest:
            results['timeframe'], results['crisis'] = run_backtesting(
                args.db_path,
                num_funds=args.num_funds
            )
        else:
            print("\nSkipping backtesting...")

        # Step 3: Training Data Generation
        if not args.skip_training:
            results['training_data'] = run_training_data_generation(
                args.db_path,
                num_examples=args.training_examples
            )
        else:
            print("\nSkipping training data generation...")

        # Step 4: Model Evaluation
        if not args.skip_evaluation:
            results['evaluation'] = run_model_evaluation()
        else:
            print("\nSkipping model evaluation...")

        # Final summary
        print("\n" + "=" * 80)
        print(f"{'EXECUTION COMPLETE':^80}")
        print("=" * 80)
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Check which files were actually generated
        expected_files = {
            'reports/data_quality_report.json': 'Data quality validation report',
            'reports/backtest_comprehensive.html': '[*] COMPREHENSIVE multi-timeframe backtesting report with detailed analysis',
            'reports/model_evaluation_report.json': 'Model performance metrics',
            'reports/prediction_scatter.png': 'Prediction accuracy visualization',
            'data/training_data_enhanced.jsonl': 'Training dataset for LLM'
        }

        print("\n" + "-" * 80)
        print("Generated Files:")
        print("-" * 80)
        files_generated = 0
        files_missing = 0

        for file_path, description in expected_files.items():
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size
                size_str = f"{file_size:,} bytes" if file_size < 1024 else f"{file_size/1024:.1f} KB"
                print(f"[OK] {file_path:<45} ({size_str})")
                print(f"     -> {description}")
                files_generated += 1
            else:
                print(f"[X] {file_path:<45} [NOT GENERATED]")
                files_missing += 1

        print("\n" + "-" * 80)
        print(f"Summary: {files_generated} files generated, {files_missing} missing")
        print("-" * 80)

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("1. View Comprehensive Backtesting Report:")
        print("   [*] Open reports/backtest_comprehensive.html in your browser")
        print("")
        print("   This comprehensive report includes:")
        print("   • Executive Summary with best-performing periods")
        print("   • Visual charts comparing all timeframes")
        print("   • Detailed metrics for SHORT (3-6mo), MEDIUM (1-3yr), LONG (5-15yr) terms")
        print("   • Crisis scenario analysis (COVID-19, 2022 correction, etc.)")
        print("   • Complete explanation of all metrics (Sharpe, Sortino, Drawdown, etc.)")
        print("   • Key insights and strategic recommendations")
        print("   • Risk considerations and disclaimers")
        print("")
        print("2. Review Data Quality:")
        print("   • Check reports/data_quality_report.json for data issues")
        print("   • Address any failed schemes if needed")
        print("")
        print("3. Model Training:")
        if Path('data/training_data_enhanced.jsonl').exists():
            print("   [OK] Training data ready in data/training_data_enhanced.jsonl")
            print("   • Use this data to fine-tune your LLM")
            print("   • Follow the training guide in documentation")
        else:
            print("   [X] Training data not generated - please investigate issues above")
        print("")
        print("4. Production Deployment:")
        print("   • Integrate trained model with evaluation framework")
        print("   • Set up automated backtesting pipeline")
        print("   • Monitor model performance over time")
        print("=" * 80)

    except Exception as e:
        print(f"\nERROR: Execution failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
