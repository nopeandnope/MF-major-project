# Phase 1 & 2 Implementation Guide

## Backtesting Framework and Model Evaluation

This document describes the implementation of Phase 1 (Backtesting Framework) and Phase 2 (Model Evaluation) from the future_idea.txt roadmap.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [What Was Implemented](#what-was-implemented)
3. [Directory Structure](#directory-structure)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Detailed Usage](#detailed-usage)
7. [Understanding the Output](#understanding-the-output)
8. [Next Steps](#next-steps)

---

## 🎯 Overview

This implementation provides a comprehensive framework for:

- **Validating data quality** of mutual fund historical data
- **Backtesting investment strategies** across multiple timeframes
- **Testing performance during crisis scenarios** (COVID-19, 2022 correction, etc.)
- **Generating high-quality training data** for LLM fine-tuning
- **Evaluating model predictions** with financial metrics
- **Walk-forward validation** to prevent data leakage

---

## ✅ What Was Implemented

### Phase 1: Backtesting Framework

#### 1.1 Data Quality Validator (`src/data_quality_validator.py`)
- Checks for missing values, outliers, date gaps
- Validates NAV values and data freshness
- Cross-validates data from multiple sources
- Generates comprehensive quality reports

#### 1.2 Mutual Fund Backtester (`backtesting/mf_backtest_engine.py`)
- **Multi-timeframe testing**: 3 months to 10+ years
- **Crisis scenario testing**: COVID-19, 2022 correction, demonetization
- **Equal-weight strategy** with quarterly/monthly/yearly rebalancing
- **Model-based strategy** support for AI predictions
- **Performance metrics**: Sharpe, Sortino, Calmar, Max Drawdown
- **HTML report generation** using quantstats

### Phase 2: Model Evaluation

#### 2.1 Advanced Training Data Generator (`src/advanced_training_data_generator.py`)
- Generates rich contextual training examples
- Includes performance metrics, risk metrics, market conditions
- Produces high-quality prompts and completions
- Supports JSONL format for LLM training

#### 2.2 Model Performance Evaluator (`evaluation/model_evaluator.py`)
- Prediction accuracy metrics (RMSE, MAE, MAPE, R²)
- Directional accuracy calculation
- Quintile analysis (top 20% vs bottom 20%)
- Category-wise performance analysis
- Visualization (scatter plots)

#### 2.3 Walk-Forward Validation (`evaluation/walk_forward_validation.py`)
- Time series cross-validation
- Prevents data leakage (no look-ahead bias)
- Expanding/rolling window support
- Split-by-split performance tracking

---

## 📁 Directory Structure

```
MF_Advisor_Project/
├── backtesting/
│   └── mf_backtest_engine.py      # Main backtesting engine
├── evaluation/
│   ├── model_evaluator.py         # Model performance evaluation
│   └── walk_forward_validation.py # Time series validation
├── src/
│   ├── data_quality_validator.py  # Data quality checks
│   └── advanced_training_data_generator.py  # Training data generation
├── reports/                        # Generated reports (created automatically)
│   ├── data_quality_report.json
│   ├── backtest_1year.html
│   ├── model_evaluation_report.json
│   └── prediction_scatter.png
├── data/                          # Data storage
│   ├── historical_20y.db          # Your existing database
│   └── training_data_enhanced.jsonl  # Generated training data
└── run_backtesting_and_evaluation.py  # Main execution script
```

---

## 🔧 Installation

### 1. Install Required Dependencies

```bash
# Activate your virtual environment
cd "C:\College notes\4th year\major project\MF_Advisor_Project"
.venv\Scripts\activate

# Install backtesting libraries (already done)
pip install bt quantstats scipy matplotlib
```

### 2. Verify Database Exists

Make sure you have the historical database at:
```
data/historical_20y.db
```

If not, run your data collector:
```bash
python src/historical_data_collector.py
```

---

## 🚀 Quick Start

### Run Complete Analysis

Execute all phases (quality check, backtesting, training data, evaluation):

```bash
python run_backtesting_and_evaluation.py
```

### Run with Custom Options

```bash
# Use custom database path
python run_backtesting_and_evaluation.py --db-path "path/to/your/database.db"

# Test with more funds
python run_backtesting_and_evaluation.py --num-funds 20

# Generate more training examples
python run_backtesting_and_evaluation.py --training-examples 5000

# Skip certain steps
python run_backtesting_and_evaluation.py --skip-quality --skip-training
```

### Available Options

- `--db-path`: Path to historical database (default: `data/historical_20y.db`)
- `--num-funds`: Number of funds to use (default: 10)
- `--training-examples`: Number of training examples (default: 1000)
- `--skip-quality`: Skip data quality validation
- `--skip-backtest`: Skip backtesting
- `--skip-training`: Skip training data generation
- `--skip-evaluation`: Skip model evaluation

---

## 📖 Detailed Usage

### 1. Data Quality Validation

```python
from src.data_quality_validator import DataQualityValidator

validator = DataQualityValidator(data_source='mfapi.in')

# Validate entire database
db_report = validator.validate_database('data/historical_20y.db')

print(f"Pass Rate: {db_report['pass_rate']}%")
print(f"Avg Quality Score: {db_report['avg_quality_score']}/100")
```

### 2. Backtesting

```python
from backtesting.mf_backtest_engine import MutualFundBacktester

# Initialize backtester
backtester = MutualFundBacktester(db_path='data/historical_20y.db')

# Run multi-timeframe tests
timeframe_results = backtester.run_timeframe_tests(num_funds=10)

# Get metrics for 1-year period
if '1_year' in timeframe_results:
    metrics = backtester.calculate_performance_metrics(timeframe_results['1_year'])
    print(f"1-Year Return: {metrics['total_return']}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']}")
    print(f"Max Drawdown: {metrics['max_drawdown']}%")

# Generate HTML report
backtester.generate_html_report(
    timeframe_results['1_year'],
    output_path='reports/backtest_1year.html'
)
```

### 3. Crisis Scenario Testing

```python
# Test during market crashes
crisis_results = backtester.test_crisis_scenarios(num_funds=10)

for crisis_name, crisis_info in crisis_results.items():
    metrics = backtester.calculate_performance_metrics(crisis_info['result'])
    print(f"{crisis_name}:")
    print(f"  Return: {metrics['total_return']}%")
    print(f"  Max Drawdown: {metrics['max_drawdown']}%")
```

### 4. Generate Training Data

```python
from src.advanced_training_data_generator import AdvancedTrainingDataGenerator

generator = AdvancedTrainingDataGenerator('data/historical_20y.db')

# Generate 1000 training examples
training_data = generator.generate_dataset(num_examples=1000)

# Save to JSONL format
generator.save_dataset(training_data, output_path='data/training_enhanced.jsonl')

# Show sample
print(training_data[0]['prompt'])
print(training_data[0]['completion'])
```

### 5. Evaluate Model Performance

```python
from evaluation.model_evaluator import ModelPerformanceEvaluator

evaluator = ModelPerformanceEvaluator()

# Add predictions
for prediction, actual in zip(model_predictions, actual_returns):
    evaluator.add_prediction(
        predicted=prediction,
        actual=actual,
        metadata={'scheme_code': scheme_code, 'category': category}
    )

# Generate report
report = evaluator.generate_evaluation_report(
    save_path='reports/model_evaluation_report.json'
)

# Create scatter plot
evaluator.plot_prediction_scatter(
    save_path='reports/prediction_scatter.png'
)
```

### 6. Walk-Forward Validation

```python
from evaluation.walk_forward_validation import WalkForwardValidator

# Define your model's prediction function
def model_predict_fn(scheme_code, date):
    # Your model prediction logic here
    # Returns: predicted return (%)
    return your_model.predict(scheme_code, date)

# Create validator
validator = WalkForwardValidator(
    db_path='data/historical_20y.db',
    n_splits=5,
    test_size=90,  # days
    gap=0
)

# Run validation
results = validator.validate_model(
    model_predict_fn=model_predict_fn,
    max_schemes=50
)

# Save results
validator.save_results(results, 'reports/walk_forward_validation.json')
```

---

## 📊 Understanding the Output

### 1. Backtest Report (HTML)

Open `reports/backtest_1year.html` in your browser to see:
- Cumulative returns chart
- Drawdown chart
- Monthly returns heatmap
- Risk metrics table
- Trade statistics

### 2. Data Quality Report (JSON)

```json
{
  "total_schemes": 200,
  "passed_schemes": 185,
  "failed_schemes": 15,
  "pass_rate": 92.5,
  "avg_quality_score": 88.3,
  "failed_scheme_details": [...]
}
```

### 3. Model Evaluation Report

```
MODEL PERFORMANCE EVALUATION REPORT
================================================================================
RMSE:                    4.23%
MAE:                     3.15%
Directional Accuracy:    67.5%
R² Score:                0.72
Spearman Correlation:    0.68
```

**Interpretation:**
- **RMSE < 5%**: Excellent prediction accuracy
- **Directional Accuracy > 65%**: Good at predicting direction
- **R² > 0.7**: Strong correlation between predictions and actuals

### 4. Training Data (JSONL)

Each line contains:
```json
{
  "prompt": "[CONTEXT]\nFund Name: HDFC Top 100...",
  "completion": "Expected 3-month return: 6.5%...",
  "metadata": {
    "scheme_code": "119551",
    "actual_return": 6.47,
    "recommendation": "BUY"
  }
}
```

---

## 🎯 Performance Benchmarks

### Backtesting Targets

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Sharpe Ratio | > 1.0 | > 1.5 | > 2.0 |
| Annualized Return | > 12% | > 15% | > 18% |
| Max Drawdown | < 30% | < 25% | < 20% |
| Sortino Ratio | > 1.5 | > 2.0 | > 2.5 |
| Win Rate | > 55% | > 60% | > 65% |

### Model Evaluation Targets

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| RMSE | < 8% | < 5% | < 3% |
| MAE | < 5% | < 3% | < 2% |
| Directional Accuracy | > 55% | > 65% | > 70% |
| R² Score | > 0.5 | > 0.7 | > 0.8 |

---

## 🔮 Next Steps

### Immediate Actions

1. **Review Generated Reports**
   - Open `reports/backtest_1year.html` to analyze strategy performance
   - Check `reports/data_quality_report.json` for data issues

2. **Use Training Data**
   - Use `data/training_data_enhanced.jsonl` to fine-tune your LLM
   - Adjust the data generator to create more diverse examples

3. **Integrate Your Model**
   - Replace dummy predictions in evaluation scripts with your actual model
   - Run walk-forward validation on your trained model

### Advanced Enhancements

1. **Add More Strategies**
   - Implement momentum-based strategies
   - Add mean-reversion strategies
   - Create risk-parity portfolios

2. **Expand Crisis Scenarios**
   - Add 2008 financial crisis testing (if data available)
   - Test during other market events

3. **Enhance Training Data**
   - Add sentiment data from news sources
   - Include economic indicators
   - Add sector-specific features

4. **Deploy Model**
   - Integrate backtesting with API
   - Create real-time prediction endpoints
   - Build user-facing dashboards

---

## 🐛 Troubleshooting

### Common Issues

**1. "Database not found" error**
```bash
# Make sure database exists
python src/historical_data_collector.py
```

**2. "Insufficient data" errors during backtesting**
- Reduce `num_funds` parameter
- Check that your database has sufficient historical data (at least 2 years)

**3. Memory errors with large datasets**
- Reduce `num_funds` parameter
- Process data in batches
- Increase available system memory

**4. HTML report not generating**
```bash
# Install missing dependencies
pip install quantstats matplotlib
```

---

## 📚 References

- **bt library**: https://pmorissette.github.io/bt/
- **quantstats**: https://github.com/ranaroussi/quantstats
- **Walk-forward validation**: Proper time series cross-validation technique

---

## 📝 Notes

- All timestamps use India Standard Time (IST)
- Risk-free rate assumed at 6.5% (India government bonds)
- NAV data sourced from mfapi.in
- Rebalancing costs not included in backtests (add if needed)

---

## 🤝 Contributing

To extend this implementation:
1. Add new strategies in `backtesting/mf_backtest_engine.py`
2. Enhance training data in `src/advanced_training_data_generator.py`
3. Add new metrics in `evaluation/model_evaluator.py`

---

**Implementation Date:** January 2025
**Based on:** future_idea.txt Phase 1 & 2
**Status:** ✅ Complete and Ready for Use
