"""Test bt Result object attributes"""
import bt
import pandas as pd
import numpy as np

# Create simple test data
dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
data = pd.DataFrame({
    'asset1': np.random.randn(len(dates)).cumsum() + 100,
    'asset2': np.random.randn(len(dates)).cumsum() + 100
}, index=dates)

# Create simple strategy
strategy = bt.Strategy('test', [
    bt.algos.RunMonthly(),
    bt.algos.SelectAll(),
    bt.algos.WeighEqually(),
    bt.algos.Rebalance()
])

# Run backtest
test = bt.Backtest(strategy, data)
result = bt.run(test)

# Check attributes
print("Result attributes:")
print(dir(result))

print("\nAvailable data:")
if hasattr(result, 'prices'):
    print(f"  prices: {type(result.prices)}")
if hasattr(result, 'returns'):
    print(f"  returns: {type(result.returns)}")
if hasattr(result, 'stats'):
    print(f"  stats: {type(result.stats)}")

# Try to access data
print("\nTrying to access prices:")
print(result.prices.head())

print("\nChecking for returns calculation:")
strategy_name = list(result.prices.columns)[0]
prices = result.prices[strategy_name]
returns = prices.pct_change()
print(f"Calculated returns shape: {returns.shape}")
