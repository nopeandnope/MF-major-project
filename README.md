# AI-Powered Mutual Fund Recommendation System

A quantitative advisory system for Indian mutual funds: a deterministic backtesting and portfolio engine, plus a fine-tuned LLM that explains its recommendations.

**Co-built by Ashwin Chellissaryil Shaji and Siraj Fahmida Senthil** as our B.Tech major project at NITK Surathkal (guide: Dr. Deepu Vijayasenan), 2025-26.

## What it does

- **Data:** NAV history for Indian open-ended mutual funds (HDFC, ICICI, Axis, Kotak, SBI; equity, debt, hybrid and ELSS) from mfapi (AMFI-sourced), with NIFTY 50 / SENSEX benchmarks from Yahoo Finance. **19 schemes pass validation, 18 of them with 10-15 years of history.**
- **Data quality:** completeness above 95%, daily-move sanity bounds, z-score outlier filter.
- **Fund scoring:** multi-factor - rolling returns, Sortino, maximum drawdown, 95% VaR, beta, expense ratio, tracking error.
- **Five backtested strategies:** Conservative, Balanced, Aggressive, Risk Parity (inverse-volatility weights) and 6-month Momentum. The API serves three strategy options per request.
- **Risk management:** position sizing (equal weight, risk parity, fractional Kelly), multi-level stop-loss and a volatility circuit breaker; 0.1% transaction cost in backtests.
- **Analysis:** 10,000-path bootstrap Monte Carlo, 3-state Hidden Markov Model for regime detection, historical stress tests (2020 COVID crash, 2022).
- **LLM layer:** Google Gemma-2B fine-tuned with LoRA (r=32, 1.54% trainable parameters) on 25,000 generated examples. **The LLM only writes explanations; every number and allocation comes from the deterministic engine.**
- **Serving:** FastAPI backend on SQLite.

## Results - read with care

Balanced strategy, 2014-2024: 14.23% annualised return, Sharpe 1.45 (NIFTY 50: 0.82), Sortino 2.18, maximum drawdown 10.15%.

**These are in-sample results.** Strategy choices were made on the same period they are measured on, so they overstate what to expect live. A re-test under **walk-forward validation** (expanding/rolling windows with a train/test gap, `evaluation/walk_forward_validation.py`) with realistic transaction costs is in progress; results will be added here when done.

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Install, collect data and train
python setup_and_train_complete.py
#   --quick           dependencies only (~5 min)
#   --skip-data       reuse existing data
#   --skip-training   reuse an existing model
#   --max-funds 50    collect fewer funds

# 3. Start the server, then open http://localhost:8000
python run_server.py
```

Training needs a Hugging Face token with access to `google/gemma-2b`, saved in a `.env` file as `HUGGINGFACE_HUB_TOKEN=...`. Requirements: Python 3.11+, 8 GB RAM minimum (16 GB recommended), ~10 GB free disk.

## API

`POST /api/recommend`

```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"amount": 100000, "horizon": 5, "risk_tolerance": "Moderate", "financial_goal": "Wealth Creation", "age": 30}'
```

## Tech

Python, pandas, NumPy, SciPy, scikit-learn, LightGBM, `bt` / `quantstats` for backtest analytics, PyTorch + Hugging Face Transformers/PEFT, FastAPI, SQLite, aiohttp.

## Disclaimer

Academic project. Not investment advice. Past performance does not guarantee future results; consult a SEBI-registered adviser before investing.
