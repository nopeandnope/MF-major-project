# 🚀 AI-Powered Mutual Fund Advisor

AI-powered investment recommendation system using Google Gemma-2B with 20-year historical data analysis.

## ⚡ Quick Start (3 Commands)

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Run automated setup (everything in one command!)
python setup_and_train_complete.py

# 3. Start the server
python run_server.py
```

That's it! Open http://localhost:8000 in your browser.

## 🎯 What Does This Do?

- **Collects** 20-year historical data for 200+ mutual funds
- **Trains** Google Gemma-2B model with LoRA fine-tuning
- **Provides** AI-powered personalized investment recommendations
- **Analyzes** risk, returns, and portfolio optimization

## 📊 Features

✅ 20-year historical NAV data (3.6M+ records)
✅ Google Gemma-2B model (2B parameters)
✅ LoRA fine-tuning (1.54% trainable params)
✅ 25,000 training examples
✅ Risk-adjusted portfolio construction
✅ Tax optimization (ELSS)
✅ SIP and lumpsum planning
✅ Real-time recommendations via API

## 🛠️ Setup Options

### Full Setup (Recommended)
```bash
python setup_and_train_complete.py
```
Time: 2-5 hours | Does: Everything

### Quick Setup (Skip Training)
```bash
python setup_and_train_complete.py --quick
```
Time: 5 minutes | Does: Dependencies only

### Custom Setup
```bash
# Skip data collection (use existing)
python setup_and_train_complete.py --skip-data

# Skip training (use existing model)
python setup_and_train_complete.py --skip-training

# Collect less data (faster)
python setup_and_train_complete.py --max-funds 50
```

## 📁 Project Structure

```
MF_Advisor_Project/
├── setup_and_train_complete.py  ⭐ One-command setup
├── run_server.py                → Start web server
├── train_enhanced_model.py      → Train model
├── PROJECT_SETUP_GUIDE.txt      → Detailed guide
│
├── api/
│   └── app.py                   → FastAPI application
├── src/
│   ├── enhanced_20_year_collector.py  → Data collection
│   └── inference_stable.py      → AI inference
├── web/
│   ├── index.html               → Web interface
│   └── static/app.js            → Frontend logic
│
├── data/
│   └── historical_20y.db        → 20-year database
├── models/
│   └── enhanced_20y_advisor/    → Trained model
└── logs/                        → All logs
```

## 🔧 Requirements

- **Python**: 3.11+
- **RAM**: 8GB minimum (16GB recommended)
- **Disk**: 10GB free space
- **Internet**: Required for data collection

## 📖 API Usage

### Start Server
```bash
python run_server.py
```

### Get Recommendations (POST /api/recommend)
```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 100000,
    "horizon": 5,
    "risk_tolerance": "Moderate",
    "financial_goal": "Wealth Creation",
    "age": 30
  }'
```

### Response
```json
{
  "recommendations": [
    {
      "fund_name": "HDFC Top 100 Fund",
      "category": "Large Cap",
      "allocation_percentage": 40,
      "amount": 40000,
      "expected_return": 12.5,
      "risk_level": "Moderate"
    }
  ],
  "total_expected_return": 11.8,
  "portfolio_summary": "Balanced growth portfolio..."
}
```

## 🐛 Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt
```

### Out of Memory
Edit `train_enhanced_model.py`:
- Line 37: Change `batch_size` to 1
- Line 38: Increase `gradient_accumulation_steps` to 32

### Hugging Face Token
1. Visit https://huggingface.co/google/gemma-2b
2. Request access and accept terms
3. Generate token at https://huggingface.co/settings/tokens
4. Create `.env` file: `HUGGINGFACE_HUB_TOKEN=your_token_here`

### Port 8000 In Use
Edit `run_server.py` and change port to 8001

## 📊 Training Progress

Monitor training:
```bash
# Windows
type logs\enhanced_model_training.log

# Linux/Mac
tail -f logs/enhanced_model_training.log
```

## 🎓 Manual Steps (Alternative)

If you prefer manual control:

```bash
# 1. Setup environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Collect data (15-45 min)
python -c "import asyncio; from src.enhanced_20_year_collector import Enhanced20YearCollector; collector = Enhanced20YearCollector(); asyncio.run(collector.collect_20_year_data(max_funds=200))"

# 3. Train model (1-4 hours)
python train_enhanced_model.py

# 4. Start server
python run_server.py
```

## 🚀 Performance Tips

**Faster Setup:**
- Use `--quick` flag
- Use `--max-funds 50`
- Reduce training examples in `train_enhanced_model.py`

**Better Accuracy:**
- Use `--max-funds 500`
- Increase `num_epochs` to 6-8
- Increase training examples to 50,000
- Use GPU (automatic detection)

## 📜 License

This project is for educational purposes. Always consult SEBI-registered financial advisors before making investment decisions.

## 🤝 Contributing

This is an academic project. Feel free to fork and modify for your needs.

## 📞 Support

For detailed documentation, see `PROJECT_SETUP_GUIDE.txt`

## ⚠️ Disclaimer

This is an AI-powered recommendation system based on historical data. Past performance does not guarantee future results. Always consult with qualified financial advisors before making investment decisions.

---

**Made with ❤️ using Google Gemma-2B and Python**

**Quick Start**: `python setup_and_train_complete.py`
**Start Server**: `python run_server.py`
**Access**: http://localhost:8000

