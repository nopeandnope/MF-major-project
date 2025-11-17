# Quick Start Guide - Fast Training Options

## ✅ Fixed Issues
1. **Data collection now returns success status** - Fixed return value issue
2. **Training optimized for speed** - Reduced from 600 hours to minutes

## 🚀 Quick Training Options

### Option 1: Ultra-Fast Training (5-10 minutes)
Best for testing and demos
```bash
python quick_train.py
```

**Settings:**
- Model: TinyLlama-1.1B
- Samples: 50
- LoRA rank: 2
- Time: 5-10 minutes

### Option 2: Optimized Training (10-20 minutes)
Balanced speed and quality
```bash
python src/train_gemma_gpu.py
```

**Settings:**
- Model: microsoft/phi-2
- Samples: 100
- LoRA rank: 4
- Time: 10-20 minutes

### Option 3: Full Setup with Quick Training (15-30 minutes)
Complete pipeline with fast training
```bash
python setup_and_train_complete.py --quick
```

**This will:**
1. ✅ Install dependencies
2. ⏭️ Skip data collection (use existing)
3. ⚡ Quick training mode (5-10 min)

### Option 4: Custom Setup

```bash
# Skip only data collection
python setup_and_train_complete.py --skip-data

# Skip only training
python setup_and_train_complete.py --skip-training

# Use quick training mode
python setup_and_train_complete.py --quick-train

# Limit data collection
python setup_and_train_complete.py --max-funds 50
```

## ⚙️ Training Speed Adjustments

If you want even FASTER training, edit these files:

### For `quick_train.py`:
- Line 30: `self.max_samples = 50` → Change to 20-30 for ultra-fast
- Line 25: `self.num_epochs = 1` → Keep at 1

### For `src/train_gemma_gpu.py`:
- Line 96: `data = data[:100]` → Change to 50 for faster
- Line 62: `self.num_epochs = 1` → Keep at 1

## 📊 Training Modes Comparison

| Mode | Script | Time | Samples | Quality |
|------|--------|------|---------|---------|
| Ultra-Fast | `quick_train.py` | 5-10 min | 50 | Demo |
| Optimized | `train_gemma_gpu.py` | 10-20 min | 100 | Good |
| Full | Original settings | 2-4 hrs | 1000+ | Best |

## 🔧 Troubleshooting

### Training too slow?
1. Reduce samples in the training script
2. Use `quick_train.py` instead
3. Use CPU if GPU has issues

### Out of memory?
1. Reduce `batch_size` in the script
2. Reduce `max_samples`
3. Use `quick_train.py` (uses TinyLlama)

### Data collection fails?
- Now fixed! Returns proper success status
- Use `--skip-data` to use existing data
- Check internet connection for API calls

## 🎯 Recommended Quick Start

For fastest setup (10-15 minutes total):
```bash
python setup_and_train_complete.py --quick
```

This uses existing data and quick training mode!
