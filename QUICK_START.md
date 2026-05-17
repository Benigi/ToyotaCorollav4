# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Python 3.8 or higher
- pip or conda
- Git (optional, for cloning)
- 500MB disk space

---

## Step 1: Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/toyota-corolla-valuation.git
cd toyota-corolla-valuation

# Create virtual environment
python -m venv venv

# Activate environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2: Prepare Data & Models

### Option A: Use Pre-trained Model (Fastest)

```bash
# Directory structure should be:
# data/
# ├── raw/
# │   └── ToyotaCorolla.csv
# ├── models/
# │   ├── rf_model.pkl
# │   └── scaler.pkl
# └── processed/

# Models are included in the repository
# Just verify files exist:
ls data/raw/ToyotaCorolla.csv
ls data/models/rf_model.pkl
```

### Option B: Train Your Own Model

```bash
# Run the Jupyter notebook
jupyter notebook notebooks/Analysis.ipynb

# Or run the training script
python train_model.py

# This will:
# 1. Load ToyotaCorolla.csv
# 2. Clean and preprocess data
# 3. Train Random Forest model
# 4. Save model artifacts
# 5. Generate evaluation report
```

---

## Step 3: Run the Application

```bash
# Start Streamlit app
streamlit run app.py

# The app will open at:
# http://localhost:8501
```

---

## Step 4: Use the Dashboard

### Adjust Vehicle Parameters
1. **Age** - Drag slider (0-80 months)
2. **Mileage** - Drag slider (0-300,000 km)
3. **Horsepower** - Set engine power (50-200 HP)
4. **Engine Size** - Set displacement (1300-2200 cc)
5. **Weight** - Adjust vehicle weight (900-1400 kg)
6. **Quarterly Tax** - Set tax amount (€0-300)
7. **Fuel Type** - Select Petrol or Diesel
8. **Transmission** - Choose Manual or Automatic
9. **Color** - Standard or Metallic
10. **Safety/Comfort** - Toggle ABS and A/C

### Get Price Prediction
- Price updates **in real-time** as you adjust parameters
- View confidence metrics and model statistics
- Explore analytics tabs for deeper insights

---

## Exploring the Jupyter Notebook

```bash
# Launch Jupyter
jupyter notebook

# Open: notebooks/Analysis.ipynb

# Sections:
# 1. Data Loading & EDA
# 2. Data Cleaning & Preprocessing
# 3. Exploratory Analysis with Visualizations
# 4. Model Training (Random Forest)
# 5. Model Evaluation & Performance
# 6. Interactive Valuation Dashboard
# 7. Model Export & Summary
```

### Running Cells

Click "Run All" to execute entire notebook, or:
- `Shift + Enter` to run current cell
- `Ctrl + A` then `Ctrl + Enter` to run all
- Click play button for individual cells

---

## File Organization

```
toyota-corolla-valuation/
├── 📓 Toyota_Corolla_Price_Prediction.ipynb
│   └── Complete analysis notebook (run in Jupyter)
│
├── 🌐 app.py
│   └── Streamlit web application
│
├── 📚 README.md
│   └── Full project documentation
│
├── 📋 requirements.txt
│   └── Python dependencies
│
├── 📁 data/
│   ├── raw/
│   │   └── ToyotaCorolla.csv (original dataset)
│   ├── models/
│   │   ├── rf_model.pkl (trained model)
│   │   └── scaler.pkl (normalizer)
│   └── processed/
│       └── normalized_data.csv (processed data)
│
├── 📚 docs/
│   ├── ARCHITECTURE.md (system design)
│   ├── DEPLOYMENT.md (deployment guide)
│   └── CONTRIBUTING.md (contribution rules)
│
└── 🧪 tests/
    └── Unit and integration tests
```

---

## Usage Examples

### Example 1: Budget-Friendly Used Car

```
Age: 72 months (6 years old)
Mileage: 150,000 km
Horsepower: 90 HP
Engine: 1800 cc
Weight: 1200 kg
Tax: €75
Fuel: Petrol
Transmission: Manual
Extras: Standard color, No ABS, No A/C

Expected Price: €8,500 - €9,500
```

### Example 2: Well-Maintained Premium

```
Age: 36 months (3 years old)
Mileage: 50,000 km
Horsepower: 110 HP
Engine: 1600 cc
Weight: 1150 kg
Tax: €100
Fuel: Diesel
Transmission: Automatic
Extras: Metallic, ABS, A/C

Expected Price: €16,000 - €17,500
```

### Example 3: High-Performance Variant

```
Age: 24 months (2 years old)
Mileage: 30,000 km
Horsepower: 192 HP
Engine: 1800 cc VVT-i
Weight: 1185 kg
Tax: €150
Fuel: Petrol
Transmission: Automatic
Extras: Metallic, ABS, A/C

Expected Price: €19,500 - €21,000
```

---

## Troubleshooting

### Problem: "Module not found" error

```bash
# Solution: Install missing package
pip install -r requirements.txt

# Or install specific package
pip install streamlit
```

### Problem: Model files not found

```bash
# Check file exists:
ls -la data/models/rf_model.pkl

# If missing, run training:
python train_model.py

# Or use provided notebook:
jupyter notebook notebooks/Analysis.ipynb
```

### Problem: Port 8501 already in use

```bash
# Use different port:
streamlit run app.py --server.port 8502
```

### Problem: Jupyter notebook not opening

```bash
# Check Jupyter installation
jupyter --version

# Install if needed
pip install jupyter notebook

# Try with explicit port
jupyter notebook --port 8888
```

### Problem: Slow performance

```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/cache

# Or restart app:
# Press Ctrl+C and rerun:
streamlit run app.py
```

---

## Next Steps

### 1. Explore Analytics
- Check feature importance rankings
- Compare your predictions with market data
- Analyze price sensitivity to each feature

### 2. Experiment with Parameters
- Try extreme values to see model behavior
- Identify which features have biggest impact
- Understand dealership profit margins

### 3. Run Analysis Notebook
- See detailed EDA visualizations
- Understand data cleaning steps
- Review model training process
- Examine feature correlations

### 4. Deploy to Production
- Follow DEPLOYMENT.md for Streamlit Cloud
- Get live URL for sharing
- Monitor app performance
- Collect user feedback

### 5. Extend the Project
- Add more vehicle models
- Integrate with dealership database
- Build API for external use
- Add predictive trending

---

## Key Statistics

### Model Performance
- **Accuracy (R²)**: 0.87 (explains 87% of price variance)
- **Error (RMSE)**: €1,502
- **Error (MAE)**: €1,089
- **Accuracy (MAPE)**: 9.2%

### Dataset
- **Vehicles**: 1,437 records
- **Features**: 33 numeric variables
- **Years**: 2002-2004 models
- **Price Range**: €4,350 - €32,500

### Model
- **Algorithm**: Random Forest
- **Trees**: 200
- **Training Time**: ~2 seconds
- **Prediction Time**: <50ms per sample

---

## Common Use Cases

### For Dealership
✅ Quick inventory valuation
✅ Identify overpriced stock
✅ Estimate profit margins
✅ Market comparison analysis

### For Buyer
✅ Check if asking price is fair
✅ Understand value factors
✅ Negotiate better deals
✅ Research market trends

### For Analyst
✅ Study vehicle pricing patterns
✅ Feature importance analysis
✅ Market segmentation
✅ Price trend forecasting

---

## Best Practices

### 1. Data Quality
- Use consistent measurement units
- Verify mileage and age accuracy
- Check for data entry errors
- Handle outliers appropriately

### 2. Model Usage
- Don't rely solely on predictions
- Consider market conditions
- Account for vehicle condition
- Factor in seasonal variations

### 3. Performance
- Cache expensive operations
- Use vectorized operations
- Profile bottlenecks
- Optimize frequently used functions

### 4. Security
- Validate all inputs
- Handle errors gracefully
- Don't expose system info
- Keep dependencies updated

---

## Learning Resources

### Included in Project
- `Toyota_Corolla_Price_Prediction.ipynb` - Full tutorial
- `ARCHITECTURE.md` - System design
- `README.md` - Project overview
- Source code with comments

### External Resources
- [Scikit-learn ML Course](https://scikit-learn.org/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [OpenHPI Data Science](https://open.hpi.de/courses/datascience2023)
- [Kaggle Datasets](https://www.kaggle.com/)

---

## Support & Help

### If Something Breaks
1. Check error message
2. Review troubleshooting section
3. Check GitHub issues
4. Consult documentation
5. Ask in forums/communities

### Getting More Help
- 📖 Read README.md
- 🏗️ Review ARCHITECTURE.md
- 🚀 Check DEPLOYMENT.md
- 💬 GitHub Discussions
- 🌐 Streamlit Community

---

## What's Next?

### Short Term (This Week)
- [ ] Run Streamlit app locally
- [ ] Experiment with sliders
- [ ] Understand predictions
- [ ] Review your results

### Medium Term (This Month)
- [ ] Complete Jupyter notebook
- [ ] Study feature importance
- [ ] Analyze market data
- [ ] Deploy to Streamlit Cloud

### Long Term (This Quarter)
- [ ] Integrate with real data
- [ ] Build API endpoint
- [ ] Monitor predictions
- [ ] Refine and improve model

---

## Quick Reference

| Task | Command |
|------|---------|
| Activate environment | `source venv/bin/activate` |
| Install dependencies | `pip install -r requirements.txt` |
| Run web app | `streamlit run app.py` |
| Run notebook | `jupyter notebook` |
| Run tests | `pytest tests/` |
| Format code | `black src/` |
| Check style | `flake8 src/` |
| Deploy | See DEPLOYMENT.md |

---

## Keyboard Shortcuts

### Streamlit App
- `Ctrl+C` - Stop server
- `R` - Rerun app
- `C` - Clear cache

### Jupyter Notebook
- `Shift+Enter` - Run cell
- `Ctrl+A` then `Ctrl+Enter` - Run all
- `M` - Change to markdown
- `Y` - Change to code

---

## Need Help?

```bash
# Print help
streamlit run app.py --help
jupyter notebook --help

# Check versions
python --version
pip list

# Verify installation
python -c "import streamlit; print(streamlit.__version__)"
```

---

**Ready to start?** Run `streamlit run app.py` and visit `http://localhost:8501` 🎉

Last updated: 2024 | Status: Production Ready ✅
