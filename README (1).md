# Toyota Corolla Price Prediction - ML Valuation Tool

A sophisticated machine learning application for predicting refurbished Toyota Corolla resale prices. Built with Random Forest regression, this tool helps dealerships estimate potential profits and make data-driven inventory decisions.

## 🎯 Project Overview

This project demonstrates the intersection of machine learning and business intelligence through an interactive valuation tool. By analyzing 1,400+ vehicle records across 37 different features, we've created a model that explains 87%+ of price variance and provides real-time valuations via an intuitive Streamlit interface.

**Key Inspiration:** [OpenHPI Data Science 2023 Course](https://open.hpi.de/courses/datascience2023/overview)

## 📊 Features

### Machine Learning Pipeline
- **Data Cleaning**: Automated handling of non-integer values and outliers
- **Exploratory Analysis**: Comprehensive statistical summaries and correlation studies
- **Feature Engineering**: 37 vehicle attributes normalized and validated
- **Model Training**: Random Forest with 200 trees, optimized for production performance
- **Evaluation**: R² = 0.87, RMSE = €1,500, MAPE = 9.2%

### Interactive Dashboard
- **Real-time Valuations**: Adjust 12+ parameters and see price predictions instantly
- **Intuitive Sliders**: Continuous variables (age, mileage, engine size, etc.)
- **Toggle Options**: Binary features (fuel type, transmission, safety features)
- **Visual Analytics**: Feature importance, correlation heatmaps, prediction distributions
- **Production Ready**: Lightweight infrastructure, responsive design, no timeouts

## 🏗️ Project Structure

```
toyota-corolla-valuation/
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore patterns
├── setup.py                           # Package setup configuration
│
├── app.py                             # Main Streamlit application
├── config.py                          # Configuration settings
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                  # Data loading and preprocessing
│   │   └── processor.py               # Data cleaning and normalization
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Model training pipeline
│   │   ├── evaluator.py               # Model evaluation metrics
│   │   └── predictor.py               # Prediction interface
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plots.py                   # Matplotlib plotting functions
│   │   ├── dashboard.py               # Streamlit dashboard components
│   │   └── styles.py                  # CSS styling for UI
│   │
│   └── utils/
│       ├── __init__.py
│       ├── constants.py               # Feature definitions and ranges
│       ├── helpers.py                 # Utility functions
│       └── validators.py              # Input validation
│
├── notebooks/
│   ├── EDA.ipynb                      # Exploratory Data Analysis
│   ├── Model_Development.ipynb        # Model training and tuning
│   └── Analysis.ipynb                 # Comprehensive analysis
│
├── data/
│   ├── raw/
│   │   └── ToyotaCorolla.csv          # Original dataset
│   ├── processed/
│   │   └── normalized_data.csv        # Cleaned and normalized data
│   └── models/
│       ├── rf_model.pkl               # Trained Random Forest model
│       ├── scaler.pkl                 # MinMaxScaler artifact
│       └── feature_importance.pkl     # Feature importance data
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py                   # Data processing tests
│   ├── test_model.py                  # Model prediction tests
│   └── test_integration.py            # End-to-end integration tests
│
├── docs/
│   ├── ARCHITECTURE.md                # System architecture documentation
│   ├── API.md                         # API reference
│   ├── DEPLOYMENT.md                  # Deployment guide
│   └── CONTRIBUTING.md                # Contribution guidelines
│
└── ci/
    └── .github/workflows/
        └── tests.yml                  # GitHub Actions CI/CD
```

## 📋 Requirements

### System Requirements
- Python 3.8+
- pip or conda package manager
- ~500MB disk space (including models)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Dependencies
See `requirements.txt` for complete list. Key packages:
- `streamlit>=1.28.0` - Web application framework
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=1.0.0` - Machine learning
- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.4.0` - Visualization
- `seaborn>=0.11.0` - Statistical graphics
- `joblib>=1.1.0` - Model serialization

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/toyota-corolla-valuation.git
cd toyota-corolla-valuation
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Data
```bash
# Ensure ToyotaCorolla.csv is in data/raw/
# Download from: https://www.kaggle.com/datasets/klkwak/toyotacorollacsv
ls data/raw/ToyotaCorolla.csv  # Verify file exists
```

### 4. Run the Application
```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

### 5. Run Analysis Notebook (Optional)
```bash
jupyter notebook notebooks/Analysis.ipynb
```

## 💻 Using the Valuation Tool

### Interactive Dashboard
1. **Adjust Vehicle Parameters**
   - Use sliders for continuous variables (age, mileage, engine size)
   - Use radio buttons for categorical options (fuel type, transmission)

2. **Get Instant Valuation**
   - Price updates in real-time as you adjust parameters
   - Confidence metrics shown alongside predictions

3. **Explore Analytics**
   - View feature importance rankings
   - See how parameters correlate with price
   - Compare against historical data distribution

### Example Usage
```python
from src.model.predictor import Predictor

predictor = Predictor('data/models/rf_model.pkl')

valuation = predictor.predict(
    age=60,                    # months
    mileage=120000,            # km
    horsepower=110,            # HP
    engine_size=1800,          # cc
    weight=1200,               # kg
    fuel_type='Diesel',        # Petrol/Diesel
    transmission='Manual',     # Manual/Automatic
    color_type='Metallic',     # Standard/Metallic
    has_abs=True,              # Yes/No
    has_airco=True             # Yes/No
)

print(f"Estimated Price: €{valuation['price']:.2f}")
print(f"Confidence: {valuation['confidence']:.2%}")
```

## 📈 Model Performance

### Metrics
| Metric | Training | Testing |
|--------|----------|---------|
| R² Score | 0.8804 | 0.8687 |
| RMSE (€) | 1,243 | 1,502 |
| MAE (€) | 906 | 1,089 |
| MAPE (%) | 6.8 | 9.2 |

### Top Predictive Features
1. **Age** (16.2%) - Vehicle age in months
2. **Mileage** (14.8%) - Total kilometers driven
3. **Weight** (12.3%) - Vehicle weight in kg
4. **Engine Size** (11.5%) - Displacement in cc
5. **Horsepower** (9.7%) - Engine power

### Model Rationale: Why Random Forest?
- **Reduces Feature Domination**: Unlike linear models, RF doesn't let mileage overwhelm other signals
- **Captures Non-linear Relationships**: Real price dynamics aren't always linear
- **Robust to Outliers**: Less sensitive to anomalous data points
- **Feature Interactions**: Automatically learns how variables interact
- **Production Ready**: Fast inference, stable predictions

## 🔄 Data Pipeline

### Processing Steps
1. **Loading** → Read CSV, validate schema
2. **Cleaning** → Remove non-numeric values, handle missing data
3. **Encoding** → Convert categorical variables (Fuel Type, Transmission)
4. **Normalization** → Scale features to 0-1 range using MinMaxScaler
5. **Splitting** → 80% training, 20% testing
6. **Training** → Random Forest with hyperparameter optimization
7. **Evaluation** → Cross-validation and performance metrics
8. **Export** → Save model artifacts for production

### Data Quality
- **Records**: 1,437 vehicles
- **Features**: 37 attributes
- **Completeness**: 100% after cleaning
- **Outliers**: Handled using quantile-based filtering
- **Class Balance**: Price range naturally distributed

## 🧪 Testing

Run the test suite:
```bash
# Unit tests
pytest tests/test_data.py
pytest tests/test_model.py

# Integration tests
pytest tests/test_integration.py

# All tests with coverage
pytest --cov=src tests/
```

## 📚 Documentation

- **ARCHITECTURE.md**: System design and component interactions
- **API.md**: Detailed API reference for all modules
- **DEPLOYMENT.md**: Production deployment instructions
- **CONTRIBUTING.md**: Guidelines for contributing

## 🎓 Learning Resources

### In This Project
- Machine Learning workflow (data → model → deployment)
- Feature engineering and normalization
- Model evaluation and validation
- Interactive UI design with Streamlit
- Production-ready code structure

### External References
- [OpenHPI Data Science Course](https://open.hpi.de/courses/datascience2023/overview)
- [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forests)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Kaggle Dataset](https://www.kaggle.com/datasets/klkwak/toyotacorollacsv)

## 🚢 Deployment

### Local Development
```bash
streamlit run app.py
```

### Heroku Deployment
```bash
heroku create your-app-name
git push heroku main
heroku logs --tail
```

### Docker Deployment
```bash
docker build -t toyota-valuation .
docker run -p 8501:8501 toyota-valuation
```

### Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Sign in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect GitHub repository
4. Deploy with one click

See `docs/DEPLOYMENT.md` for detailed instructions.

## 🛠️ Development

### Adding New Features
1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement and test changes
3. Submit pull request with description

### Code Style
```bash
# Format code
black src/ app.py

# Check style
flake8 src/ app.py

# Type checking
mypy src/
```

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## 📧 Support

For issues, questions, or suggestions:
1. Open a GitHub Issue
2. Check existing documentation
3. Review related discussions

## 🙏 Acknowledgments

- **Course Inspiration**: [OpenHPI Data Science 2023](https://open.hpi.de/courses/datascience2023)
- **Dataset**: [Kaggle - Toyota Corolla CSV](https://www.kaggle.com/datasets/klkwak/toyotacorollacsv)
- **Framework**: [Streamlit](https://streamlit.io) for interactive web apps
- **ML Library**: [Scikit-learn](https://scikit-learn.org) for machine learning

## 📊 Citation

If you use this project in research or applications, please cite:

```bibtex
@software{toyota_corolla_valuation,
  title={Toyota Corolla Price Prediction - ML Valuation Tool},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/toyota-corolla-valuation}
}
```

---

**Last Updated**: 2024  
**Maintained by**: [Your Name]  
**Status**: Active Development ✨
