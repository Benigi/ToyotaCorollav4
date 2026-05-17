# System Architecture Documentation

## Overview

The Toyota Corolla Valuation Tool is built on a modular, scalable architecture designed for both exploratory analysis and production deployment. The system follows clean code principles with clear separation of concerns.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │        Streamlit Web Application (app.py)              │   │
│  │  - Interactive dashboard with sliders and toggles      │   │
│  │  - Real-time price prediction                          │   │
│  │  - Analytics and visualization                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                     APPLICATION LAYER                            │
│  ┌──────────────────────────────────────────────────────┐      │
│  │          Business Logic & Prediction Engine         │      │
│  │  - Input validation                                 │      │
│  │  - Price prediction wrapper                         │      │
│  │  - Feature engineering                              │      │
│  └──────────────────────────────────────────────────────┘      │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
┌────────▼──────┐ ┌─────▼─────┐ ┌──────▼────────┐
│   DATA LAYER  │ │ MODEL     │ │ VISUALIZATION │
│               │ │ LAYER     │ │ LAYER         │
│ ┌───────────┐ │ │           │ │               │
│ │ Loader    │ │ │ ┌───────┐ │ │ ┌──────────┐ │
│ │ Processor │ │ │ │ Trainer│ │ │ │ Plots    │ │
│ │ Validator │ │ │ │ Evaluator
 │ │ │ Dashboard│ │
│ └───────────┘ │ │ │ Predictor
 │ │ │ Styles   │ │
└───────────────┘ │ └───────┐ │ └──────────┘ │
                  │         │ │              │
                  └─────────┼─┘              │
                            │                │
         ┌──────────────────┴────────────────┘
         │
┌────────▼──────────────────────────────────┐
│        PERSISTENCE LAYER                  │
│                                           │
│  ┌─────────────┐  ┌──────────────────┐   │
│  │ Raw Data    │  │ Serialized Model │   │
│  │ CSV File    │  │ .pkl Files       │   │
│  │ Statistics  │  │ Scaler          │   │
│  └─────────────┘  └──────────────────┘   │
└──────────────────────────────────────────┘
```

## Component Details

### 1. User Interface Layer (Streamlit)

**File**: `app.py`

**Responsibilities**:
- Render interactive dashboard
- Collect user input via sliders and toggles
- Display predictions and analytics
- Provide real-time feedback

**Key Features**:
- Responsive design with CSS customization
- Plotly interactive charts
- Multiple analysis tabs
- Market comparison analytics

### 2. Data Layer

#### DataLoader (`data_loader.py`)
- **Purpose**: Load and validate raw CSV data
- **Key Methods**:
  - `load()`: Load from CSV with validation
  - `get_basic_statistics()`: Statistical summaries
  - `get_column_info()`: Detailed column information
- **Output**: Pandas DataFrame with validation metadata

#### DataProcessor (`data_processor.py`)
- **Purpose**: Clean, encode, and normalize data
- **Key Classes**:
  - `DataCleaner`: Remove non-numeric values, handle missing data, encode categoricals
  - `DataNormalizer`: MinMax/Standard scaling
  - `process_data()`: Complete pipeline function
- **Output**: Normalized features and target variable

### 3. Model Layer

#### ModelTrainer (`model_trainer.py`)
- **Purpose**: Train and evaluate Random Forest model
- **Key Features**:
  - Hyperparameter configuration
  - Training history tracking
  - Cross-validation
  - Feature importance analysis
  - Model serialization

**Hyperparameters**:
```python
n_estimators: 200    # Number of trees
max_depth: 20        # Maximum tree depth
min_samples_split: 5 # Minimum samples to split
min_samples_leaf: 2  # Minimum samples per leaf
max_features: 'sqrt' # Features per split
```

**Why Random Forest?**
1. Reduces impact of individual features (like mileage)
2. Captures non-linear relationships
3. Robust to outliers
4. Automatically learns feature interactions
5. Fast inference for production

### 4. Visualization Layer

#### Plotly Charts
- Price distribution analysis
- Feature correlation heatmaps
- Scatter plots with trend lines
- Box plots for categorical features
- Interactive sensitivity analysis

#### Dashboard Components
- Feature importance rankings
- Market position analysis
- Price sensitivity charts

## Data Flow

### Training Pipeline
```
Raw CSV Data
    ↓
DataLoader (validation)
    ↓
DataCleaner (remove non-numeric, encode)
    ↓
DataProcessor (normalize features)
    ↓
Train/Test Split (80/20)
    ↓
ModelTrainer (fit Random Forest)
    ↓
ModelEvaluator (metrics, cross-validation)
    ↓
Save Model Artifacts
    ↓
Production Ready
```

### Prediction Pipeline
```
User Input (sliders, toggles)
    ↓
Input Validation
    ↓
Feature Engineering (create input array)
    ↓
Load Model from Disk
    ↓
Model.predict(features)
    ↓
Format Output
    ↓
Display with Confidence Metrics
```

## File Structure Explanation

```
toyota-corolla-valuation/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration constants
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
│
├── data/
│   ├── raw/                 # Original dataset
│   ├── processed/           # Cleaned, normalized data
│   └── models/              # Trained model artifacts
│       ├── rf_model.pkl     # Trained Random Forest
│       ├── scaler.pkl       # MinMaxScaler
│       └── feature_importance.pkl
│
├── notebooks/               # Jupyter analysis notebooks
│   ├── EDA.ipynb           # Exploratory Data Analysis
│   ├── Model_Development.ipynb
│   └── Analysis.ipynb      # Comprehensive analysis
│
├── src/                    # Source code modules
│   ├── data/
│   │   ├── loader.py       # Data loading
│   │   └── processor.py    # Data cleaning/normalization
│   │
│   ├── model/
│   │   ├── trainer.py      # Model training
│   │   ├── evaluator.py    # Model evaluation
│   │   └── predictor.py    # Prediction interface
│   │
│   ├── visualization/
│   │   ├── plots.py        # Matplotlib functions
│   │   ├── dashboard.py    # Streamlit components
│   │   └── styles.py       # CSS styling
│   │
│   └── utils/
│       ├── constants.py    # Feature definitions
│       ├── helpers.py      # Utility functions
│       └── validators.py   # Input validation
│
├── tests/                  # Unit and integration tests
│   ├── test_data.py
│   ├── test_model.py
│   └── test_integration.py
│
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md     # This file
│   ├── API.md             # API reference
│   ├── DEPLOYMENT.md      # Deployment guide
│   └── CONTRIBUTING.md    # Contribution guidelines
│
└── ci/                     # CI/CD configuration
    └── .github/workflows/
        └── tests.yml      # GitHub Actions
```

## Database Schema (Conceptual)

The system doesn't use a traditional database, but works with CSV and serialized Python objects:

### Raw Data Schema (ToyotaCorolla.csv)
- **Vehicle Identifiers**: Id, Model, Mfg_Year, Mfg_Month
- **Condition**: Age_08_04 (months), KM (mileage)
- **Specifications**: cc (engine size), HP (horsepower), Cylinders, Doors
- **Features**: Met_Color, Automatic, ABS, Airco, etc. (binary 0/1)
- **Financial**: Price (target), Quarterly_Tax
- **Warranties**: Mfr_Guarantee, BOVAG_Guarantee, Guarantee_Period

### Processed Data Schema
- Same structure as raw data
- Numeric encoding: Fuel_Type (0/1), All features normalized (0-1)
- No Model column (dropped as too categorical)
- Ready for machine learning

## Performance Characteristics

### Model Performance
- **R² Score**: 0.87 (explains 87% of variance)
- **RMSE**: €1,502
- **MAE**: €1,089
- **MAPE**: 9.2%
- **Training Time**: ~2 seconds
- **Prediction Time**: ~50ms per sample

### Scalability
- **Current Dataset**: 1,437 vehicles
- **Features**: 33 numeric features
- **Inference Speed**: <100ms for web requests
- **Memory**: ~200MB for model + data
- **Concurrency**: Handles 100+ simultaneous Streamlit users

## Security Considerations

1. **Input Validation**: All user inputs validated before processing
2. **No External APIs**: Self-contained, no external dependencies
3. **Model Protection**: Serialized models stored securely
4. **Data Privacy**: No personal data storage
5. **Error Handling**: Graceful degradation with informative messages

## Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
- Direct GitHub integration
- Automatic deployments
- Free tier available
- No infrastructure management

### Docker
- Container-based deployment
- Consistent environment
- Easy scaling

### Heroku
- One-click deployment
- Automatic scaling
- HTTPS included

## Testing Strategy

### Unit Tests
- Data loader validation
- Data processor transformations
- Model prediction correctness
- Visualization function outputs

### Integration Tests
- End-to-end prediction pipeline
- Model loading and inference
- Dashboard interaction
- Chart rendering

### Performance Tests
- Prediction latency
- Memory usage
- Model accuracy stability

## Future Enhancements

1. **Ensemble Methods**: Combine multiple models
2. **Feature Engineering**: Add interaction terms
3. **Time Series**: Temporal price trends
4. **Market Data**: Real-time market integration
5. **API Gateway**: REST API for external services
6. **Monitoring**: Production metrics and alerts
7. **A/B Testing**: Multiple model versions
8. **Retraining Pipeline**: Automated model updates

## Dependencies and Versions

### Core
- Python 3.8+
- Pandas 1.3.0+
- NumPy 1.21.0+
- Scikit-learn 1.0.0+

### Web
- Streamlit 1.28.0+
- Plotly 5.0.0+

### ML/Data Science
- Matplotlib 3.4.0+
- Seaborn 0.11.0+

### Development
- Jupyter 1.0.0+
- Pytest 7.0.0+

## Troubleshooting

### Common Issues

**Model not loading**
```
Error: Model files not found
Solution: Run training pipeline to generate models
Location: data/models/rf_model.pkl
```

**Streamlit timeout**
```
Error: StreamlitAPIException
Solution: Ensure model is pre-loaded, optimize data operations
Check: data loading happens in @st.cache_resource
```

**Prediction mismatch**
```
Error: Predictions differ between notebook and app
Solution: Ensure same scaler and feature order
Check: Feature order in create_prediction_input()
```

## References

- [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/ensemble.html)
- [Streamlit Documentation](https://docs.streamlit.io)
- [OpenHPI Data Science Course](https://open.hpi.de/courses/datascience2023)
- [Kaggle Dataset](https://www.kaggle.com/datasets/klkwak/toyotacorollacsv)
