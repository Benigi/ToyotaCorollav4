# Toyota Corolla Valuation Tool

Une application Streamlit pour estimer les prix de revente d'une Toyota Corolla.
Ce repository contient le front-end `app.py` et le module d'entraînement `model_trainer.py`.

## 🧩 Structure actuelle du dépôt

- `app.py` : application Streamlit principale
- `model_trainer.py` : code de construction, d'entraînement et d'évaluation du modèle
- `requirements.txt` : dépendances Python
- `data/` : dossier attendu pour les données et modèles
  - `data/models/` : modèles sérialisés
  - `data/raw/` : données source (CSV)

## 🚀 Installation et exécution

### 1. Créer et activer l'environnement
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer l'application
```bash
streamlit run app.py
```

- Si `data/models/rf_model.pkl` est absent, `app.py` génère automatiquement un modèle de repli.
- Le dossier `data/models/` est créé automatiquement si nécessaire.

## 📌 Notes importantes

- `model_trainer.py` ne contient pas de chargeur de données intégré dans ce dépôt.
- Pour entraîner un modèle personnalisé, charger vos données dans un DataFrame pandas puis appeler :

```python
from model_trainer import train_model
trainer, train_info, eval_metrics = train_model(X, y)
trainer.save('data/models/rf_model.pkl')
```

- `app.py` utilise un modèle de repli synthétique si aucun modèle pré-entraîné n'est disponible.

## 🧪 Vérification rapide

- Les fichiers Python sont compilables : `python3 -m py_compile app.py model_trainer.py`
- Le module principal s'importe correctement : `python3 -c "import app, model_trainer"`

## 🛠️ Détails

### `app.py`
- affichage Streamlit interactif
- chargement du modèle depuis `data/models/rf_model.pkl`
- génération d'un modèle de repli si le fichier du modèle manque
- affichage de graphiques Plotly

### `model_trainer.py`
- classe `ModelTrainer` pour construire, entraîner, évaluer et sauvegarder un RandomForestRegressor
- fonction `train_model(X, y)` pour entraîner et évaluer un modèle sur un DataFrame pandas

## 📋 Dépendances clés

- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `joblib`
- `scikit-learn`

## 🧩 Fichiers attendus

- `data/models/rf_model.pkl` (facultatif, créé automatiquement)
- `data/raw/ToyotaCorolla.csv` (optionnel pour le support de comparaison de marché)

## Résumé

Le dépôt est désormais cohérent : l'application peut démarrer même sans modèle préexistant, et `model_trainer.py` est utilisable pour entraîner un modèle personnalisé. Si tu souhaites, je peux aussi ajouter un petit script `train.py` pour générer proprement `rf_model.pkl` à partir d'un CSV existant.
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
