# Quick Start Guide

## 30-Second Setup


```
# 1. Create virtual environment
`python -m venv venv`

# 2. Activate it
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
streamlit run app.py
```


That's it! Open your browser to http://localhost:8501

## First-Time Usage
---
## 🖥️ Streamlit Pages Overview
| Page                     | Purpose                                                |
| ------------------------ | ------------------------------------------------------ |
| 🏠 **Home**              | Overview, features & workflow                          |
| 🔍 **Single Prediction** | Real-time fraud detection                              |
| 📊 **Batch Analysis**    | CSV upload → Prediction summary & charts               |
| 📈 **Model Performance** | Metrics, ROC-AUC, confusion matrix, feature importance |
| 📚 **Dataset Info**      | Kaggle dataset details & feature descriptions          |
| ℹ️ **About**             | Architecture, methodology & credits                    |

---

## Sample Transaction

Try this legitimate transaction:
- Amount: $150.00
- Age: 35 years
- Hour: 14 (2:00 PM)
- Merchant: New York
- Customer: New York

Expected: LOW RISK (Legitimate)

Try this suspicious transaction:
- Amount: $5000.00
- Age: 22 years
- Hour: 3 (3:00 AM)
- Merchant: Different city
- Customer: Different location

Expected: HIGH RISK (Fraudulent)

## Common Commands

```
# Update dependencies
pip install --upgrade -r requirements.txt

# Retrain model
python scripts/train_model.py

# Run tests
pytest tests/

# Format code
black *.py backend/*.py

# Check code style
pylint *.py backend/*.py
```

## Need Help?

1. Read **README.md** - Comprehensive documentationp

## Performance Tips

- First run loads model (takes 1-2 minutes)
- Subsequent runs are instant
- Batch processing is fastest for multiple transactions
- Use "Load Sample Data" for quick testing

---

🎯 Start detecting fraud in under 30 seconds!
