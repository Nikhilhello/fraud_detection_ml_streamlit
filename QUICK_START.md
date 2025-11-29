# Quick Start Guide

## 30-Second Setup

\`\`\`bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
streamlit run app.py
\`\`\`

That's it! Open your browser to http://localhost:8501

## First-Time Usage

1. **Home Tab**: Review system overview and metrics
2. **Single Prediction Tab**: Enter sample transaction data
3. **Batch Analysis Tab**: Upload CSV with multiple transactions
4. **Model Performance Tab**: See detailed evaluation metrics
5. **Analytics Tab**: View visualizations and statistics
6. **About Tab**: Learn how the system works

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

\`\`\`bash
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
\`\`\`

## Need Help?

1. Read **README.md** - Comprehensive documentation
2. Review **INSTALLATION_GUIDE.md** - Setup help

## Performance Tips

- First run loads model (takes 1-2 minutes)
- Subsequent runs are instant
- Batch processing is fastest for multiple transactions
- Use "Load Sample Data" for quick testing


---

Start detecting fraud in 30 seconds!
