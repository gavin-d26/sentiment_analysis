# Sentiment Analysis
IMDB reviews sentiment classification using Logistic Regression and LSTM models.

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Run Logistic Regression model
python logistic_regression.py

# Run LSTM model
python lstm.py
```

## Output
Both models generate results in the project directory:
- Model performance metrics (JSON)
- Learning curves (PNG)
- Validation predictions (CSV)
- Best hyperparameters (TXT)
