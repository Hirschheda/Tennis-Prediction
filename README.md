# Tennis Match Prediction\n\nThis project predicts ATP tennis match outcomes using historical data, feature engineering, and machine learning.\n\n## Project Structure\n\n```
project-root/
├── data/
│   ├── raw/
│   │   └── tennis_atp/             # JeffSackmann repo
│   └── processed/
│       └── matches_combined.csv    # after concat & cleaning
├── notebooks/
│   └── 01-explore-data.ipynb       # EDA, feature ideas, Elo derivation
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── elo.py
│   ├── train.py
│   └── predict.py
├── models/
│   ├── gradient_boosting.pkl
│   └── scaler.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

1. Clone the [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) repo into `data/raw/tennis_atp/`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run EDA in `notebooks/01-explore-data.ipynb`.
4. Train a model:
   ```bash
   python src/train.py
   ```
5. Make predictions:
   ```bash
   python src/predict.py
   ```

## Features
- Data loading and cleaning
- Feature engineering (H2H, surface stats, recent form, Elo, rankings)
- Model training and calibration
- Prediction and probability output

## Requirements
See `requirements.txt` for dependencies. 