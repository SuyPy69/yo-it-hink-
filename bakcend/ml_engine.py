import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os

LOG_FILE = 'ml_training_log.csv'
COLUMNS = ['traffic', 'rain', 'pop', 'dist', 'is_neg', 'deficit']


def recalibrate():
    # Ensure file has headers
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=COLUMNS).to_csv(LOG_FILE, index=False)

    # New Data Point (5 features + 1 target)
    new_data = {
        'traffic': [round(np.random.uniform(0.5, 4.0), 2)],
        'rain': [np.random.randint(0, 100)],
        'pop': [np.random.choice([5000, 15000, 25000])],
        'dist': [round(np.random.uniform(1, 30), 2)],
        'is_neg': [np.random.choice([0, 1])],
        'deficit': [np.random.randint(5, 30)]
    }

    pd.DataFrame(new_data).to_csv(LOG_FILE, mode='a', header=False, index=False)
    df = pd.read_csv(LOG_FILE)

    if len(df) >= 5:
        # X MUST HAVE 5 COLUMNS
        X = df[['traffic', 'rain', 'pop', 'dist', 'is_neg']]
        y = df['deficit']
        model = GradientBoostingRegressor(n_estimators=100)
        model.fit(X, y)
        joblib.dump(model, 'blood_predictor.pkl')
        print(f">> SUCCESS: Trained 5-parameter model. Dataset: {len(df)}")
    else:
        print(f">> STATUS: Need {5 - len(df)} more runs.")


if __name__ == "__main__":
    recalibrate()