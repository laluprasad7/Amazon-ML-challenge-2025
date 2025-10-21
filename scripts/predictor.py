import pandas as pd
import numpy as np
import xgboost as xgb
from config import output_csv

def predict_and_save(model, test_feats, test_df):
    dtest = xgb.DMatrix(test_feats)
    test_preds = np.expm1(model.predict(dtest))
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': test_preds
    })
    output_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")