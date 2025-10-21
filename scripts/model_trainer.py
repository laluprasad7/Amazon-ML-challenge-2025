import numpy as np
import xgboost as xgb

def smape(y_true, y_pred):
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def train_model(train_feats, train_prices, val_feats, val_prices):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist'
    }

    dtrain = xgb.DMatrix(train_feats, label=train_prices)
    dval = xgb.DMatrix(val_feats, label=val_prices)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    val_preds = model.predict(dval)
    print(f'Validation SMAPE: {smape(val_prices, val_preds):.2f}%')
    return model