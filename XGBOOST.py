import xgboost as xgb
from sklearn.model_selection import train_test_split

def xgboost(df):
    X = df.drop(columns = {'sales','test','id'})
    y = df['sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    hyper_params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'reg:regression', 'metric': ['l1', 'l2'],
                    'learning_rate': 0.1,
                    'feature_fraction': 0.9, 'bagging_fraction': 0.7, 'bagging_freq': 10, 'verbose': 0, "max_depth": 50,
                    "num_leaves": 128, "max_bin": 512}

    # Initialize the XGBoost Regressor
    xgb_regressor = xgb.XGBRegressor()

    # Train the model
    xgb_regressor.fit(X_train, y_train)

    # Predict on the test set
    y_pred = xgb_regressor.predict(X_test)

    train_score = xgb_regressor.score(X_train, y_train)
    test_score = xgb_regressor.score(X_test, y_test)

    print("Train Score:", train_score)
    print("Test Score:", test_score)

    return xgb_regressor

