from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_squared_log_error


# Assuming you have your features stored in X and target variable in y
def randomforest(X_train, X_test, y_train, y_test):

    # Initialize the Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=21, random_state=42)

    # Train the model
    rf_regressor.fit(X_train, y_train)

    train_score = rf_regressor.score(X_train, y_train)
    test_score = rf_regressor.score(X_test, y_test)
    print("Train Score:", train_score)
    print("Test Score:", test_score)

    # Predict on the test set
    y_test_pred = rf_regressor.predict(X_test)

    def rmsle(y_true, y_pred):
        return np.sqrt(mean_squared_log_error(y_true, y_pred))

    error = rmsle(y_test, y_test_pred)
    rounded_error = round(error, 4)
    print(f"RMSLE: {rounded_error}")
    return rf_regressor