import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeRegressor

def dt(df):
    X = df.drop(columns = {'sales','test','id'})
    y = df['sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_Regressor = DecisionTreeRegressor()
    best_Regressor.fit(X_train, y_train)

    # Calculate train and test scores
    train_score = best_Regressor.score(X_train, y_train)
    test_score = best_Regressor.score(X_test, y_test)
    print("Train Score:", train_score)
    print("Test Score:", test_score)

    y_test_pred = best_Regressor.predict(X_test)

    def rmsle(y_true, y_pred):
        return np.sqrt(mean_squared_log_error(y_true, y_pred))

    error = rmsle(y_test, y_test_pred)
    rounded_error = round(error, 4)
    print(f"RMSLE: {rounded_error}")
    return best_Regressor
