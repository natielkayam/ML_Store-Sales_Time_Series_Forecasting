import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# Fit the KNN model
def activeKNN(df):

    X = df.drop(columns = {'sales','test','id'})
    y = df['sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Build and fit KNN model
    k_values = range(1, 21)
    mse_values = []  # List to store mean squared errors for different K values

    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        mse = np.mean((knn.predict(X_test) - y_test) ** 2)
        mse_values.append(mse)

    # Find the optimal value of K
    optimal_k = np.argmin(mse_values) + 1

    # Build the final model with the optimal K
    knn = KNeighborsRegressor(n_neighbors=optimal_k)
    knn.fit(X_train, y_train)

    # Evaluate the model
    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)
    print(f"Train score: {train_score:.3f}, Test score: {test_score:.3f}")
    y_pred = knn.predict(X_test)
    # rmse = root_mean_squared_error(y_test, y_pred)
    # print(rmse)
    return knn
