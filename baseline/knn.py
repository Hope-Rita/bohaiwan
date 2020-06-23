from sklearn.neighbors import KNeighborsRegressor


def rf_predict(x_train, y_train, x_test):
    return model_fit(x_train, y_train).predict(x_test)


def predict(model, x_test):
    return model.predict(x_test)


def model_fit(x_train, y_train):
    knn_reg = KNeighborsRegressor()
    knn_reg.fit(x_train, y_train)
    return knn_reg
