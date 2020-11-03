from sklearn.ensemble import RandomForestRegressor


def rf_predict(x_train, y_train, x_test):
    return model_fit(x_train, y_train).predict(x_test)


def predict(model, x_test):
    return model.predict(x_test)


def model_fit(x_train, y_train):
    rf_reg = RandomForestRegressor(n_estimators=10)
    rf_reg.fit(x_train, y_train)
    return rf_reg
