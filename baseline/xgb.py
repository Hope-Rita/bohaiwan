import xgboost as xgb


def xgb_predict(x_train, y_train, x_test):
    return model_fit(x_train, y_train).predict(x_test)


def predict(model, x_test):
    return model.predict(x_test)


def model_fit(x_train, y_train):
    model = xgb.XGBRegressor(learning_rate=0.1,
                             max_depth=15,
                             gamma=1e-3,
                             n_estimators=20,
                             objective='reg:squarederror'
                             )
    model.fit(x_train, y_train)
    return model
