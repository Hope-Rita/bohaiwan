import xgboost as xgb


def xgb_predict(x_train, y_train, x_test):
    return model_fit(x_train, y_train).predict(x_test)


def predict(model, x_test):
    return model.predict(x_test)


def model_fit(x_train, y_train):
    model = xgb.XGBRegressor(learning_rate=0.3,
                             max_depth=2,
                             gamma=0,
                             n_estimators=30,
                             objective='reg:squarederror'
                             )
    model.fit(x_train, y_train)
    return model
