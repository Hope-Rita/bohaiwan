from sklearn.linear_model import LinearRegression


def lr_predict(x_train, y_train, x_test):
    return model_fit(x_train, y_train).predict(x_test)


def predict(model, x_test):
    return model.predict(x_test)


def model_fit(x_train, y_train):
    linear_reg = LinearRegression()
    linear_reg.fit(x_train, y_train)
    return linear_reg
