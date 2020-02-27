from sklearn.svm import LinearSVR


def svr_predict(x_train, y_train, x_test):
    return model_fit(x_train, y_train).predict(x_test)


def predict(model, x_test):
    return model.predict(x_test)


def model_fit(x_train, y_train):
    svm_reg = LinearSVR()
    svm_reg.fit(x_train, y_train)
    return svm_reg
