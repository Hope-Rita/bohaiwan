from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def lr_pca_predict(x_train, y_train, x_test):
    from utils.config import get_config
    estimator = PCA(n_components=get_config('../section_predict/config.json',
                                            'model-parameters',
                                            'lr',
                                            'pca-components'))

    x_train_pca = estimator.fit_transform(x_train)
    x_test_pca = estimator.transform(x_test)
    return lr_predict(x_train_pca, y_train, x_test_pca)


def lr_predict(x_train, y_train, x_test):
    return model_fit(x_train, y_train).predict(x_test)


def predict(model, x_test):
    return model.predict(x_test)


def model_fit(x_train, y_train):
    linear_reg = LinearRegression()
    linear_reg.fit(x_train, y_train)
    return linear_reg
