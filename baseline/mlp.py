from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from utils.config import get_config


hidden_size = tuple(get_config('../union_predict/section_config.json', 'model-parameters', 'mlp', 'hidden-size'))


def mlp_predict(x_train, y_train, x_test):
    return model_fit(x_train, y_train).predict(x_test)


def predict(model, x_test):
    return model.predict(x_test)


def model_fit(x_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=hidden_size, activation='identity')
    model.fit(x_train, y_train)
    return model
