from sklearn import linear_model
import numpy as np
import stock_data as sd

def build_model(X,y):
	linear_mod = linear_model.LinearRegression() #define linear regression model
	X = np.reshape(X,(X.shape[0], 1))
	y = np.reshape(y,(y.shape[0], 1))
	linear_mod.fit(X,y)
	
	return linear_mod
	
def predict_prices(model, x, label_range):
    """
    Predict the label for given test sets
    :param model: a linear regression model
    :param x: testing features
    :param label_range: normalised range of label data
    :return: predicted labels for given features
    """
    x = np.reshape(x, (x.shape[0], 1))
    predicted_price = model.predict(x)
    predictions_rescaled, re_range = sd.scale_range(predicted_price, input_range=[-1.0, 1.0], target_range=label_range)

    return predictions_rescaled.flatten()