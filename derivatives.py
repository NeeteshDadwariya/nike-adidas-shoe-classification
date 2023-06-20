import numpy as np

def get_der_err_wrt_b2(prediction, actual, y_h, f_dash):
    return (prediction - actual) * f_dash(y_h)

def get_der_err_wrt_w2(h, der_err_wrt_b2):
    return np.dot(h.T, der_err_wrt_b2)

def get_der_err_wrt_b1(h_h, der_err_wrt_b2, weights2, f_dash):
    return np.dot(der_err_wrt_b2, weights2.T) * f_dash(h_h)

def get_der_err_wrt_w1(x, der_err_wrt_b1):
    return np.dot(x.T, der_err_wrt_b1)