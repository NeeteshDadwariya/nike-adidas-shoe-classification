import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

def sq_err(prediction, actual):
    squared_err = np.power(prediction - actual, 2)
    return 0.5 * np.sum(squared_err)


def get_metrics(pred, true):
    cm = multilabel_confusion_matrix(true, pred)
    return (cm)