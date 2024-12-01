import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class FeaturesToDict(BaseEstimator, TransformerMixin):
    """
    Converts dataframe, or numpy array, into a dictionary oriented by records. This is a necessary pre-processing step
    for DictVectorizer().
    """
    def __int__(self):
        pass

    def fit(self, X, Y=None):
        return self

    def transform(self, X, Y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = X.to_dict(orient='records')
        return X


class TakeLog(BaseEstimator, TransformerMixin):
    """
    Based on the argument, takes the log of provided columns
    """
    def __init__(self, columns, take_log='yes'):
        self.take_log = take_log
        self.columns = columns

    def fit(self, X, Y=None):
        return self

    def transform(self, X, Y=None):
        if self.take_log == 'yes':
            for col in self.columns:
                X[col] = np.log(X[col])
                X[col] = X[col].replace([np.inf, -np.inf], 0)
                return X
        elif self.take_log == 'no':
            return X
        else:
            return X
