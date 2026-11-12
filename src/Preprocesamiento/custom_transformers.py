from sklearn.base import BaseEstimator, TransformerMixin
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, fold=1.5):
        self.fold = fold
        self.cuantiles = {}
        self.columnas = None

    def get_feature_names_out(self, input_features=None):
        return self.columnas

    def fit(self, X, y=None):
        self.columnas = X.columns
        for n in self.columnas:
            Q3 = X[n].quantile(0.75)
            Q1 = X[n].quantile(0.25)
            self.cuantiles[n] = (Q1, Q3)
        return self


    def transform(self, X):
        fold = self.fold
        for n in self.columnas:
            Q1, Q3 = self.cuantiles[n]
            IQR = Q3 - Q1
            outliers = X[(X[n] < - fold * IQR + Q1 ) | (X[n] >  fold * IQR + Q3 )].index
            X.loc[outliers,n] = None
        return X
