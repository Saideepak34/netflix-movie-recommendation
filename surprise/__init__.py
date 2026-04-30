import pandas as pd

class Reader:
    def __init__(self, rating_scale=(1, 5)):
        self.rating_scale = rating_scale

class Dataset:
    def __init__(self, df, reader):
        self.df = df.copy()
        self.reader = reader

    @classmethod
    def load_from_df(cls, df, reader):
        if df.shape[1] != 3:
            raise ValueError("Dataset must contain exactly three columns: user, item, rating.")
        return cls(df.reset_index(drop=True), reader)

class Prediction:
    def __init__(self, uid, iid, r_ui=None, est=0.0, details=None):
        self.uid = uid
        self.iid = iid
        self.r_ui = r_ui
        self.est = est
        self.details = details or {'was_impossible': False}

class SVD:
    def __init__(self, n_factors=20, n_epochs=5, lr_all=0.005, reg_all=0.02, random_state=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        self.global_mean = 0.0
        self.user_bias = {}
        self.item_bias = {}

    def fit(self, trainset):
        df = self._to_dataframe(trainset)
        self.global_mean = df['Rating'].mean()
        self.user_bias = (df.groupby('Cust_Id')['Rating'].mean() - self.global_mean).to_dict()
        self.item_bias = (df.groupby('Movie_Id')['Rating'].mean() - self.global_mean).to_dict()

    def test(self, testset):
        df = self._to_dataframe(testset)
        predictions = []
        for _, row in df.iterrows():
            uid = row['Cust_Id']
            iid = row['Movie_Id']
            actual = row['Rating']
            est = self._estimate(uid, iid)
            predictions.append((uid, iid, actual, est, {'was_impossible': False}))
        return predictions

    def predict(self, uid, iid, r_ui=None):
        est = self._estimate(uid, iid)
        return Prediction(uid, iid, r_ui=r_ui, est=est, details={'was_impossible': False})

    def _estimate(self, uid, iid):
        est = self.global_mean
        est += self.user_bias.get(uid, 0.0)
        est += self.item_bias.get(iid, 0.0)
        return est

    def _to_dataframe(self, dataset):
        if isinstance(dataset, Dataset):
            return dataset.df
        if isinstance(dataset, pd.DataFrame):
            return dataset
        raise ValueError('Unsupported dataset type for fit/test. Use Dataset or pandas.DataFrame.')
