import pandas as pd

from surprise import Dataset


def train_test_split(data, test_size=0.2, random_state=42):
    if isinstance(data, Dataset):
        df = data.df
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError('train_test_split expects a surprise.Dataset or pandas.DataFrame.')

    test = df.sample(frac=test_size, random_state=random_state)
    train = df.drop(test.index).reset_index(drop=True)
    test = test.reset_index(drop=True)
    return Dataset(train, data.reader if isinstance(data, Dataset) else None), Dataset(test, data.reader if isinstance(data, Dataset) else None)
