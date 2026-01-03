import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer


class BFRData(object):
    def __init__(self, N: int, SUM: np.ndarray, SUMSQ: np.ndarray):
        self.N = N
        self.SUM = SUM
        self.SUMSQ = SUMSQ

    def id(self):
        return self.__repr__()

    def update(self, points: np.ndarray):
        self.N += len(points)
        self.SUM += np.sum(points, axis=0)
        self.SUMSQ += np.sum(points, axis=0) ** 2

    def mu(self):
        return self.SUM / self.N

    def var(self):
        mu = self.mu()
        return self.SUMSQ / self.N - mu**2

    @staticmethod
    def merge(first: "BFRData", second: "BFRData"):
        return BFRData(
            first.N + second.N,
            first.SUM + second.SUM,
            first.SUMSQ + second.SUMSQ,
        )


class Dataset:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    @staticmethod
    def split_data(
        x: np.ndarray,
        y: np.ndarray,
        test_ratio=0.3,
        use_y_for_stratify=True,
    ):
        if use_y_for_stratify:
            y_stratify = y

        else:
            estimator = KBinsDiscretizer(
                n_bins=10, encode="ordinal", strategy="quantile"
            )
            y_stratify = estimator.fit_transform(y.reshape(-1, 1)).flatten()

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_ratio,
            stratify=y_stratify,
        )

        d_train = Dataset(x_train, y_train)
        d_test = Dataset(x_test, y_test)

        return {
            "d_train": d_train,
            "d_test": d_test,
        }


class DataLoader:
    def __init__(self, data: np.ndarray, batch_size: int, shuffle: bool = True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = data.shape[0]
        self.current_index = 0
        self.indices = np.arange(self.num_samples)

        if self.shuffle:
            self.shuffle_indices()

    def __len__(self):
        return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration

        batch_indices = self.indices[
            self.current_index : self.current_index + self.batch_size
        ]
        batch = self.data[batch_indices]
        self.current_index += self.batch_size
        return batch

    def reset(self):
        self.current_index = 0
        if self.shuffle:
            self.shuffle_indices()

    def shuffle_indices(self):
        np.random.shuffle(self.indices)
