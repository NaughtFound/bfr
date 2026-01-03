from itertools import combinations
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils import BFRData, DataLoader


class BFR:
    def __init__(
        self,
        k: int,
        batch_size: int,
        ds_thresh: float,
        cs_rs_thresh: int,
        cs_thresh: float,
    ):
        self.k = k
        self.ds_thresh = ds_thresh
        self.cs_rs_thresh = cs_rs_thresh
        self.cs_thresh = cs_thresh
        self.batch_size = batch_size

        self.DS: list[BFRData] = []
        self.CS: list[BFRData] = []
        self.RS: list[np.ndarray] = []

    def calc_metrics(self, points: np.ndarray):
        N = len(points)
        SUM = np.sum(points, axis=0)
        SUMSQ = np.sum(points**2, axis=0)

        return BFRData(N, SUM, SUMSQ)

    def calc_md(self, points: np.ndarray, bfrData: BFRData, eps: float = 1e-6):
        centroid = bfrData.mu()
        variance = bfrData.var()

        variance[variance == 0] = eps

        return np.sqrt(np.sum((points - centroid) ** 2 / variance, axis=1))

    def init_clusters(self, points: np.ndarray):
        cls = AgglomerativeClustering(n_clusters=self.k)
        cls.fit(points)
        labels = cls.labels_

        self.DS.clear()

        for cls_idx in range(self.k):
            cls_points = points[labels == cls_idx]
            cls_metrics = self.calc_metrics(cls_points)
            if len(cls_points) > 0:
                self.DS.append(cls_metrics)

    def merge_cs(self, thresh: float):
        merged = set()

        added_cs = []

        pairs = combinations(self.CS, 2)
        len_cs = len(self.CS)

        for cs1, cs2 in tqdm(
            pairs,
            total=len_cs * (len_cs - 1) // 2,
            desc="Merging CS",
        ):
            merged_cs = BFRData.merge(cs1, cs2)
            merged_var = np.mean(merged_cs.var())

            if merged_var <= thresh:
                if cs1.id() in added_cs or cs2.id() in added_cs:
                    continue

                added_cs.append(cs1.id())
                added_cs.append(cs2.id())

                merged.add(merged_cs)

        for cs in self.CS:
            if cs.id() not in added_cs:
                merged.add(cs)

        self.CS = list(merged)

    def update_ds(self, points: np.ndarray, thresh: float):
        for ds in self.DS:
            dist = self.calc_md(points, ds)
            dist_idx = np.where(dist <= thresh)[0]

            close_points = points[dist_idx]
            ds.update(close_points)

            points = points[~np.isin(np.arange(len(points)), dist_idx)]
            if len(points) == 0:
                break
        return points

    def update_cs_rs(self, points: np.ndarray, thresh: int):
        k = min(len(points) // 2, self.k // 4)

        if k == 0:
            self.RS = list(points)
            return

        cls = AgglomerativeClustering(n_clusters=k)
        cls.fit(points)

        labels = cls.labels_

        self.RS.clear()

        for cls_idx in range(k):
            cls_points = points[labels == cls_idx]
            cls_metrics = self.calc_metrics(cls_points)

            if len(cls_points) == 0:
                continue

            if len(cls_points) > thresh:
                self.CS.append(cls_metrics)
            else:
                self.RS.append(cls_points)

        if len(self.RS) > 0:
            self.RS = list(np.concatenate(self.RS, axis=0))

    def create_remaining_points(self, points: np.ndarray):
        if len(self.RS) == 0 and len(points) == 0:
            return np.array([])
        elif len(self.RS) == 0:
            remaining_points = points
        elif len(points) == 0:
            remaining_points = np.array(self.RS)
        else:
            remaining_points = np.vstack((self.RS, points))

        return remaining_points

    def find_min_md(self, points: np.ndarray):
        min_md = self.calc_md(points, self.DS[0])
        min_md_idx = np.zeros(len(points))

        for i, ds in enumerate(self.DS[1:]):
            md = self.calc_md(points, ds)

            md_matrix = np.array([min_md_idx, (md < min_md) * (i + 1)])
            min_md_idx = np.max(md_matrix, axis=0)

            md_matrix = np.array([min_md, md])
            min_md = np.min(md_matrix, axis=0)

        return min_md_idx, min_md

    def finalize_clusters(self):
        if len(self.CS) > 0:
            arr_mu_cs = []
            arr_mu_ds = []

            for cs in self.CS:
                arr_mu_cs.append(cs.mu())

            for ds in self.DS:
                arr_mu_ds.append(ds.mu())

            cs_matrix = cosine_similarity(arr_mu_cs, arr_mu_ds)

            cs_max_idx = np.argmax(cs_matrix, axis=1)

            for i, ds_i in enumerate(cs_max_idx):
                self.DS[ds_i] = BFRData.merge(self.DS[ds_i], self.CS[i])

        if len(self.RS) > 0:
            rs = np.array(self.RS)

            min_md_idx, _ = self.find_min_md(rs)

            for i in range(len(self.DS)):
                points = rs[min_md_idx == i]

                if len(points) > 0:
                    self.DS[i].update(points)

        self.RS.clear()
        self.CS.clear()

    def fit(self, data: np.ndarray, num_samples: int):
        L = len(data)
        samples_idx = np.random.choice(L, min(num_samples, L), replace=False)
        self.init_clusters(data[samples_idx])

        remaining_data = data[~np.isin(np.arange(L), samples_idx)]

        dataloader = DataLoader(remaining_data, self.batch_size)

        for batch in tqdm(
            dataloader,
            desc="Reading data in Batches",
            total=len(dataloader),
        ):
            points = self.update_ds(batch, self.ds_thresh)
            remaining_points = self.create_remaining_points(points)
            if len(remaining_points) > 0:
                self.update_cs_rs(remaining_points, self.cs_rs_thresh)
        self.merge_cs(self.cs_thresh)
        self.finalize_clusters()

    @property
    def labels_(self):
        labels_ = []

        for i, ds in enumerate(self.DS):
            labels_.append([i] * ds.N)

        labels_ = np.concatenate(labels_, axis=0)
        return labels_

    def predict(self, data: np.ndarray):
        dataloader = DataLoader(data, self.batch_size, shuffle=False)

        labels = []

        for batch in tqdm(
            dataloader,
            desc="Reading data in Batches",
            total=len(dataloader),
        ):
            min_md_idx, _ = self.find_min_md(batch)

            labels.append(min_md_idx)

        return np.concatenate(labels, axis=0)
