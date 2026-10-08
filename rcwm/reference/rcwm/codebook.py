"""Deterministic train-only k-means diagnostic, not a learned VQ implementation."""
from __future__ import annotations
import numpy as np


class TrainOnlyKMeans:
    def __init__(self, clusters: int, seed: int = 0, iterations: int = 30):
        if clusters < 1 or iterations < 1:
            raise ValueError("positive clusters and iterations required")
        self.clusters, self.seed, self.iterations = clusters, seed, iterations
        self.centers = None

    def fit(self, x, *, split: str):
        if split != "train":
            raise ValueError("codebook fitting requires TRAIN; no dev/test fitting")
        x = np.asarray(x, float)
        if x.ndim != 2 or len(x) < self.clusters or not np.isfinite(x).all():
            raise ValueError("finite [samples,features] with samples >= clusters required")
        rng = np.random.default_rng(self.seed)
        centers = x[rng.choice(len(x), self.clusters, replace=False)].copy()
        for _ in range(self.iterations):
            labels = ((x[:,None,:]-centers[None,:,:])**2).sum(-1).argmin(1)
            new = np.array([x[labels==k].mean(0) if (labels==k).any() else centers[k]
                            for k in range(self.clusters)])
            if np.allclose(new, centers, rtol=0, atol=1e-10):
                centers = new
                break
            centers = new
        self.centers = centers
        return self

    def predict(self, x):
        if self.centers is None:
            raise RuntimeError("fit on TRAIN first")
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != self.centers.shape[1] or not np.isfinite(x).all():
            raise ValueError("finite matching features required")
        return ((x[:,None,:]-self.centers[None,:,:])**2).sum(-1).argmin(1)
