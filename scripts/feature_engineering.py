"""Transformer fitur yang bergantung pada label.

Semua kelas di sini belajar dari label training. Agar tidak bocor:

- fit_transform(X, y) membuat fitur baris training secara out-of-fold, sehingga
  baris tidak pernah melihat labelnya sendiri atau label fold dalamnya.
- transform(X) membuat fitur baris baru memakai seluruh label training yang disimpan.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold


def _id_array(X):
    """Ubah DataFrame satu kolom berisi building_id menjadi array int64 datar."""
    return np.asarray(X).ravel().astype(np.int64)


def _inner_folds(y, n_splits, random_state):
    """Kembalikan daftar pasangan (indeks_train, indeks_validasi), stratified pada y."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(y)), y))


class NeighborFeatures(BaseEstimator, TransformerMixin):
    """Ringkas grade k bangunan berlabel dengan building_id terdekat.

    Id yang berurutan berada di ward yang sama, jadi id berfungsi sebagai lokasi.
    Outputnya: rata-rata grade, standar deviasi, dan proporsi tiap grade 1-5.
    """

    def __init__(self, k=5, n_splits=5, random_state=42):
        self.k = k
        self.n_splits = n_splits
        self.random_state = random_state

    def _lookup(self, query, src_ids, src_grades):
        """Hitung 7 fitur tetangga untuk id query dari sumber berlabel yang terurut."""
        k = self.k
        m = len(src_ids)
        pos = np.searchsorted(src_ids, query)
        cand = pos[:, None] + np.arange(-k, k)[None, :]
        valid = (cand >= 0) & (cand < m)
        cand = np.clip(cand, 0, m - 1)
        cand_ids = src_ids[cand]
        dist = np.abs(cand_ids - query[:, None]).astype(float)
        dist[~valid | (cand_ids == query[:, None])] = np.inf
        order = np.argsort(dist, axis=1, kind="stable")[:, :k]
        chosen = np.take_along_axis(cand, order, axis=1)
        ok = np.isfinite(np.take_along_axis(dist, order, axis=1))
        grades = src_grades[chosen]
        n_ok = ok.sum(axis=1)
        safe = np.maximum(n_ok, 1)
        mean = (grades * ok).sum(axis=1) / safe
        std = np.sqrt((((grades - mean[:, None]) ** 2) * ok).sum(axis=1) / safe)
        fractions = np.stack([((grades == g) & ok).sum(axis=1) / safe for g in range(1, 6)],
                             axis=1)
        mean[n_ok == 0] = np.nan
        std[n_ok == 0] = np.nan
        fractions[n_ok == 0] = 0.2
        return np.column_stack([mean, std, fractions])

    def fit(self, X, y):
        """Simpan building_id dan grade berlabel, terurut berdasarkan id."""
        ids = _id_array(X)
        order = np.argsort(ids)
        self.ids_ = ids[order]
        self.grades_ = (np.asarray(y) + 1)[order]
        return self

    def fit_transform(self, X, y):
        """Fit, lalu kembalikan fitur out-of-fold untuk baris training."""
        self.fit(X, y)
        ids = _id_array(X)
        grades = np.asarray(y) + 1
        out = np.zeros((len(ids), 7))
        for tr, va in _inner_folds(y, self.n_splits, self.random_state):
            order = np.argsort(ids[tr])
            out[va] = self._lookup(ids[va], ids[tr][order], grades[tr][order])
        return out

    def transform(self, X):
        """Kembalikan fitur baris baru memakai semua label yang disimpan."""
        return self._lookup(_id_array(X), self.ids_, self.grades_)

    def get_feature_names_out(self, input_features=None):
        """Kembalikan nama 7 kolom output."""
        return np.array(["nbr_mean_grade", "nbr_std_grade"]
                        + [f"nbr_frac_{g}" for g in range(1, 6)])


class WeakestLink(BaseEstimator, TransformerMixin):
    """Nilai kelemahan material terburuk sebuah bangunan.

    Tiap kategori material diberi peringkat menurut rata-rata damage grade-nya.
    Skor bangunan adalah peringkat tertinggi dari semua kolom material.
    """

    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    @staticmethod
    def _ranks(X, grade):
        """Kembalikan {kolom: kategori -> peringkat keparahan} dari baris yang diberikan."""
        return {c: grade.groupby(X[c]).mean().rank() for c in X.columns}

    @staticmethod
    def _score(X, ranks):
        """Kembalikan peringkat keparahan maksimum dari kolom material tiap baris."""
        scores = pd.DataFrame({c: X[c].map(ranks[c]) for c in X.columns})
        return scores.max(axis=1).to_numpy()

    def fit(self, X, y):
        """Pelajari peringkat keparahan dari semua baris berlabel."""
        grade = pd.Series(np.asarray(y) + 1, index=X.index)
        self.ranks_ = self._ranks(X, grade)
        return self

    def fit_transform(self, X, y):
        """Fit, lalu kembalikan skor out-of-fold untuk baris training."""
        self.fit(X, y)
        grade = pd.Series(np.asarray(y) + 1, index=X.index)
        out = np.full(len(X), np.nan)
        for tr, va in _inner_folds(y, self.n_splits, self.random_state):
            ranks = self._ranks(X.iloc[tr], grade.iloc[tr])
            out[va] = self._score(X.iloc[va], ranks)
        return out.reshape(-1, 1)

    def transform(self, X):
        """Kembalikan skor baris baru dengan peringkat dari semua baris berlabel."""
        return self._score(X, self.ranks_).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        """Kembalikan nama kolom output."""
        return np.array(["weakest_link"])


class LabelPropagation(BaseEstimator, TransformerMixin):
    """Sebarkan grade yang diketahui lewat graf bangunan yang terurut berdasarkan id.

    Tiap bangunan terhubung ke k bangunan sebelum dan sesudahnya dengan bobot
    1 / (jarak id + 1). Label dihaluskan selama n_iter putaran. Outputnya
    proporsi tiap grade 1-5.
    """

    def __init__(self, k=10, alpha=0.8, n_iter=15, n_splits=5, random_state=42):
        self.k = k
        self.alpha = alpha
        self.n_iter = n_iter
        self.n_splits = n_splits
        self.random_state = random_state

    def _graph(self, sorted_ids):
        """Bangun matriks bobot sparse ternormalisasi baris dari id yang terurut naik."""
        n = len(sorted_ids)
        rows, cols, vals = [], [], []
        for d in range(1, self.k + 1):
            i = np.arange(n - d, dtype=np.int32)
            w = 1.0 / (sorted_ids[i + d] - sorted_ids[i] + 1.0)
            rows += [i, i + d]
            cols += [i + d, i]
            vals += [w, w]
        graph = sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n))
        sums = np.asarray(graph.sum(axis=1)).ravel()
        sums[sums == 0] = 1
        return sparse.diags(1.0 / sums) @ graph

    def _spread(self, graph, seed):
        """Jalankan iterasi penghalusan dan kembalikan proporsi grade ternormalisasi."""
        state = seed.copy()
        for _ in range(self.n_iter):
            state = self.alpha * (graph @ state) + (1 - self.alpha) * seed
        return state / np.clip(state.sum(axis=1, keepdims=True), 1e-9, None)

    def fit(self, X, y):
        """Simpan building_id berlabel dan grade one-hot-nya, terurut berdasarkan id."""
        ids = _id_array(X)
        order = np.argsort(ids)
        self.ids_ = ids[order]
        self.onehot_ = np.eye(5)[np.asarray(y)[order]]
        return self

    def fit_transform(self, X, y):
        """Fit, lalu kembalikan fitur propagasi out-of-fold untuk baris training."""
        self.fit(X, y)
        ids = _id_array(X)
        y = np.asarray(y)
        order = np.argsort(ids)
        position = np.empty(len(ids), dtype=np.int64)
        position[order] = np.arange(len(ids))
        graph = self._graph(ids[order])
        onehot = np.eye(5)[y]
        out = np.zeros((len(ids), 5))
        for tr, va in _inner_folds(y, self.n_splits, self.random_state):
            seed = np.zeros((len(ids), 5))
            seed[position[tr]] = onehot[tr]
            out[va] = self._spread(graph, seed)[position[va]]
        return out

    def transform(self, X):
        """Kembalikan fitur baris baru dari graf bangunan tersimpan dan baru."""
        new_ids = _id_array(X)
        n_fit = len(self.ids_)
        all_ids = np.concatenate([self.ids_, new_ids])
        order = np.argsort(all_ids)
        seed = np.zeros((len(all_ids), 5))
        seed[:n_fit] = self.onehot_
        shares = self._spread(self._graph(all_ids[order]), seed[order])
        restored = np.empty_like(shares)
        restored[order] = shares
        return restored[n_fit:]

    def get_feature_names_out(self, input_features=None):
        """Kembalikan nama 5 kolom output."""
        return np.array([f"prop_g{g}" for g in range(1, 6)])


class WeakestLinkInteraction(BaseEstimator, TransformerMixin):
    """Tambah kolom weakest_link dikali nbr_mean_grade.

    Rata-rata tetangga mewakili intensitas guncangan lokal dan weakest_link mewakili
    kerentanan struktur, sehingga hasil kalinya menggambarkan dampak pada bangunan rapuh.
    """

    def fit(self, X, y=None):
        """Catat kolom input; tidak ada yang dipelajari."""
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        """Kembalikan salinan X dengan kolom interaksi tambahan."""
        out = X.copy()
        out["weakest_link_x_nbr"] = out["weakest_link"] * out["nbr_mean_grade"]
        return out

    def get_feature_names_out(self, input_features=None):
        """Kembalikan nama input ditambah nama kolom interaksi."""
        return np.append(np.asarray(input_features, dtype=object), "weakest_link_x_nbr")