"""Definisi kolom dan preprocessing untuk pipeline damage grade."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "submissions"
FIGURE_DIR = ROOT / "reports" / "figures"

ID_COL = "building_id"
TARGET_COL = "damage_grade"
NUM_COLS = ["count_floors_pre_eq", "age_building", "plinth_area_sq_ft", "height_ft_pre_eq"]
CAT_COLS = ["land_surface_condition", "foundation_type", "roof_type", "ground_floor_type",
            "other_floor_type", "position", "plan_configuration", "superstructure"]
LOC_COLS = ["district_id", "vdcmun_id", "ward_id"]
MAT_COLS = ["foundation_type", "roof_type", "ground_floor_type", "other_floor_type",
            "superstructure"]


def parse_target(df):
    """Ubah kolom damage_grade menjadi kelas integer 0-4 (menerima 1-5 atau "Grade 3")."""
    return df[TARGET_COL].astype(str).str[-1].astype(int).to_numpy() - 1


class BasicFeatures(BaseEstimator, TransformerMixin):
    """Batasi usia bangunan, lalu tambah age_unknown, height_per_floor, dan area_per_floor.

    Tidak memakai label, jadi tidak ada kebocoran.
    """

    def __init__(self, age_cap=200):
        self.age_cap = age_cap

    def fit(self, X, y=None):
        """Tidak ada yang dipelajari; mengembalikan self."""
        return self

    def transform(self, X):
        """Kembalikan kolom numerik beserta tiga kolom hasil rekayasa."""
        out = pd.DataFrame(index=X.index)
        age = X["age_building"].clip(upper=self.age_cap)
        out["count_floors_pre_eq"] = X["count_floors_pre_eq"]
        out["age_building"] = age
        out["plinth_area_sq_ft"] = X["plinth_area_sq_ft"]
        out["height_ft_pre_eq"] = X["height_ft_pre_eq"]
        out["age_unknown"] = (age >= self.age_cap).astype(int)
        out["height_per_floor"] = X["height_ft_pre_eq"] / X["count_floors_pre_eq"]
        out["area_per_floor"] = X["plinth_area_sq_ft"] / X["count_floors_pre_eq"]
        return out

    def get_feature_names_out(self, input_features=None):
        """Kembalikan nama kolom output."""
        return np.array(["count_floors_pre_eq", "age_building", "plinth_area_sq_ft",
                         "height_ft_pre_eq", "age_unknown", "height_per_floor",
                         "area_per_floor"])


class CategoryCaster(BaseEstimator, TransformerMixin):
    """Ubah kolom ke dtype category agar LightGBM memperlakukannya sebagai nominal.

    Nilai yang tidak ada saat fit menjadi NaN.
    """

    def fit(self, X, y=None):
        """Pelajari daftar kategori terurut dari setiap kolom."""
        self.columns_ = list(X.columns)
        self.categories_ = {c: pd.Categorical(X[c].astype(str)).categories for c in self.columns_}
        return self

    def transform(self, X):
        """Kembalikan kolom sebagai kategorikal sesuai kategori hasil fit."""
        out = pd.DataFrame(index=X.index)
        for c in self.columns_:
            out[c] = pd.Categorical(X[c].astype(str), categories=self.categories_[c])
        return out

    def get_feature_names_out(self, input_features=None):
        """Kembalikan nama kolom apa adanya."""
        return np.array(self.columns_)


def build_preprocessor(neighbor, weakest, propagation):
    """Susun ColumnTransformer yang membuat seluruh fitur model.

    Args:
        neighbor: transformer fitur tetangga berdasarkan building_id.
        weakest: transformer fitur weakest-link dari kolom material.
        propagation: transformer fitur label propagation dari building_id.

    Returns:
        ColumnTransformer dengan output pandas tanpa prefiks nama fitur.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("basic", BasicFeatures(), NUM_COLS),
            ("cats", CategoryCaster(), CAT_COLS + LOC_COLS),
            ("neighbors", neighbor, [ID_COL]),
            ("weakest", weakest, MAT_COLS),
            ("propagation", propagation, [ID_COL]),
        ],
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor