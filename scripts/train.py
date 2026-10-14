"""Bangun, evaluasi, tuning, dan simpan pipeline damage grade.

Jalankan dari root repositori dengan: python -m scripts.train
"""

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from scripts.feature_engineering import (LabelPropagation, NeighborFeatures,
                                         WeakestLink, WeakestLinkInteraction)
from scripts.preprocess import (DATA_DIR, FIGURE_DIR, MODEL_DIR, TARGET_COL,
                                build_preprocessor, parse_target)

warnings.filterwarnings("ignore", category=Warning, module="lightgbm")

DEFAULT_MODEL_PARAMS = dict(
    objective="multiclass", class_weight="balanced", learning_rate=0.08, n_estimators=230,
    num_leaves=127, min_child_samples=40, colsample_bytree=0.8, subsample=0.8,
    subsample_freq=1, reg_lambda=5.0, cat_smooth=20, cat_l2=10, n_jobs=-1,
    verbosity=-1, random_state=42,
)

SEARCH_SPACE = {
    "model__num_leaves": [63, 127, 255],
    "model__min_child_samples": [20, 40, 80],
    "model__n_estimators": [150, 200, 250, 300],
    "model__learning_rate": [0.05, 0.08, 0.1],
}


def load_data(data_dir=DATA_DIR):
    """Baca file kompetisi.

    Returns:
        X_train (tanpa target), y (kelas 0-4), X_test, dan sample submission.
    """
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_sub = pd.read_csv(data_dir / "sample_submission.csv")
    y = parse_target(train)
    return train.drop(columns=[TARGET_COL]), y, test, sample_sub


def build_pipeline(memory=None, **model_params):
    """Buat pipeline lengkap: ColumnTransformer fitur, interaksi, dan LightGBM.

    Args:
        memory: folder cache opsional untuk langkah fitur yang sudah di-fit.
        **model_params: parameter LightGBM yang menimpa DEFAULT_MODEL_PARAMS.

    Returns:
        Pipeline scikit-learn yang belum di-fit.
    """
    params = {**DEFAULT_MODEL_PARAMS, **model_params}
    features = build_preprocessor(NeighborFeatures(), WeakestLink(), LabelPropagation())
    return Pipeline(
        steps=[
            ("features", features),
            ("interaction", WeakestLinkInteraction()),
            ("model", LGBMClassifier(**params)),
        ],
        memory=memory,
    )


def cross_validated_probs(pipeline, X, y, n_splits=5, random_state=42):
    """Kembalikan probabilitas kelas out-of-fold untuk setiap baris training.

    Seluruh pipeline di-fit ulang di tiap fold, sehingga label fold validasi tidak terpakai.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")


def tune_offsets(probs, y_true, n_iter=300):
    """Cari offset log-probabilitas per kelas yang memaksimalkan macro F1.

    Initial simplex eksplisit diperlukan: simplex bawaan Nelder-Mead terlalu kecil
    untuk mengubah keputusan argmax, sehingga pencarian tidak pernah membaik.

    Args:
        probs: array (n, 5) probabilitas prediksi.
        y_true: kelas sebenarnya 0-4.
        n_iter: iterasi Nelder-Mead maksimum.

    Returns:
        Array 5 offset yang ditambahkan ke log-probabilitas sebelum argmax.
    """
    def neg_f1(offsets):
        pred = (np.log(probs + 1e-9) + offsets).argmax(axis=1)
        return -f1_score(y_true, pred, average="macro")

    x0 = np.zeros(5)
    simplex = np.vstack([x0] + [x0 + 0.5 * np.eye(5)[i] for i in range(5)])
    result = minimize(neg_f1, x0=x0, method="Nelder-Mead",
                      options={"maxiter": n_iter, "xatol": 1e-3, "fatol": 1e-6,
                               "initial_simplex": simplex, "adaptive": True})
    return result.x


def apply_offsets(probs, offsets):
    """Kembalikan kelas prediksi 0-4 setelah log-probabilitas digeser oleh offset."""
    return (np.log(probs + 1e-9) + offsets).argmax(axis=1)


def tune_hyperparameters(X, y, n_iter=8, n_splits=3, random_state=42):
    """Cari parameter LightGBM secara acak melalui seluruh pipeline.

    Langkah fitur di-cache, sehingga di-fit sekali per fold, bukan per kandidat.

    Returns:
        Objek RandomizedSearchCV yang sudah di-fit.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        build_pipeline(memory=str(MODEL_DIR / "cache")), SEARCH_SPACE, n_iter=n_iter,
        scoring="f1_macro", cv=cv, refit=False, random_state=random_state, verbose=2)
    return search.fit(X, y)


def save_figures(pipeline, y, probs, offsets, figure_dir=FIGURE_DIR):
    """Simpan plot feature importance dan confusion matrix sebagai PNG."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    names = pipeline[:-1].get_feature_names_out()
    gain = pipeline.named_steps["model"].booster_.feature_importance("gain")
    top = pd.Series(gain, index=names).sort_values().tail(15)
    top.plot(kind="barh", figsize=(7, 5), title="Feature importance (gain)")
    plt.tight_layout()
    plt.savefig(figure_dir / "feature_importance.png", dpi=150)
    plt.close()
    pred = apply_offsets(probs, offsets)
    ConfusionMatrixDisplay.from_predictions(y + 1, pred + 1, normalize="true",
                                            values_format=".2f", cmap="Blues")
    plt.title("Out-of-fold confusion matrix (row-normalised)")
    plt.tight_layout()
    plt.savefig(figure_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def main(run_search=False, n_search=8):
    """Evaluasi dengan cross-validation, fit pada semua data, dan simpan artefak."""
    X, y, _, _ = load_data()
    model_params = {}
    if run_search:
        search = tune_hyperparameters(X, y, n_iter=n_search)
        model_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        print("best params:", model_params, "| cv macro F1:", round(search.best_score_, 4))
    pipeline = build_pipeline(**model_params)
    probs = cross_validated_probs(pipeline, X, y)
    print("OOF macro F1 (argmax):", round(f1_score(y, probs.argmax(axis=1), average="macro"), 4))
    offsets = tune_offsets(probs, y)
    print("OOF macro F1 (tuned):", round(f1_score(y, apply_offsets(probs, offsets), average="macro"), 4))
    pipeline.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "pipeline.pkl", compress=3)
    np.save(MODEL_DIR / "offsets.npy", offsets)
    np.save(MODEL_DIR / "oof_probs.npy", probs)
    save_figures(pipeline, y, probs, offsets)


if __name__ == "__main__":
    main()