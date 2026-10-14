"""Buat submission Kaggle dari pipeline yang tersimpan.

Jalankan dari root repositori dengan: python -m scripts.predict
"""

import joblib
import numpy as np
import pandas as pd

from scripts.preprocess import DATA_DIR, ID_COL, MODEL_DIR, SUBMISSION_DIR
from scripts.train import apply_offsets


def load_artifacts(model_dir=MODEL_DIR):
    """Muat pipeline yang sudah di-fit dan offset kelas dari disk."""
    return joblib.load(model_dir / "pipeline.pkl"), np.load(model_dir / "offsets.npy")


def predict_grades(pipeline, offsets, test):
    """Prediksi damage grade 1-5 untuk baris test, sesuai urutan file test."""
    return apply_offsets(pipeline.predict_proba(test), offsets) + 1


def make_submission(test, grades, sample_sub):
    """Cocokkan prediksi dengan sample submission lewat building_id, lalu validasi.

    Returns:
        DataFrame berisi building_id dan damage_grade.
    """
    labels = pd.Series(grades, index=test[ID_COL])
    submission = pd.DataFrame({ID_COL: sample_sub[ID_COL],
                               "damage_grade": sample_sub[ID_COL].map(labels)})
    assert submission["damage_grade"].notna().all()
    assert submission.shape == sample_sub.shape
    return submission


def main():
    """Tulis submissions/submission.csv."""
    pipeline, offsets = load_artifacts()
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    submission = make_submission(test, predict_grades(pipeline, offsets, test), sample_sub)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_DIR / "submission.csv", index=False)
    print(submission["damage_grade"].value_counts(normalize=True).sort_index().round(3))


if __name__ == "__main__":
    main()