"""
==========================================================
Employee Attrition Machine Learning

test_feature_engineering.py

Author : Pramod Prakash Jadhav
==========================================================
"""

import numpy as np

from src.data_loader import load_data
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer


def prepare_data():

    """
    Prepare dataset for testing.
    """

    df = load_data()

    processor = DataPreprocessor(df)

    clean_df = processor.process()

    engineer = FeatureEngineer(clean_df)

    return engineer.prepare_data()


def test_train_test_split():

    """
    Verify train-test split.
    """

    X_train, X_test, y_train, y_test = prepare_data()

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0


def test_feature_count():

    """
    Verify train and test have same number of features.
    """

    X_train, X_test, _, _ = prepare_data()

    assert X_train.shape[1] == X_test.shape[1]


def test_target_values():

    """
    Verify encoded target values are binary.
    """

    _, _, y_train, y_test = prepare_data()

    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})


def test_scaled_features():

    """
    Verify scaled feature arrays.
    """

    X_train, X_test, _, _ = prepare_data()

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)


def test_matching_samples():

    """
    Verify sample counts match.
    """

    X_train, X_test, y_train, y_test = prepare_data()

    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)


if __name__ == "__main__":

    test_train_test_split()
    test_feature_count()
    test_target_values()
    test_scaled_features()
    test_matching_samples()

    print("All Feature Engineering Tests Passed Successfully.")
  
