"""
==========================================================
Employee Attrition Machine Learning

test_data_loader.py

Author : Pramod Prakash Jadhav
==========================================================
"""

import pandas as pd

from src.data_loader import load_data


def test_load_data():
    """
    Test whether the dataset is loaded successfully.
    """

    df = load_data()

    assert isinstance(df, pd.DataFrame)


def test_dataset_not_empty():
    """
    Test whether the dataset is empty.
    """

    df = load_data()

    assert len(df) > 0


def test_target_column_exists():
    """
    Test whether target column exists.
    """

    df = load_data()

    assert "Attrition" in df.columns


def test_expected_columns():
    """
    Verify dataset has expected number of columns.
    IBM HR Attrition dataset contains 35 columns.
    """

    df = load_data()

    assert df.shape[1] == 35


def test_no_duplicate_column_names():
    """
    Verify duplicate column names do not exist.
    """

    df = load_data()

    assert df.columns.duplicated().sum() == 0


if __name__ == "__main__":

    test_load_data()
    test_dataset_not_empty()
    test_target_column_exists()
    test_expected_columns()
    test_no_duplicate_column_names()

    print("All Data Loader Tests Passed Successfully.")
