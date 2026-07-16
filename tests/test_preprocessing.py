"""
==========================================================
Employee Attrition Machine Learning

test_preprocessing.py

Author : Pramod Prakash Jadhav
==========================================================
"""

import pandas as pd

from src.data_loader import load_data
from src.preprocessing import DataPreprocessor


def test_preprocessing_returns_dataframe():
    """
    Test whether preprocessing returns a DataFrame.
    """

    df = load_data()

    processor = DataPreprocessor(df)

    processed_df = processor.process()

    assert isinstance(processed_df, pd.DataFrame)


def test_preprocessing_not_empty():
    """
    Test whether processed dataset is not empty.
    """

    df = load_data()

    processor = DataPreprocessor(df)

    processed_df = processor.process()

    assert not processed_df.empty


def test_target_column_exists():
    """
    Test whether Attrition column exists.
    """

    df = load_data()

    processor = DataPreprocessor(df)

    processed_df = processor.process()

    assert "Attrition" in processed_df.columns


def test_no_missing_values():
    """
    Test whether processed dataset contains missing values.
    """

    df = load_data()

    processor = DataPreprocessor(df)

    processed_df = processor.process()

    assert processed_df.isnull().sum().sum() == 0


def test_no_duplicate_rows():
    """
    Test whether duplicate rows are removed.
    """

    df = load_data()

    processor = DataPreprocessor(df)

    processed_df = processor.process()

    assert processed_df.duplicated().sum() == 0


if __name__ == "__main__":

    test_preprocessing_returns_dataframe()
    test_preprocessing_not_empty()
    test_target_column_exists()
    test_no_missing_values()
    test_no_duplicate_rows()

    print("All Preprocessing Tests Passed Successfully.")
