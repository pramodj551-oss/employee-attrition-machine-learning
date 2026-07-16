"""
==========================================================
Employee Attrition Machine Learning

preprocessing.py

Author : Pramod Prakash Jadhav
==========================================================
"""

import pandas as pd

from src.logger import get_logger

logger = get_logger()


class DataPreprocessor:
    """
    Production-ready data preprocessing class.
    """

    def __init__(self, dataframe: pd.DataFrame):

        self.df = dataframe.copy()

    # ======================================================
    # Remove Duplicate Records
    # ======================================================

    def remove_duplicates(self):

        duplicate_count = self.df.duplicated().sum()

        if duplicate_count > 0:

            self.df.drop_duplicates(inplace=True)

            logger.info(
                f"Removed {duplicate_count} duplicate rows."
            )

        else:

            logger.info("No duplicate records found.")

    # ======================================================
    # Handle Missing Values
    # ======================================================

    def handle_missing_values(self):

        missing = self.df.isnull().sum().sum()

        if missing == 0:

            logger.info("No missing values found.")

            return

        numeric_columns = self.df.select_dtypes(
            include=["number"]
        ).columns

        categorical_columns = self.df.select_dtypes(
            include=["object"]
        ).columns

        self.df[numeric_columns] = self.df[
            numeric_columns
        ].fillna(
            self.df[numeric_columns].median()
        )

        for column in categorical_columns:

            mode = self.df[column].mode()

            if not mode.empty:

                self.df[column] = self.df[column].fillna(
                    mode[0]
                )

        logger.info(
            "Missing values handled successfully."
        )

    # ======================================================
    # Validate Target Column
    # ======================================================

    def validate_target(self):

        if "Attrition" not in self.df.columns:

            raise ValueError(
                "Target column 'Attrition' not found."
            )

        logger.info(
            "Target column validated successfully."
        )

    # ======================================================
    # Remove Constant Columns
    # ======================================================

    def remove_constant_columns(self):

        constant_columns = []

        for column in self.df.columns:

            if self.df[column].nunique() == 1:

                constant_columns.append(column)

        if constant_columns:

            self.df.drop(
                columns=constant_columns,
                inplace=True
            )

            logger.info(
                f"Removed constant columns: {constant_columns}"
            )

        else:

            logger.info("No constant columns found.")

    # ======================================================
    # Validate Data Types
    # ======================================================

    def validate_dataframe(self):

        if self.df.empty:

            raise ValueError("Dataset is empty.")

        logger.info("Dataset validation successful.")

    # ======================================================
    # Complete Pipeline
    # ======================================================

    def process(self):

        logger.info("=" * 60)
        logger.info("Starting preprocessing...")
        logger.info("=" * 60)

        self.validate_dataframe()

        self.remove_duplicates()

        self.handle_missing_values()

        self.validate_target()

        self.remove_constant_columns()

        logger.info("Preprocessing completed.")

        logger.info(f"Final Shape : {self.df.shape}")

        return self.df


if __name__ == "__main__":

    from src.data_loader import load_data

    dataframe = load_data()

    processor = DataPreprocessor(dataframe)

    cleaned_df = processor.process()

    print(cleaned_df.head())
