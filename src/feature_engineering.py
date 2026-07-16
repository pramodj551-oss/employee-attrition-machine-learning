"""
==========================================================
Employee Attrition Machine Learning

feature_engineering.py

Author : Pramod Prakash Jadhav
==========================================================
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE
from src.config import TARGET_COLUMN
from src.config import TEST_SIZE
from src.logger import get_logger

logger = get_logger()


class FeatureEngineer:
    """
    Performs feature engineering and prepares
    machine learning datasets.
    """

    def __init__(self, dataframe: pd.DataFrame):

        self.df = dataframe.copy()

        self.scaler = StandardScaler()

    # ======================================================
    # Encode Target Variable
    # ======================================================

    def encode_target(self):

        logger.info("Encoding target variable...")

        encoder = LabelEncoder()

        self.df[TARGET_COLUMN] = encoder.fit_transform(
            self.df[TARGET_COLUMN]
        )

    # ======================================================
    # Encode Categorical Features
    # ======================================================

    def encode_features(self):

        logger.info("Encoding categorical features...")

        categorical_columns = self.df.select_dtypes(
            include=["object"]
        ).columns.tolist()

        if TARGET_COLUMN in categorical_columns:

            categorical_columns.remove(TARGET_COLUMN)

        self.df = pd.get_dummies(
            self.df,
            columns=categorical_columns,
            drop_first=True
        )

    # ======================================================
    # Split Features & Target
    # ======================================================

    def split_features_target(self):

        X = self.df.drop(
            columns=[TARGET_COLUMN]
        )

        y = self.df[TARGET_COLUMN]

        return X, y

    # ======================================================
    # Train Test Split
    # ======================================================

    def train_test(self, X, y):

        logger.info("Splitting dataset...")

        return train_test_split(

            X,

            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=y

        )

    # ======================================================
    # Feature Scaling
    # ======================================================

    def scale_features(

        self,

        X_train,

        X_test

    ):

        logger.info("Scaling features...")

        X_train_scaled = self.scaler.fit_transform(
            X_train
        )

        X_test_scaled = self.scaler.transform(
            X_test
        )

        return X_train_scaled, X_test_scaled

    # ======================================================
    # Complete Pipeline
    # ======================================================

    def prepare_data(self):

        logger.info("=" * 60)

        logger.info(
            "Starting Feature Engineering..."
        )

        logger.info("=" * 60)

        self.encode_target()

        self.encode_features()

        X, y = self.split_features_target()

        X_train, X_test, y_train, y_test = self.train_test(
            X,
            y
        )

        X_train, X_test = self.scale_features(
            X_train,
            X_test
        )

        logger.info(
            "Feature engineering completed successfully."
        )

        logger.info(
            f"Training Samples : {X_train.shape[0]}"
        )

        logger.info(
            f"Testing Samples : {X_test.shape[0]}"
        )

        logger.info(
            f"Number of Features : {X_train.shape[1]}"
        )

        return (

            X_train,

            X_test,

            y_train,

            y_test

        )


if __name__ == "__main__":

    from src.data_loader import load_data
    from src.preprocessing import DataPreprocessor

    df = load_data()

    processor = DataPreprocessor(df)

    clean_df = processor.process()

    engineer = FeatureEngineer(clean_df)

    X_train, X_test, y_train, y_test = engineer.prepare_data()

    print("Training Shape :", X_train.shape)

    print("Testing Shape :", X_test.shape)
  
