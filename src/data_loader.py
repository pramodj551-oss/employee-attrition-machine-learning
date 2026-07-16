"""
==========================================================
Employee Attrition Machine Learning

data_loader.py

Author : Pramod Prakash Jadhav
==========================================================
"""

import pandas as pd

from src.config import DATASET_PATH
from src.logger import get_logger

logger = get_logger()


def load_data() -> pd.DataFrame:
    """
    Load the employee attrition dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    Exception
        If any other error occurs while loading.
    """

    try:

        logger.info("Loading dataset...")

        if not DATASET_PATH.exists():

            raise FileNotFoundError(
                f"Dataset not found: {DATASET_PATH}"
            )

        df = pd.read_csv(DATASET_PATH)

        logger.info("Dataset loaded successfully.")

        logger.info(f"Rows : {df.shape[0]}")
        logger.info(f"Columns : {df.shape[1]}")

        return df

    except FileNotFoundError as error:

        logger.error(error)

        raise

    except Exception as error:

        logger.exception(
            f"Error while loading dataset: {error}"
        )

        raise


def dataset_summary(df: pd.DataFrame) -> None:
    """
    Print and log dataset summary.
    """

    logger.info("=" * 60)
    logger.info("Dataset Summary")
    logger.info("=" * 60)

    logger.info(f"Shape : {df.shape}")

    logger.info(f"Columns : {list(df.columns)}")

    logger.info(
        f"Missing Values : {df.isnull().sum().sum()}"
    )

    logger.info(
        f"Duplicate Rows : {df.duplicated().sum()}"
    )


if __name__ == "__main__":

    dataframe = load_data()

    dataset_summary(dataframe)

    print(dataframe.head())
