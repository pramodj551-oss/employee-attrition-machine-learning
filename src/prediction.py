"""
==========================================================
Employee Attrition Machine Learning

prediction.py

Author : Pramod Prakash Jadhav
==========================================================
"""

from pathlib import Path

import pandas as pd

from src.config import MODEL_PATH, REPORTS_DIR
from src.logger import get_logger
from src.utils import load_model

logger = get_logger()


class Predictor:
    """
    Prediction pipeline for Employee Attrition.

    Supports:
    - Batch Prediction
    - Prediction Probabilities
    - Save Predictions
    """

    def __init__(self, model=None):

        if model is None:

            logger.info("Loading trained model...")

            self.model = load_model(MODEL_PATH)

        else:

            self.model = model

    # ======================================================
    # Predict Labels
    # ======================================================

    def predict(self, X):

        logger.info("Generating predictions...")

        predictions = self.model.predict(X)

        logger.info(
            f"Generated {len(predictions)} predictions."
        )

        return predictions

    # ======================================================
    # Predict Probability
    # ======================================================

    def predict_probability(self, X):

        if hasattr(self.model, "predict_proba"):

            logger.info(
                "Generating prediction probabilities..."
            )

            probabilities = self.model.predict_proba(X)

            return probabilities

        logger.warning(
            "Selected model does not support predict_proba()."
        )

        return None

    # ======================================================
    # Save Predictions
    # ======================================================

    def save_predictions(self, X):

        predictions = self.predict(X)

        results = pd.DataFrame({

            "Prediction": predictions

        })

        if hasattr(self.model, "predict_proba"):

            probability = self.predict_proba(X)

            results["Probability_No"] = probability[:, 0]

            results["Probability_Yes"] = probability[:, 1]

        output_file = REPORTS_DIR / "predictions.csv"

        results.to_csv(

            output_file,

            index=False

        )

        logger.info(
            f"Predictions saved to: {output_file}"
        )

        return results


# ==========================================================
# Test Prediction Module
# ==========================================================

if __name__ == "__main__":

    from src.data_loader import load_data
    from src.preprocessing import DataPreprocessor
    from src.feature_engineering import FeatureEngineer

    dataframe = load_data()

    processor = DataPreprocessor(dataframe)

    clean_df = processor.process()

    engineer = FeatureEngineer(clean_df)

    X_train, X_test, y_train, y_test = engineer.prepare_data()

    predictor = Predictor()

    prediction_df = predictor.save_predictions(X_test)

    print(prediction_df.head())
