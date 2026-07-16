"""
==========================================================
Employee Attrition Machine Learning
main.py

Author : Pramod Prakash Jadhav
==========================================================
"""

from src.logger import get_logger
from src.data_loader import load_data
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator
from src.prediction import Predictor


def main():

    logger = get_logger()

    logger.info("=" * 60)
    logger.info("Employee Attrition Machine Learning Started")
    logger.info("=" * 60)

    try:

        # ---------------------------------------------
        # Step 1 : Load Dataset
        # ---------------------------------------------
        logger.info("Loading dataset...")

        df = load_data()

        logger.info(f"Dataset Shape : {df.shape}")

        # ---------------------------------------------
        # Step 2 : Data Preprocessing
        # ---------------------------------------------
        logger.info("Running preprocessing...")

        processor = DataPreprocessor(df)

        clean_df = processor.process()

        logger.info(f"Processed Shape : {clean_df.shape}")

        # ---------------------------------------------
        # Step 3 : Feature Engineering
        # ---------------------------------------------
        logger.info("Performing feature engineering...")

        engineer = FeatureEngineer(clean_df)

        X_train, X_test, y_train, y_test = engineer.prepare_data()

        # ---------------------------------------------
        # Step 4 : Model Training
        # ---------------------------------------------
        logger.info("Training Machine Learning models...")

        trainer = ModelTrainer()

        best_model = trainer.train(
            X_train,
            y_train
        )

        # ---------------------------------------------
        # Step 5 : Model Evaluation
        # ---------------------------------------------
        logger.info("Evaluating model...")

        evaluator = ModelEvaluator(best_model)

        evaluator.evaluate(
            X_test,
            y_test
        )

        # ---------------------------------------------
        # Step 6 : Prediction
        # ---------------------------------------------
        logger.info("Generating predictions...")

        predictor = Predictor(best_model)

        predictions = predictor.predict(X_test)

        logger.info(
            f"Generated {len(predictions)} predictions."
        )

        logger.info("=" * 60)
        logger.info("Machine Learning Pipeline Completed")
        logger.info("=" * 60)

    except Exception as error:

        logger.exception(f"Pipeline Failed : {error}")


if __name__ == "__main__":

    main()
