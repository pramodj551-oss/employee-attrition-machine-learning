"""
==========================================================
Employee Attrition Machine Learning

test_prediction.py

Author : Pramod Prakash Jadhav
==========================================================
"""

from pathlib import Path

import pandas as pd

from src.data_loader import load_data
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.prediction import Predictor
from src.config import REPORTS_DIR


def prepare_pipeline():
    """
    Prepare complete pipeline and return trained model
    with test dataset.
    """

    df = load_data()

    processor = DataPreprocessor(df)

    clean_df = processor.process()

    engineer = FeatureEngineer(clean_df)

    X_train, X_test, y_train, y_test = engineer.prepare_data()

    trainer = ModelTrainer()

    model = trainer.train(
        X_train,
        y_train,
        X_test,
        y_test
    )

    return model, X_test, y_test


def test_prediction():

    """
    Verify predictions are generated.
    """

    model, X_test, y_test = prepare_pipeline()

    predictor = Predictor(model)

    predictions = predictor.predict(X_test)

    assert len(predictions) == len(y_test)


def test_prediction_probability():

    """
    Verify prediction probabilities.
    """

    model, X_test, _ = prepare_pipeline()

    predictor = Predictor(model)

    probability = predictor.predict_probability(X_test)

    if probability is not None:

        assert probability.shape[0] == X_test.shape[0]


def test_prediction_csv():

    """
    Verify predictions.csv is created.
    """

    model, X_test, _ = prepare_pipeline()

    predictor = Predictor(model)

    predictor.save_predictions(X_test)

    output_file = REPORTS_DIR / "predictions.csv"

    assert Path(output_file).exists()


def test_prediction_dataframe():

    """
    Verify prediction output dataframe.
    """

    model, X_test, _ = prepare_pipeline()

    predictor = Predictor(model)

    results = predictor.save_predictions(X_test)

    assert isinstance(results, pd.DataFrame)

    assert "Prediction" in results.columns


if __name__ == "__main__":

    test_prediction()

    test_prediction_probability()

    test_prediction_csv()

    test_prediction_dataframe()

    print("All Prediction Tests Passed Successfully.")
