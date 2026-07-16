"""
==========================================================
Employee Attrition Machine Learning

test_model_training.py

Author : Pramod Prakash Jadhav
==========================================================
"""

from pathlib import Path

from src.data_loader import load_data
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.config import MODEL_PATH


def prepare_pipeline():
    """
    Prepare complete ML pipeline.
    """

    df = load_data()

    processor = DataPreprocessor(df)
    clean_df = processor.process()

    engineer = FeatureEngineer(clean_df)

    return engineer.prepare_data()


def test_model_training():

    """
    Test whether model trains successfully.
    """

    X_train, X_test, y_train, y_test = prepare_pipeline()

    trainer = ModelTrainer()

    model = trainer.train(
        X_train,
        y_train,
        X_test,
        y_test
    )

    assert model is not None


def test_model_has_predict():

    """
    Verify trained model supports prediction.
    """

    X_train, X_test, y_train, y_test = prepare_pipeline()

    trainer = ModelTrainer()

    model = trainer.train(
        X_train,
        y_train,
        X_test,
        y_test
    )

    assert hasattr(model, "predict")


def test_model_prediction():

    """
    Verify model predicts successfully.
    """

    X_train, X_test, y_train, y_test = prepare_pipeline()

    trainer = ModelTrainer()

    model = trainer.train(
        X_train,
        y_train,
        X_test,
        y_test
    )

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_model_saved():

    """
    Verify best_model.pkl is created.
    """

    X_train, X_test, y_train, y_test = prepare_pipeline()

    trainer = ModelTrainer()

    trainer.train(
        X_train,
        y_train,
        X_test,
        y_test
    )

    assert Path(MODEL_PATH).exists()


if __name__ == "__main__":

    test_model_training()
    test_model_has_predict()
    test_model_prediction()
    test_model_saved()

    print("All Model Training Tests Passed Successfully.")
