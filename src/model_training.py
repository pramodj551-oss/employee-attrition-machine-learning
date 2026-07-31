"""
Employee Attrition Machine Learning
model_training.py
Author : Pramod Prakash Jadhav
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from src.config import (
    MODEL_PATH,
    RANDOM_STATE,
    FEATURE_COLUMNS_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
)
from src.logger import get_logger
from src.utils import save_model

logger = get_logger()


class ModelTrainer:
    """
    Train multiple machine learning models,
    compare their performance,
    and save the best model.
    """

    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(
                random_state=RANDOM_STATE, max_iter=1000
            ),
            "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(
                random_state=RANDOM_STATE, n_estimators=100
            ),
            "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "K-Nearest Neighbors": KNeighborsClassifier(),
        }

    # ======================================================
    # Train All Models
    # ======================================================
    def train(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        scaler=None,
        label_encoder=None,
        feature_engineer=None,
    ):
        logger.info("=" * 60)
        logger.info("Training Machine Learning Models")
        logger.info("=" * 60)

        best_model = None
        best_score = 0.0
        best_model_name = ""

        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}")

            model.fit(X_train, y_train)

            train_prediction = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_prediction)

            test_prediction = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_prediction)

            logger.info(f"{model_name} Training Accuracy : {train_accuracy:.4f}")
            logger.info(f"{model_name} Test Accuracy : {test_accuracy:.4f}")

            if test_accuracy > best_score:
                best_score = test_accuracy
                best_model = model
                best_model_name = model_name

        logger.info("=" * 60)
        logger.info(f"Best Model : {best_model_name}")
        logger.info(f"Best Accuracy : {best_score:.4f}")
        logger.info("=" * 60)

        # --- Save best model ---
        save_model(best_model, MODEL_PATH)
        logger.info(f"Best model saved to : {MODEL_PATH}")

        # --- Save preprocessing artifacts (single source of truth: feature_engineer) ---
        joblib.dump(X_train.columns.tolist(), FEATURE_COLUMNS_PATH)
        logger.info(f"Feature columns saved : {FEATURE_COLUMNS_PATH}")

        if feature_engineer is not None:
            joblib.dump(feature_engineer.scaler, SCALER_PATH)
            joblib.dump(feature_engineer.label_encoder, LABEL_ENCODER_PATH)
            logger.info("Preprocessing artifacts saved successfully.")
        else:
            if scaler is not None:
                joblib.dump(scaler, SCALER_PATH)
                logger.info(f"Scaler saved : {SCALER_PATH}")
            if label_encoder is not None:
                joblib.dump(label_encoder, LABEL_ENCODER_PATH)
                logger.info(f"Label Encoder saved : {LABEL_ENCODER_PATH}")

        return best_model


if __name__ == "__main__":
    from src.data_loader import load_data

    dataframe = load_data()
    processor = DataPreprocessor(dataframe)
    cleaned_df = processor.process()
    print(cleaned_df.head())
    df = load_data()
    processor = DataPreprocessor(df)
    clean_df = processor.process()

    engineer = FeatureEngineer(clean_df)
    X_train, X_test, y_train, y_test = engineer.prepare_data()

    trainer = ModelTrainer()
    model = trainer.train(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_engineer=engineer,
                                                           )
