"""
==========================================================
Employee Attrition Machine Learning

model_training.py

Author : Pramod Prakash Jadhav
==========================================================
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from src.config import MODEL_PATH
from src.config import RANDOM_STATE
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
                random_state=RANDOM_STATE,
                max_iter=1000
            ),

            "Decision Tree": DecisionTreeClassifier(
                random_state=RANDOM_STATE
            ),

            "Random Forest": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_estimators=100
            ),

            "Gradient Boosting": GradientBoostingClassifier(
                random_state=RANDOM_STATE
            ),

            "K-Nearest Neighbors": KNeighborsClassifier()

        }

    # ======================================================
    # Train All Models
    # ======================================================

    def train(
        self,
        X_train,
        y_train
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

            train_accuracy = accuracy_score(
                y_train,
                train_prediction
            )

            logger.info(
                f"{model_name} Training Accuracy : "
                f"{train_accuracy:.4f}"
            )

            if train_accuracy > best_score:

                best_score = train_accuracy

                best_model = model

                best_model_name = model_name

        logger.info("=" * 60)

        logger.info(
            f"Best Model : {best_model_name}"
        )

        logger.info(
            f"Best Accuracy : {best_score:.4f}"
        )

        logger.info("=" * 60)

        save_model(
            best_model,
            MODEL_PATH
        )

        logger.info(
            f"Best model saved to : {MODEL_PATH}"
        )

        return best_model


if __name__ == "__main__":

    from src.data_loader import load_data
    from src.preprocessing import DataPreprocessor
    from src.feature_engineering import FeatureEngineer

    df = load_data()

    processor = DataPreprocessor(df)

    clean_df = processor.process()

    engineer = FeatureEngineer(clean_df)

    X_train, X_test, y_train, y_test = engineer.prepare_data()

    trainer = ModelTrainer()

    trainer.train(
        X_train,
        y_train
  )
