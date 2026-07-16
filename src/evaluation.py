"""
==========================================================
Employee Attrition Machine Learning

evaluation.py

Author : Pramod Prakash Jadhav
==========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.config import CHARTS_DIR
from src.config import REPORTS_DIR
from src.logger import get_logger

logger = get_logger()


class ModelEvaluator:
    """
    Evaluate trained machine learning models.
    """

    def __init__(self, model):

        self.model = model

    # ======================================================
    # Evaluate Model
    # ======================================================

    def evaluate(self, X_test, y_test):

        logger.info("=" * 60)
        logger.info("Evaluating Model")
        logger.info("=" * 60)

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        try:
            probability = self.model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, probability)

        except Exception:

            roc_auc = None

        logger.info(f"Accuracy  : {accuracy:.4f}")
        logger.info(f"Precision : {precision:.4f}")
        logger.info(f"Recall    : {recall:.4f}")
        logger.info(f"F1 Score  : {f1:.4f}")

        if roc_auc is not None:
            logger.info(f"ROC-AUC   : {roc_auc:.4f}")

        self.save_report(y_test, y_pred)

        self.plot_confusion_matrix(y_test, y_pred)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
        }

    # ======================================================
    # Save Classification Report
    # ======================================================

    def save_report(self, y_test, y_pred):

        report = classification_report(
            y_test,
            y_pred
        )

        report_path = REPORTS_DIR / "classification_report.txt"

        with open(report_path, "w") as file:
            file.write(report)

        logger.info(
            f"Classification report saved: {report_path}"
        )

    # ======================================================
    # Plot Confusion Matrix
    # ======================================================

    def plot_confusion_matrix(self, y_test, y_pred):

        matrix = confusion_matrix(
            y_test,
            y_pred
        )

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title("Confusion Matrix")

        plt.xlabel("Predicted")

        plt.ylabel("Actual")

        output_path = CHARTS_DIR / "confusion_matrix.png"

        plt.tight_layout()

        plt.savefig(output_path)

        plt.close()

        logger.info(
            f"Confusion matrix saved: {output_path}"
        )


if __name__ == "__main__":

    from src.data_loader import load_data
    from src.preprocessing import DataPreprocessor
    from src.feature_engineering import FeatureEngineer
    from src.model_training import ModelTrainer

    df = load_data()

    processor = DataPreprocessor(df)

    clean_df = processor.process()

    engineer = FeatureEngineer(clean_df)

    X_train, X_test, y_train, y_test = engineer.prepare_data()

    trainer = ModelTrainer()

    model = trainer.train(
        X_train,
        y_train
    )

    evaluator = ModelEvaluator(model)

    evaluator.evaluate(
        X_test,
        y_test
)
