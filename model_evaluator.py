"""Model evaluation and visualization utilities.

This module provides comprehensive visualization and analysis tools for
evaluating trained machine learning models. The ModelEvaluator class generates
confusion matrices, feature importance plots, spam score prediction scatter plots,
and learning curves to help interpret model performance and behavior.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve


class ModelEvaluator:
    """
    Advanced model evaluation and visualization using matplotlib and seaborn.

    This class generates comprehensive visualizations that help interpret
    model performance and behavior. Includes confusion matrices, feature
    importance plots, spam score predictions scatter plots, and learning curves.
    """

    def __init__(self, tutorial_instance):
        """
        Initialize the evaluator with a trained tutorial instance.

        Args:
            tutorial_instance: EmailSpamMLTutorial with trained models.
        """
        self.tutorial = tutorial_instance
        self.setup_plotting()

    def setup_plotting(self) -> None:
        """
        Configure matplotlib and seaborn for consistent visualization styling.

        Sets up the plotting backend with professional aesthetics for better
        understanding of model performance.
        """
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def plot_confusion_matrix(self) -> None:
        """
        Generate a confusion matrix heatmap for spam/ham classification.

        Displays which categories are correctly classified and which are
        confused. Diagonal entries represent correct predictions.
        """
        print("Creating Confusion Matrix...")

        # Get predictions
        X_test_scaled = self.tutorial.feature_scaler.transform(self.tutorial.X_test)
        predictions = self.tutorial.spam_classifier.predict(X_test_scaled)

        # Create confusion matrix
        cm = confusion_matrix(self.tutorial.y_category_test, predictions)
        category_names = [
            self.tutorial.data_generator.get_category_name(i) for i in range(2)
        ]

        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=category_names,
            yticklabels=category_names,
        )
        plt.title("Spam/Ham Classification Confusion Matrix")
        plt.ylabel("Actual Category")
        plt.xlabel("Predicted Category")
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self) -> None:
        """
        Generate a bar plot showing feature importance in spam classification.

        RandomForest computes feature importance based on how much each feature
        contributes to reducing impurity. High importance indicates the feature
        is critical for distinguishing between spam and ham.
        """
        print("Plotting Feature Importance...")

        feature_names = [
            "Word Count",
            "Spam Words",
            "Ham Words",
            "Spam Ratio",
            "Has Urgency",
            "Has Money Words",
            "Has Exclamation",
            "Exclamation Count",
            "Avg Word Length",
            "Uppercase Ratio",
        ]

        importances = self.tutorial.spam_classifier.feature_importances_

        # Create plot
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)[::-1]  # Sort by importance

        plt.bar(range(len(importances)), importances[indices])
        plt.title("Feature Importance in Spam Classification")
        plt.xlabel("Features")
        plt.ylabel("Importance Score")
        plt.xticks(
            range(len(importances)), [feature_names[i] for i in indices], rotation=45
        )
        plt.tight_layout()
        plt.show()

    def plot_spam_score_predictions(self) -> None:
        """
        Generate a scatter plot comparing actual vs predicted spam scores.

        Points close to the diagonal red line indicate accurate predictions.
        Distance from the line indicates prediction error.
        """
        print("Plotting Spam Score Prediction Results...")

        X_test_scaled = self.tutorial.feature_scaler.transform(self.tutorial.X_test)
        spam_score_predictions = self.tutorial.spam_score_predictor.predict(
            X_test_scaled
        )

        plt.figure(figsize=(10, 6))
        plt.scatter(self.tutorial.y_spam_score_test, spam_score_predictions, alpha=0.6)
        plt.plot([0, 1], [0, 1], "r--", lw=2)  # Perfect prediction line
        plt.xlabel("Actual Spam Score")
        plt.ylabel("Predicted Spam Score")
        plt.title("Spam Score Prediction Accuracy")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_learning_curves(self) -> None:
        """
        Generate learning curves showing training vs validation performance.

        Displays how model accuracy improves as training set size increases.
        Helps diagnose whether the model is underfitting (high bias) or
        overfitting (high variance), and whether more training data would help.
        """
        print("Creating Learning Curves...")

        X_train_scaled = self.tutorial.feature_scaler.transform(self.tutorial.X_train)

        # Pylance may think this returns 5 values, so we slice the first 3
        lc_result = learning_curve(
            self.tutorial.spam_classifier,
            X_train_scaled,
            self.tutorial.y_category_train,
            cv=5,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
        )
        train_sizes, train_scores, val_scores = lc_result[:3]

        plt.figure(figsize=(10, 6))
        plt.plot(
            train_sizes, np.mean(train_scores, axis=1), "o-", label="Training Score"
        )
        plt.plot(
            train_sizes, np.mean(val_scores, axis=1), "o-", label="Validation Score"
        )
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy Score")
        plt.title("Learning Curves - Spam Classification")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def create_comprehensive_report(self) -> None:
        """
        Generate all visualizations and display comprehensive analysis report.

        Creates confusion matrix, feature importance, spam score predictions, and
        learning curves visualizations. Provides guidance on interpreting results.
        """
        print("\nCOMPREHENSIVE MODEL ANALYSIS REPORT")
        print("=" * 60)

        # Run all visualizations
        self.plot_confusion_matrix()
        self.plot_feature_importance()
        self.plot_spam_score_predictions()
        self.plot_learning_curves()

        print("All visualizations complete!")
        print("Key insights to look for:")
        print("   Confusion Matrix: How well does it distinguish spam from ham?")
        print("   Feature Importance: Which text patterns matter most?")
        print("   Spam Score Predictions: How accurate are our spam scores?")
        print("   Learning Curves: Do we need more training data?")


# Usage example
if __name__ == "__main__":
    from model_trainer import EmailSpamMLTutorial

    # Run the tutorial first
    tutorial = EmailSpamMLTutorial()
    tutorial.run_complete_tutorial()

    # Then create advanced analysis
    evaluator = ModelEvaluator(tutorial)
    evaluator.create_comprehensive_report()
