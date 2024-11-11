import mlflow
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from pycm import ConfusionMatrix
from mlflow.models.signature import infer_signature
import os
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    mean_squared_error, r2_score
)
import numpy as np
import pandas as pd
import joblib

class SelfContainedModel(mlflow.pyfunc.PythonModel, ABC):

    def __init__(self, pip_requirements):
        super().__init__()
        self.model = None
        self.model_type = None  # This will be inferred
        self.pip_requirements = pip_requirements + ["pycm", "matplotlib", "scikit-learn", "joblib"]

    @abstractmethod
    def gen_data(self):
        """Generate and preprocess data."""
        pass

    @abstractmethod
    def get_model(self):
        """Train the model using generated data."""
        pass

#    def load_context(self, context):
#        self.model = joblib.load(context.artifacts['model'])

    def train(self):
        X_train, X_test, y_train, y_test = self.gen_data()
        self.model = self.get_model()
        self.model.fit(X_train, y_train)
        return self.validate_and_log(self.model, X_train, X_test, y_test)
    
    def predict(self, context, model_input, predict_proba=False):
        """Predict using the preprocessed data, handle both probability and label predictions."""
        if predict_proba and hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(model_input)
        return self.model.predict(model_input)

    def infer_task_type(self, y):
        """Infer the task type (regression or classification) based on y."""
        if pd.api.types.is_numeric_dtype(y) and len(pd.unique(y)) > len(y) * 0.1:
            self.model_type = "regression"
        else:
            self.model_type = "classification"

    def log_metrics_and_params(self, params, metrics):
        """Logs both metrics and parameters to MLflow."""
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

    def plot_and_save_precision_recall_curve(self, y_test, predictions_proba, output_path="precision_recall_curve.png"):
        """Plots and saves precision-recall curve for multi-class classification."""
        plt.figure(figsize=(10, 8))
        classes = np.unique(y_test)
        for i, class_label in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_test == class_label, predictions_proba[:, i])
            ap = average_precision_score(y_test == class_label, predictions_proba[:, i])
            plt.plot(recall, precision, lw=2, label=f'Class {class_label} (AP={ap:.2f})')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title("Precision-Recall curve (Multi-Class)")
        plt.legend(loc="best")
        plt.savefig(output_path)
        plt.close()

    def plot_and_save_lift_chart(self, y_test, predictions_proba, output_path="lift_chart.png"):
        """Plots and saves a lift chart for multi-class classification."""
        plt.figure(figsize=(10, 8))
        classes = np.unique(y_test)
        for i, class_label in enumerate(classes):
            sorted_probs = np.sort(predictions_proba[:, i])[::-1]
            sorted_y_test = np.array(y_test == class_label)[np.argsort(-predictions_proba[:, i])]
            cumulative_positives = np.cumsum(sorted_y_test)
            total_positives = cumulative_positives[-1]
            lift = cumulative_positives / (np.arange(1, len(sorted_y_test) + 1) * (total_positives / len(sorted_y_test)))
            
            plt.plot(np.arange(1, len(lift) + 1) / len(sorted_y_test), lift, lw=2, label=f'Class {class_label}')
        
        plt.xlabel("Proportion of Sample")
        plt.ylabel("Lift")
        plt.ylim([0.0, max(2, plt.ylim()[1])])  # Adjust y-axis dynamically
        plt.xlim([0.0, 1.0])
        plt.title("Lift Chart (Multi-Class)")
        plt.legend(loc="best")
        plt.savefig(output_path)
        plt.close()

    def log_classification_artifacts(self, y_test, predictions_proba, predicted_labels):

        """Generates and logs the precision-recall curve, lift chart, and confusion matrix for classification models."""
        # Plot and save the precision-recall curve for multi-class
        pr_curve_path = "precision_recall_curve.png"
        self.plot_and_save_precision_recall_curve(y_test, predictions_proba, output_path=pr_curve_path)
        mlflow.log_artifact(pr_curve_path)
        os.remove(pr_curve_path)

        # Plot and save the lift chart for multi-class
        lift_chart_path = "lift_chart.png"
        self.plot_and_save_lift_chart(y_test, predictions_proba, output_path=lift_chart_path)
        mlflow.log_artifact(lift_chart_path)
        os.remove(lift_chart_path)

        # Generate and save the confusion matrix
        cm = ConfusionMatrix(actual_vector=y_test.tolist(), predict_vector=predicted_labels.tolist())
        cm_path = "confusion_matrix"
        cm.save_html(cm_path)
        cm_path += ".html"
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

    def validate_and_log(self, model, X_train, X_test, y_test):
        self.infer_task_type(y_test)
        predictions = self.predict(None, X_test, predict_proba=(self.model_type == "classification"))
        metrics = {}
        
        with mlflow.start_run():
            if self.model_type == "classification":
                predicted_labels = self.predict(None, X_test)
                predictions_proba = predictions

                # Compute metrics for multi-class classification
                metrics.update({
                    'Accuracy': accuracy_score(y_test, predicted_labels),
                    'F1 Score': f1_score(y_test, predicted_labels, average='weighted'),
                    'Precision': precision_score(y_test, predicted_labels, average='weighted'),
                    'Recall': recall_score(y_test, predicted_labels, average='weighted'),
                    'AUC': roc_auc_score(y_test, predictions_proba[:,1], multi_class='ovr'),
                    'Average Precision': average_precision_score(y_test, predictions_proba[:, 1], average='weighted')
                })

                # Log classification-specific artifacts
                self.log_classification_artifacts(y_test, predictions_proba, predicted_labels)

            elif self.model_type == "regression":
                # Compute regression metrics
                metrics.update({
                    'MSE': mean_squared_error(y_test, predictions),
                    'R^2': r2_score(y_test, predictions)
                })

            # Log metrics and parameters
            self.log_metrics_and_params(model.get_params(), metrics)

            sample_size = 10000
            train_sample = X_train.sample(n=min(sample_size, len(X_train)), random_state=42)

            # Save the sample as a CSV file
            sample_path = "train_data_sample.csv"
            train_sample.to_csv(sample_path, index=False)
            mlflow.log_artifact(sample_path)
            os.remove(sample_path)

            # Example input and model signature
            input_example = X_test.iloc[0:1]
            signature = infer_signature(input_example, predictions)

            # Log the model
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=self,
                pip_requirements=self.pip_requirements,
                signature=signature
            )

            # Print the model URI for easy access
            model_uri = mlflow.get_artifact_uri("model")
            return model_uri

