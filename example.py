from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
from model_package.standardmodel import SelfContainedModel 

# Example usage with a RandomForestClassifier
class ExampleClassificationModel(SelfContainedModel):
    def __init__(self):
        super().__init__(pip_requirements = ["pandas"])

    def gen_data(self):
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def get_model(self):
        return RandomForestClassifier(n_estimators=100, random_state=42)

if __name__ == "__main__":
        model = ExampleClassificationModel()
        model_uri = model.train()  # Trains the model and logs all details

        # Example of how to load the model from MLflow
        model_uri = model_uri.replace("/artifacts/model","")
        print(model_uri)
        loaded_model = mlflow.pyfunc.load_model(model_uri)

        # Example of loading a data row - modify as necessary to load actual data
        X_train, X_test, y_train, y_test = model.gen_data()
        data_row = X_test.iloc[0:1]  # Selects the first row of the test data

        # Making a prediction
        prediction = loaded_model.predict(data_row)
        print(f"Prediction for the input data: {prediction}")

