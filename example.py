from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
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
        uri = model.train()  # This trains the model and logs all details
        print(uri)
