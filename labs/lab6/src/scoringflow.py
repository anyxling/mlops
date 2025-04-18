# src/scoringflow.py
from metaflow import FlowSpec, step
import mlflow.sklearn
from dataprocessing import load_data, preprocess_data
from sklearn.metrics import accuracy_score

class ScoringFlow(FlowSpec):

    @step
    def start(self):
        self.X, self.y = load_data()
        self.next(self.preprocess)

    @step
    def preprocess(self):
        _, self.X_test_scaled_df, _, self.y_test = preprocess_data(self.X, self.y)
        self.next(self.load_model)

    @step
    def load_model(self):
        mlflow.set_tracking_uri("https://mlflowgcrun-1052850358730.us-west2.run.app")
        model_uri = "models:/BestModelFromLastExp/1" 

        self.model = mlflow.sklearn.load_model(model_uri)
        self.next(self.predict)

    @step
    def predict(self):
        self.preds = self.model.predict(self.X_test_scaled_df)
        self.next(self.output)

    @step
    def output(self):
        print("Predictions (first 10):")
        print(self.preds[:10])
        acc = accuracy_score(self.y_test, self.preds)
        print(f"Accuracy: {acc:.4f}")
        self.next(self.end)
    
    @step
    def end(self):
        print("Scoring flow complete.")

if __name__ == '__main__':
    ScoringFlow()
