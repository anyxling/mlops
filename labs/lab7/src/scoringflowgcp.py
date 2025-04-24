# src/scoringflowgcp.py
from metaflow import FlowSpec, step, resources, timeout, retry, catch, conda_base
import mlflow.sklearn
from dataprocessing import load_data, preprocess_data
from sklearn.metrics import accuracy_score

@conda_base(libraries={'mlflow': '2.11.1', 'scikit-learn': '1.2.2'}, python='3.9.16')
class ScoringFlow(FlowSpec):

    @step
    @resources(cpu=2, memory=4096)
    @timeout(seconds=300)
    @retry(times=2)
    @catch(var='load_error')
    def start(self):
        self.X, self.y = load_data()
        self.next(self.preprocess)

    @step
    @resources(cpu=2, memory=4096)
    def preprocess(self):
        _, self.X_test_scaled_df, _, self.y_test = preprocess_data(self.X, self.y)
        self.next(self.load_model)

    @step
    @timeout(seconds=180)
    def load_model(self):
        mlflow.set_tracking_uri("https://mlflowlab7-905802433874.us-west2.run.app")
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