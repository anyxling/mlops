# src/scoringflow.py
from metaflow import FlowSpec, step
import mlflow.sklearn
from dataprocessing import load_data, preprocess_data
from sklearn.metrics import accuracy_score

# Define the Metaflow Scoring Flow
class ScoringFlow(FlowSpec):

    @step
    def start(self):
        # Load the data
        self.X, self.y = load_data()
        self.next(self.preprocess)

    @step
    def preprocess(self):
        # Preprocess the data (e.g., scaling or encoding)
        # Here we discard train data since only test data is needed
        _, self.X_test_scaled_df, _, self.y_test = preprocess_data(self.X, self.y)
        self.next(self.load_model)

    @step
    def load_model(self):
        # Set the MLflow tracking server URI
        mlflow.set_tracking_uri("https://mlflowgcrun-1052850358730.us-west2.run.app")

        # Define the URI of the registered model version
        model_uri = "models:/BestModelFromLastExp/1"  # change 1 to "Production" if promoted

        # Load the model from MLflow Model Registry
        self.model = mlflow.sklearn.load_model(model_uri)
        self.next(self.predict)

    @step
    def predict(self):
        # Use the loaded model to make predictions on the test set
        self.preds = self.model.predict(self.X_test_scaled_df)
        self.next(self.output)

    @step
    def output(self):
        # Display the first 10 predictions
        print("Predictions (first 10):")
        print(self.preds[:10])

        # Calculate and display accuracy
        acc = accuracy_score(self.y_test, self.preds)
        print(f"Accuracy: {acc:.4f}")
        
        self.next(self.end)
    
    @step
    def end(self):
        # Final step of the flow
        print("Scoring flow complete.")

# Entry point to run the flow
if __name__ == '__main__':
    ScoringFlow()
