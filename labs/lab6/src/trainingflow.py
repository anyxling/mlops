# src/trainingflow.py
from metaflow import FlowSpec, step, Parameter
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from dataprocessing import load_data, preprocess_data
from xgboost import XGBClassifier

# Define the training flow using Metaflow
class TrainModelFlow(FlowSpec):

    # Define a configurable seed parameter
    seed = Parameter('seed', default=42)

    @step
    def start(self):
        # Load raw dataset
        self.X, self.y = load_data()
        self.next(self.preprocess)

    @step
    def preprocess(self):
        # Preprocess the data and split into train/test sets
        self.X_train_scaled_df, self.X_test_scaled_df, self.y_train, self.y_test = preprocess_data(self.X, self.y)
        self.next(self.fetch_best_params)

    @step
    def fetch_best_params(self):
        # Connect to your GCP-hosted MLflow tracking server
        mlflow.set_tracking_uri("https://mlflowgcrun-1052850358730.us-west2.run.app")

        # Use MlflowClient to fetch the latest registered model version
        client = MlflowClient()
        best_model = client.get_latest_versions("dry_bean_best_xgb", stages=["None"])[0]  # fallback if no Production

        # Get the run associated with the best model
        run = client.get_run(best_model.run_id)
        params = run.data.params

        # Extract and cast hyperparameters
        self.max_depth = int(params["max_depth"])
        self.n_estimators = int(params["n_estimators"])
        self.subsample = float(params["subsample"])

        self.next(self.train)

    @step
    def train(self):
        # Initialize and train an XGBoost classifier using the fetched parameters
        self.model = XGBClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            subsample=self.subsample
        )
        self.model.fit(self.X_train_scaled_df, self.y_train)
        self.next(self.register)

    @step
    def register(self):
        # Set the MLflow tracking URI and experiment name
        mlflow.set_tracking_uri("https://mlflowgcrun-1052850358730.us-west2.run.app")
        mlflow.set_experiment("metaflow_training")

        # Log model, parameters, and metadata to MLflow
        with mlflow.start_run():
            mlflow.log_params({
                'max_depth': self.max_depth,
                'n_estimators': self.n_estimators,
                'subsample': self.subsample
            })
            mlflow.sklearn.log_model(
                self.model,
                artifact_path="model",
                registered_model_name="BestModelFromLastExp"
            )

        self.next(self.end)

    @step
    def end(self):
        # Final step to indicate completion
        print("Training flow complete. Model registered.")

# Entry point to run the training flow
if __name__ == '__main__':
    TrainModelFlow()
