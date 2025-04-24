# src/trainingflowgcp.py
from metaflow import FlowSpec, step, Parameter, resources, conda_base, retry, timeout, catch
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from dataprocessing import load_data, preprocess_data
from xgboost import XGBClassifier

@conda_base(libraries={'mlflow': '2.11.1', 'xgboost': '2.0.3', 'scikit-learn': '1.2.2'}, python='3.9.16')
class TrainModelFlow(FlowSpec):
    seed = Parameter('seed', default=42)

    @resources(cpu=2, memory=4096)
    @timeout(seconds=300)
    @retry(times=2)
    @catch(var='load_error')
    @step
    def start(self):
        self.X, self.y = load_data()
        self.next(self.preprocess)

    @resources(cpu=2, memory=4096)
    @timeout(seconds=300)
    @step
    def preprocess(self):
        self.X_train_scaled_df, self.X_test_scaled_df, self.y_train, self.y_test = preprocess_data(self.X, self.y)
        self.next(self.fetch_best_params)

    @resources(cpu=1, memory=2048)
    @timeout(seconds=300)
    @step
    def fetch_best_params(self):
        mlflow.set_tracking_uri("https://mlflowlab7-905802433874.us-west2.run.app")
        client = MlflowClient()
        best_model = client.get_latest_versions("dry_bean_best_xgb", stages=["None"])[0]
        run = client.get_run(best_model.run_id)
        params = run.data.params
        self.max_depth = int(params["max_depth"])
        self.n_estimators = int(params["n_estimators"])
        self.subsample = float(params["subsample"])
        self.next(self.train)

    @resources(cpu=4, memory=8192)
    @timeout(seconds=600)
    @step
    def train(self):
        self.model = XGBClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            subsample=self.subsample
        )
        self.model.fit(self.X_train_scaled_df, self.y_train)
        self.next(self.register)

    @timeout(seconds=300)
    @step
    def register(self):
        mlflow.set_tracking_uri("https://mlflowlab7-905802433874.us-west2.run.app")
        mlflow.set_experiment("metaflow_training")
        with mlflow.start_run():
            mlflow.log_params({
                'max_depth': self.max_depth,
                'n_estimators': self.n_estimators,
                'subsample': self.subsample
            })
            mlflow.sklearn.log_model(self.model, artifact_path="model", registered_model_name="BestModelFromLastExp")
        self.next(self.end)

    @step
    def end(self):
        print("Training flow complete. Model registered.")

if __name__ == '__main__':
    TrainModelFlow()