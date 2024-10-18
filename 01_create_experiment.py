import mlflow
from mlflow_utils import create_mlflow_experiment

if __name__ == "__main__":
    experiment_name = "testing_mlflow1"
    experiment_id = create_mlflow_experiment(
        experiment_name=experiment_name,
        artifact_location="artifacts_folder/testing_mlflow1",
        tags={"env": "dev", "version": "1.0.0"},
    )
    print(experiment_id)
