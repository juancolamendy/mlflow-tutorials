import mlflow

from mlflow_utils import create_mlflow_experiment, get_mlflow_experiment

if __name__=="__main__":
    experiment_name = "testing_mlflow1"
    # experiment_id = create_mlflow_experiment(
    #     experiment_name=experiment_name,
    #     artifact_location="testing_mlflow1_artifacts",
    #     tags={"env": "dev", "version": "1.0.0"},
    # )
    # print("Experiment ID: {}".format(experiment_id))
    experiment = get_mlflow_experiment(experiment_name=experiment_name)
    if experiment is None:
        print("Experiment not found")
        exit(1)
    #mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name="testing", experiment_id = experiment.experiment_id) as run:
        # Your machine learning code goes here
        mlflow.log_param("learning_rate",0.01)
        # print run info    
        print("run_id: {}".format(run.info.run_id))
        print("experiment_id: {}".format(run.info.experiment_id))
        print("status: {}".format(run.info.status))
        print("start_time: {}".format(run.info.start_time))
        print("end_time: {}".format(run.info.end_time))
        print("lifecycle_stage: {}".format(run.info.lifecycle_stage))