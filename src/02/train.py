import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow
from mlflow import MlflowClient
from pprint import pprint



def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

    
def fetch_logged_data(run_id):
    client = MlflowClient("http://127.0.0.1:5000")
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    #print(f"tracking URI: '{mlflow.get_tracking_uri()}'")
    mlflow.sklearn.autolog()
    
    with mlflow.start_run() as run:
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
        
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        
    # fetch logged data
    params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
    pprint(params)
    pprint(metrics)


if __name__ == '__main__':
    run_train()
