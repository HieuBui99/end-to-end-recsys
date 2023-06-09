import os

from dotenv import load_dotenv
from fastai.callback.wandb import WandbCallback
from fastai.collab import *
from fastai.tabular.all import *
from prefect import flow, task
from prefect_gcp import GcpCredentials

import wandb
from config import Location

load_dotenv()


@task(log_prints=True)
def get_data():
    credential = GcpCredentials.load("gcp-credential")
    df = pd.read_gbq(
        "SELECT * from movie_lens.ratings",
        credentials=credential.get_credentials_from_service_account(),
        project_id="data-eng-383104",
    )
    print("Pulled data from Google BigQuery")
    print(df.dtypes)
    return df


@task(log_prints=True)
def train_model(df: pd.DataFrame, location: Location = Location()):
    wandb.init()

    print("Starting training")
    df = df.astype({"rating": "float"})
    dls = CollabDataLoaders.from_df(
        df, item_name="title", bs=8, path=location.model_dir
    )
    learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))
    learn.fit_one_cycle(
        2, 5e-3, wd=0.1, cbs=[ShortEpochCallback(), WandbCallback(log_preds=False)]
    )

    print("Finished training")
    return learn


@task(log_prints=True)
def save_model(learn: Learner):
    if not os.path.exists(learn.path):
        os.makedirs(learn.path)
    learn.export("model.pkl")

    # log to wandb
    artifact_model = wandb.Artifact(name="embedding-model", type="model")
    artifact_model.add_file(str(learn.path / "model.pkl"))
    wandb.run.log_artifact(artifact_model)


@flow
def train(location: Location = Location()):
    df = get_data()
    learn = train_model(df)
    save_model(learn)


# if __name__ == "__main__":
#     # deployment = Deployment.build_from_flow(
#     #     flow=train,
#     #     name="train-model",
#     #     infra_overrides={"env": {"PREFECT_LOGGING_LEVEL": "DEBUG"}},
#     #     work_queue_name="movielens",
#     # )
#     # deployment.apply()
