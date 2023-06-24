import wandb

from dotenv import load_dotenv
from fastai.collab import *
from fastai.tabular.all import *
from fastai.callback.wandb import WandbCallback
from prefect import flow, task
from prefect_gcp import GcpCredentials
from prefect.deployments import Deployment

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
def train_model(df: pd.DataFrame):
    wandb.init()

    print("Starting training")
    df = df.loc[:, ["user", "movie", "rating"]]
    df = df.astype({"rating": "float"})
    dls = CollabDataLoaders.from_df(df, bs=8)
    learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))
    learn.fit_one_cycle(2, 5e-3, wd=0.1, cbs=[WandbCallback(log_preds=False, log_model=True)])

    print("Finished training")
    return learn


@task(log_prints=True)
def save_model(learn: Learner, save_dir: Union[str, Path]):
    # learn.path = Path(save_dir)
    learn.export("model.pkl")


@flow
def train(location: Location = Location()):
    df = get_data()
    learn = train_model(df)
    save_model(learn, location.model_dir)


if __name__ == "__main__":
    train()
    # deployment = Deployment.build_from_flow(
    #     flow=train,
    #     name="train-model",
    #     infra_overrides={"env": {"PREFECT_LOGGING_LEVEL": "DEBUG"}},
    #     work_queue_name="movielens",
    # )
    # deployment.apply()
    