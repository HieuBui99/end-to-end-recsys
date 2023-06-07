from fastai.collab import *
from fastai.tabular.all import *
from prefect import flow, task
from prefect_gcp import GcpCredentials

from config import Location


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
    print("Starting training")
    dls = CollabDataLoaders.from_df(df.loc[:, ["user", "movie", "rating"]], bs=8)
    print(dls.show_batch())
    learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))
    learn.fit_one_cycle(5, 5e-3, wd=0.1, cbs=[ShortEpochCallback()])

    print("Finished training")
    return learn


@task(log_prints=True)
def save_model(learn: Learner, save_dir: Union[str, Path]):
    learn.path = Path(save_dir)
    learn.export("model.pkl")


@flow
def train(location: Location = Location()):
    df = get_data()
    learn = train_model(df)
    save_model(learn, location.model_dir)


if __name__ == "__main__":
    train()