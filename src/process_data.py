from datetime import timedelta
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from fastai.collab import *
from fastai.data.external import URLs
from fastai.tabular.all import *
from fastdownload import FastDownload
from prefect import flow, task
from prefect.tasks import task_input_hash
from prefect_gcp import GcpCredentials

from config import Location


@task(log_prints=True)
def download_data(data_dir: Union[str, Path]):
    path = FastDownload(base=data_dir, archive="downloaded", data="extracted").get(
        URLs.ML_100k
    )

    print(f"Raw data saved to {path}")
    return path


@task(log_prints=True)
def preprocess(raw_path: Union[str, Path], save_path: Union[str, Path]):
    raw_path, save_path = Path(raw_path), Path(save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ratings = pd.read_csv(
        raw_path / "u.data",
        delimiter="\t",
        header=None,
        usecols=(0, 1, 2),
        names=["user", "movie", "rating"],
    )
    print(f"Number of ratings {ratings.shape[0]}")

    movies = pd.read_csv(
        raw_path / "u.item",
        delimiter="|",
        encoding="latin-1",
        usecols=(0, 1),
        names=("movie", "title"),
        header=None,
    )
    print(f"Number of movies {movies.shape[0]}")

    ratings = ratings.merge(movies)
    ratings.to_csv(save_path / "processed_data.csv", index=False)

    return save_path / "processed_data.csv"


@flow(log_prints=True)
def ingest_gbq(path: Union[str, Path]):
    df = pd.read_csv(path)

    credential = GcpCredentials.load("gcp-credential")

    df.to_gbq(
        "movie_lens.ratings",
        "data-eng-383104",
        if_exists="replace",
        credentials=credential.get_credentials_from_service_account(),
    )


@flow(log_prints=True)
def process_data(location: Location = Location()):
    raw_path = download_data(location.raw_dir)
    save_path = preprocess(raw_path, location.processed_dir)
    print(f"Data saved to {save_path}")
    ingest_gbq(save_path)
    print(f"Data logged to Google BigQuery")


if __name__ == "__main__":
    process_data()
