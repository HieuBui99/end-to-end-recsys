import wandb

from dotenv import load_dotenv
import bentoml
from fastai.collab import *
from fastai.tabular.all import *

load_dotenv()

api = wandb.Api()

artifact_model = api.artifact('hieubui99/end-to-end-recsys-src/embedding-model:v0', type='model')
print(artifact_model.metadata)
dir = artifact_model.download(root="bentoml")

model = load_learner(Path(dir) / 'model.pkl').model
signatures = {"__call__": {"batchable": True}}

saved_model = bentoml.pytorch.save_model(
    "recommendation_model",
    model,
    signatures=signatures
)
print(f"Model saved: {saved_model}")