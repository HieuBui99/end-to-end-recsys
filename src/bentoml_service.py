import torch
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, JSON

runner = bentoml.pytorch.get("recommendation_model:latest").to_runner()

svc = bentoml.Service("recommendation_service", runners=[runner])

@svc.api(input=NumpyNdarray(dtype=np.int32, enforce_dtype=True), output=JSON())
async def predict(x: np.ndarray) -> dict:
    x = torch.tensor(x).unsqueeze(0)
    score = await runner.async_run(x)
    return {"score": score.item()}
