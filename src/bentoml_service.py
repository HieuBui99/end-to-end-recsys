import numpy as np
import torch
from bentoml.io import JSON, NumpyNdarray

import bentoml

runner = bentoml.pytorch.get("recommendation_model:latest").to_runner()

svc = bentoml.Service("recommendation_service", runners=[runner])


@svc.api(input=NumpyNdarray(dtype=np.int32, enforce_dtype=True), output=JSON())
async def predict(x: np.ndarray) -> dict:
    x = torch.tensor(x).unsqueeze(0)
    score = await runner.async_run(x)
    return {"score": score.item()}
