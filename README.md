# end-to-end-recsys
Personal project to demonstrate ML pipeline. This project uses the Movielens dataset as an example. 

Tech stack:
* Prefect for workflow orchestration
* Google Cloud Storage and BigQuery
* BentoML for serving ML model
* Fastai and Pytorch for training collaborative filtering model

## TODO 
- [x] Data pipeline
- [x] Training pipeline
- [x] Experiment tracking 
- [x] Model registry
- [x] Model serving
- [x] Testing
- [x] CI/CD
- [x] Docker 

## Setup

1. Install dependencies
```python
pip install -r requirements.txt
```

2. Start the prefect server and agent in a docker container. This will also start a MinIO server to store flow code
```
make prefect
```

3. Export wandb API key
```
export WANDB_API_KEY=***
```
or put your key in a `.env` file

4. Build model server. This will download the model weight from the model registry to build and containerize the model server:
```
make bento
```

5. Install pre-commit hook for auto-formatting and linting:
 ```
 pre-commit install
 ```
