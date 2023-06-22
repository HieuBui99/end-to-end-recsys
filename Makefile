SHELL :=/bin/bash

prefect-server:
	docker compose --profile server --profile minio up

bento:
	python src/bentoml_save.py && 
	bentoml build -f ./bentoml/bentofile.yaml ./src/ &&
	bentoml containerize recommendation_service -t recommendation_service:latest

bento-up:
	docker compose up bento_service