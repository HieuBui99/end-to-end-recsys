prefect deployment build src/train_model.py:train \
    -n movielens-train \
    -sb remote-file-system/minio-bucket \
    -q movielens --apply