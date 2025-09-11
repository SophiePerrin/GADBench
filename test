import s3fs
import os

BUCKET = "projet-clustering-ano-graphe"

fs = s3fs.S3FileSystem(
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    token=os.environ.get("AWS_SESSION_TOKEN"),
    client_kwargs={"endpoint_url": os.environ["AWS_S3_ENDPOINT"], "region_name": "us-east-1"}
)

# test de connexion sur ton bucket uniquement
print(fs.ls(BUCKET))