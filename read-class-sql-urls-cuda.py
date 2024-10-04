from typing import Dict
import numpy as np
import pandas as pd
import psycopg2
import torch
import ray
import os

class Load:

    def __init__(self):
        print(torch.cuda.is_available())

    def __call__(self, batch):
        return pd.DataFrame(batch)

def create_connection():
    return psycopg2.connect(
        user="postgres",
        password="password",
        host="localhost",
        dbname="vectordb",
    )

URL_FILE = os.getenv("URL_FILE", "./developers.uri")
text_file = open(URL_FILE, "r")
websites = text_file.read().splitlines()

ds = (
    ray.data.from_items(websites)
    .map_batches(
        Load,
        # workers with one GPU each
        concurrency=2,
        # Batch size is required if you're using GPUs.
        batch_size=50,
        num_gpus=1
    )
)
ds.show(limit=len(websites))
