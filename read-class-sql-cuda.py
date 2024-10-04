from typing import Dict
import numpy as np
import pandas as pd
import psycopg2
import torch
import ray

class SQLLoad:

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

ds = (
    ray.data.read_sql("select * from langchain_pg_embedding limit 10;", create_connection)
    .map_batches(
        SQLLoad,
        # workers with one GPU each
        concurrency=1,
        # Batch size is required if you're using GPUs.
        batch_size=2,
        num_gpus=1
    )
)
ds.show()
