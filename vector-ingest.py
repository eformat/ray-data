#!/usr/bin/env python
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from lxml.html.clean import clean_html
from pathlib import Path
from tqdm import tqdm

import itertools as it
import requests
from typing import Dict
import numpy as np
import pandas as pd
import psycopg2
import torch
import json
import ray
import os

URL_FILE = os.getenv("URL_FILE", "./developers.uri")
COLLECTION_NAME = "documents_test"
CONNECTION_STRING = os.getenv("CONNECTION_STRING", "postgresql+psycopg://postgres:password@localhost:5432/vectordb")
URL_CHUNK_SIZE = 10

class Load:

    def create_connection(self):
        return psycopg2.connect(
            user="postgres",
            password="password",
            host="localhost",
            dbname="vectordb",
        )

    def check_duplicate(self, uri):
        conn = self.create_connection()
        with conn.cursor() as cursor:
            try:
                cursor.execute(
                    "select distinct cmetadata->>'source' as source from langchain_pg_embedding where cmetadata->>'source' = '%s'"
                    % uri
                )
            except psycopg2.errors.UndefinedTable as e:
                return False
            rows = cursor.fetchone()
            if not rows:
                return False
            for row in rows:
                return True

    def __init__(self):
        print(torch.cuda.is_available())
        self.embeddings = HuggingFaceEmbeddings(show_progress=True)

    def __call__(self, batch):
        print(">>> batch:", batch)
        pbar = tqdm(total=len(batch))

        for x in batch['item']:
            pbar.update(URL_CHUNK_SIZE)
            websites_list = list()

            if self.check_duplicate(x):
                print(f">> skipping {x} already exists ...")
            else:
                websites_list.append(x)

            if len(websites_list) == 0:
                continue
            print(f">> processing {websites_list}")

            website_loader = WebBaseLoader(websites_list)
            docs = website_loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=40)
            all_splits = text_splitter.split_documents(docs)
            all_splits[0]

            # Cleanup documents as PostgreSQL won't accept the NUL character, '\x00', in TEXT fields.
            for doc in all_splits:
                doc.page_content = doc.page_content.replace("\x00", "")

            db = PGVector.from_documents(
                documents=all_splits,
                embedding=self.embeddings,
                collection_name=COLLECTION_NAME,
                connection_string=CONNECTION_STRING,
                use_jsonb=True,
            )

        return pd.DataFrame(batch)


text_file = open(URL_FILE, "r")
websites = text_file.read().splitlines()

ds = (
    ray.data.from_items(websites)
    .map_batches(
        Load,
        # workers with one GPU each
        concurrency=1,
        # Batch size is required if you're using GPUs.
        batch_size=URL_CHUNK_SIZE,
        num_gpus=1
    )
)
#ds.show(limit=len(websites))
#ds.show(limit=10)
ds.take_batch(len(websites))
