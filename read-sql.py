import psycopg2

import ray

def create_connection():
    return psycopg2.connect(
        user="postgres",
        password="password",
        host="localhost",
        dbname="vectordb",
    )

# Get a single embedding
dataset = ray.data.read_sql("select * from langchain_pg_embedding limit 1;", create_connection)
dataset.show()
print(dataset.schema())