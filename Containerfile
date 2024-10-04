FROM quay.io/rhoai/ray:2.23.0-py39-cu121
USER root
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
USER 1001
