FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # Install python, gcc (for pycocotools) and other tools
    python3 python3-pip python3-venv gcc build-essential wget unzip \
    # Install audio dependencies
    ffmpeg libsndfile-dev \
    # Install Java
    software-properties-common openjdk-8-jdk-headless && export JAVA_HOME \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Configure Poetry
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install Poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==1.2.2
# Add `poetry` to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"
# Install Python dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install


# Download spacy 'en' model as well as the GPT2 model for training
RUN poetry run python3 -m spacy download en_core_web_sm
RUN wget -nc -P ./playntell/audio_gpt/ https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin


# Download and extract the playntell datasets and the trained model
RUN mkdir -p /data/playlist-captioning/p/curated-deezer/algorithm-data/
RUN wget -nc https://zenodo.org/record/7419490/files/playntell_data_and_model.tar
RUN tar -xf playntell_data_and_model.tar -C .

# Extract the curated-deezer and curated-spotify datasets, and a test playlist containing noise
RUN cp -r playntell_data_and_model/playntell_datasets/* /data/playlist-captioning/p/
# Extract the already trained model and copy the discog tag embeddings
RUN cp -r playntell_data_and_model/playntell /data/playlist-captioning/p/curated-deezer/algorithm-data/
COPY data/discogs_tags_embeddings.npy /data/playlist-captioning/p/
RUN rm -r playntell_data_and_model.tar playntell_data_and_model/

COPY playntell/ playntell/

CMD poetry run python3 playntell/training_experiments/train_playntell.py