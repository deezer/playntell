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


# Download and extract curated-deezer and curated-spotify datasets, and a test playlist containing noise
RUN mkdir tmp && mkdir -p /data/playlist-captioning/p/
RUN wget -nc -P tmp/ https://zenodo.org/record/7418837/files/playntell_datasets.zip
RUN unzip -q -d tmp/ tmp/playntell_datasets && cp -r tmp/playntell_datasets/* /data/playlist-captioning/p/

# Download and extract an already trained model (for inference) and discog tag embeddings
COPY playntell_model.tar tmp/
RUN mkdir -p /data/playlist-captioning/p/curated-deezer/algorithm-data/
RUN tar -xf tmp/playntell_model.tar -C /data/playlist-captioning/p/curated-deezer/algorithm-data/
COPY discogs_tags_embeddings.npy /data/playlist-captioning/p/
RUN rm -r tmp/

COPY playntell/ playntell/

CMD poetry run python3 playntell/training_experiments/train_playntell.py