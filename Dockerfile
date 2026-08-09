# syntax=docker/dockerfile:1.7

ARG PYTORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755
FROM ${PYTORCH_IMAGE}

ARG APP_UID=1000
ARG APP_GID=1000

ENV HF_HOME=/home/ai-factory/.cache/huggingface \
    MPLCONFIGDIR=/home/ai-factory/.cache/matplotlib \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace

COPY docker/constraints.txt docker/requirements.txt /tmp/ai-factory/

RUN python -m pip install \
        --constraint /tmp/ai-factory/constraints.txt \
        --requirement /tmp/ai-factory/requirements.txt \
    && python -m pip check \
    && rm -rf /tmp/ai-factory

RUN groupadd --gid "${APP_GID}" ai-factory \
    && useradd \
        --create-home \
        --gid "${APP_GID}" \
        --shell /bin/bash \
        --uid "${APP_UID}" \
        ai-factory \
    && mkdir -p \
        /data/allowed/read \
        /data/allowed/write \
        /data/state \
        /home/ai-factory/.cache/huggingface \
        /home/ai-factory/.cache/matplotlib \
        /workspace/src/training_output \
    && chown -R ai-factory:ai-factory /data /home/ai-factory /workspace

COPY --chown=ai-factory:ai-factory . /workspace

USER ai-factory

CMD ["python", "-m", "src.main", "--help"]
