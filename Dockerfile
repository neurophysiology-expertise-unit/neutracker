FROM mambaorg/micromamba:latest

COPY environment.yml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
USER root
COPY . /app
WORKDIR /app
RUN pip install .

USER $MAMBA_USER
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "neutracker-cli"]
