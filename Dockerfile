FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev \
    python3 python3-pip python3-venv \
    curl protobuf-compiler \
    graphviz graphviz-dev git \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup toolchain install 1.83.0

COPY . /root/step_artifact
WORKDIR /root/step_artifact

RUN python3 -m venv venv

RUN source venv/bin/activate && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install maturin numpy sympy networkx pygraphviz pandas protobuf pytest

RUN echo "source /root/step_artifact/venv/bin/activate" >> /root/.bashrc
