FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev \
    python3 python3-pip python3-venv \
    curl protobuf-compiler \
    graphviz graphviz-dev git \
    tmux \
    build-essential \
    cmake \
    g++-12 \
    tcl tk \
    texlive-latex-base \
    texlive-pictures \
    latexmk \
    libyaml-cpp-dev libfmt-dev libspdlog-dev \
    texlive-luatex texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

# Set g++-12 as the default g++ compiler
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup toolchain install 1.83.0

COPY step_artifact  /root/step_artifact
COPY step-artifact-hdl /root/step-artifact-hdl
WORKDIR /root/step_artifact

RUN python3 -m venv venv

RUN source venv/bin/activate && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install maturin numpy sympy networkx pygraphviz pandas protobuf pytest matplotlib seaborn

# Install Bluespec Compiler (BSC)
ARG BSC_VERSION=2025.07
RUN curl -L -o /tmp/bsc.tar.gz \
    https://github.com/B-Lang-org/bsc/releases/download/${BSC_VERSION}/bsc-${BSC_VERSION}-ubuntu-24.04.tar.gz && \
    mkdir -p /opt/bluespec && \
    tar -xzf /tmp/bsc.tar.gz -C /opt/bluespec --strip-components=1 && \
    rm /tmp/bsc.tar.gz

# Add Bluespec to PATH
ENV PATH="/opt/bluespec/bin:${PATH}"

RUN echo "source /root/step_artifact/venv/bin/activate" >> /root/.bashrc
RUN echo 'export PYTHONPATH=/root/step_artifact/src:/root/step_artifact/src/step_py:/root/step_artifact/src/sim:/root/step_artifact/src/proto:$PYTHONPATH' >> /root/.bashrc
