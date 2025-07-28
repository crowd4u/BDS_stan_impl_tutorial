# Base image
FROM python:3.10

# Install required packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CmdStanPy and dependencies
RUN pip install matplotlib==3.7.1 numpy==1.24.3 scipy==1.10.1 \
                sympy==1.11.1 pandas==2.0.1 arviz==0.20.0 \
                jupyterlab==4.3.4 ipykernel==6.22.0 nest-asyncio==1.6.0 \
                ipywidgets==8.1.7

# Paun et al. (2018) のStan実装コードを動かすために古いバージョンを指定しているので注意
RUN pip3 install cmdstanpy[all]==1.2.5
RUN python3 -c 'import cmdstanpy; cmdstanpy.install_cmdstan(version="2.32.2")'

# パッケージを追加したい場合はこの下に書いていってください
RUN pip install crowd-kit==1.3.0.post0 
RUN pip install polars==1.19.0

# Set working directory
WORKDIR /app

# Copy the contents of the host directory to the container
COPY . /app

