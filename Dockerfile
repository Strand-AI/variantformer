FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    build-essential \
    autotools-dev \
    autoconf \
    wget \
    ca-certificates \
    zlib1g-dev \
    liblzma-dev \
    libbz2-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libncurses5-dev \
    bedtools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/build

# Build htslib
RUN wget -O htslib-1.21.tar.bz2 https://github.com/samtools/htslib/releases/download/1.21/htslib-1.21.tar.bz2 && \
    tar -xjf htslib-1.21.tar.bz2 && \
    cd htslib-1.21 && \
    make && \
    make install && \
    cd .. && \
    rm -rf htslib-1.21 htslib-1.21.tar.bz2

# Build samtools
RUN wget -O samtools-1.21.tar.bz2 https://github.com/samtools/samtools/releases/download/1.21/samtools-1.21.tar.bz2 && \
    tar -xjf samtools-1.21.tar.bz2 && \
    cd samtools-1.21 && \
    make && \
    make install && \
    cd .. && \
    rm -rf samtools-1.21 samtools-1.21.tar.bz2

# Build bcftools
RUN wget -O bcftools-1.21.tar.bz2 https://github.com/samtools/bcftools/releases/download/1.21/bcftools-1.21.tar.bz2 && \
    tar -xjf bcftools-1.21.tar.bz2 && \
    cd bcftools-1.21 && \
    make && \
    make install && \
    cd .. && \
    rm -rf bcftools-1.21 bcftools-1.21.tar.bz2

RUN ldconfig

WORKDIR /app

# Ubuntu doesn't allow pip install --system so put everything into a venv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
RUN uv venv $UV_PROJECT_ENVIRONMENT --python=3.12
ENV VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app:/opt/venv/lib/python3.12/site-packages

COPY pyproject.toml ./
RUN uv pip install -e .[notebook,test]

COPY . .

CMD ["/bin/bash"]
