FROM ubuntu:20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV WRF_VERSION=4.4
ENV WPS_VERSION=4.4

# 安装基本依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    csh \
    gfortran \
    m4 \
    perl \
    python3 \
    python3-pip \
    wget \
    git \
    libnetcdf-dev \
    libnetcdff-dev \
    libhdf5-serial-dev \
    libpng-dev \
    libjpeg-dev \
    libjasper-dev \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /wrf

# 下载并编译WRF
RUN wget https://github.com/wrf-model/WRF/archive/v${WRF_VERSION}.tar.gz \
    && tar xzf v${WRF_VERSION}.tar.gz \
    && cd WRF-${WRF_VERSION} \
    && ./configure \
    && ./compile em_real \
    && cd ..

# 下载并编译WPS
RUN wget https://github.com/wrf-model/WPS/archive/v${WPS_VERSION}.tar.gz \
    && tar xzf v${WPS_VERSION}.tar.gz \
    && cd WPS-${WPS_VERSION} \
    && ./configure \
    && ./compile \
    && cd ..

# 设置环境变量
ENV PATH="/wrf/WRF-${WRF_VERSION}/main:${PATH}"
ENV PATH="/wrf/WPS-${WPS_VERSION}:${PATH}"

# 安装Python依赖
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# 暴露端口
EXPOSE 8000

# 设置工作目录
WORKDIR /app

# 复制应用代码
COPY . .

# 启动命令
CMD ["python3", "app.py"] 