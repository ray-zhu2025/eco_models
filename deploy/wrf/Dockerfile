# 使用Ubuntu 20.04作为基础镜像
FROM ubuntu:20.04

# 设置非交互式安装
ENV DEBIAN_FRONTEND=noninteractive

# 安装基本工具和依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    csh \
    gfortran \
    m4 \
    libcurl4-openssl-dev \
    libnetcdf-dev \
    libnetcdff-dev \
    netcdf-bin \
    nco \
    wget \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /wrf

# 复制WRF源码
COPY models/wrf /wrf/WRF

# 设置环境变量
ENV NETCDF=/usr \
    WRFIO_NCD_LARGE_FILE_SUPPORT=1

# 编译WRF
WORKDIR /wrf/WRF
RUN ./configure << EOF
34
1
EOF

RUN ./compile em_real

# 设置工作目录
WORKDIR /wrf

# 创建数据目录
RUN mkdir -p /wrf/data

# 设置入口点
ENTRYPOINT ["/bin/bash"] 