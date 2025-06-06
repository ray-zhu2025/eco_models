# 环境科学模型

## 模型总结

### 1. 全球天气预报模型

| 模型名称 | 开发机构 | 开放类型 | 操作系统 | 编程语言 | 代码链接 |
|---------|---------|----------|----------|----------|----------|
| [FourCastNet](https://arxiv.org/abs/2202.11214) | NVIDIA | 开源 | Linux/Ubuntu 20.04+ | Python, CUDA | [GitHub](https://github.com/NVlabs/FourCastNet) |
| [Pangu-Weather](https://arxiv.org/abs/2211.02556) | 华为 | 开源 | CentOS/Ubuntu 18.04+ | Python, PyTorch | [GitHub](https://github.com/198808xc/Pangu-Weather) |
| [GraphCast](https://arxiv.org/abs/2212.12794) | Google/DeepMind | 开源 | Ubuntu 20.04+ | Python, JAX | [GitHub](https://github.com/google-deepmind/graphcast) |
| [ClimaX](https://arxiv.org/abs/2301.10343) | Microsoft | 开源 | Ubuntu 20.04+ | Python, PyTorch | [GitHub](https://github.com/microsoft/ClimaX) |
| [FengWu](https://arxiv.org/abs/2304.02948) | 上海人工智能实验室 | 开源 | Ubuntu 20.04+ | Python, PyTorch | [GitHub](https://github.com/OpenEarthLab/FengWu) |
| [FuXi](https://arxiv.org/abs/2408.05472) | 复旦大学 | 开源 | Ubuntu 20.04+ | Python, PyTorch | [GitHub](https://github.com/tpys/FuXi) |
| [Aurora](https://www.nature.com/articles/s41586-025-09005-y) | Microsoft | 开源 | Ubuntu 20.04+ | Python, JAX | [GitHub](https://github.com/microsoft/aurora) |

### 2. 降水预报模型

| 模型名称 | 开发机构 | 开放类型 | 操作系统 | 编程语言 | 代码链接 |
|---------|---------|----------|----------|----------|----------|
| [MetNet](https://arxiv.org/abs/2003.12140) | Google | 开源 | Ubuntu 18.04+ | Python, TensorFlow | [GitHub](https://github.com/google-research/metnet) |
| [DeepRain](https://arxiv.org/abs/1711.02316) | - | 开源 | Ubuntu 18.04+ | Python, PyTorch | [GitHub](https://github.com/thgnaedi/DeepRain) |
| [DeepStorm](https://www.nature.com/articles/s41586-018-0001-6) | - | 开源 | Ubuntu 18.04+ | Python, TensorFlow | [GitHub](https://github.com/EliasNehme/Deep-STORM) |

### 3. 环境模型

#### 3.1 大气环境模型
| 模型名称 | 类型 | 开放类型 | 操作系统 | 编程语言 | 代码/资源链接 |
|---------|------|----------|----------|----------|--------------|
| [CMAQ](https://www.epa.gov/cmaq) | 大气污染 | 开源 | CentOS/RHEL 7+, Ubuntu 18.04+ | Fortran | [GitHub](https://github.com/USEPA/CMAQ) |
| [CALPUFF](https://www.epa.gov/scram/air-quality-dispersion-modeling-alternative-models#calpuff) | 大气污染 | 商业 | RHEL/CentOS 7+, Windows | Fortran | [SRC](https://www.src.com/calpuff/) |
| [AERMOD](https://www.epa.gov/scram/air-quality-dispersion-modeling-preferred-and-recommended-models#aermod) | 大气污染 | 商业 | RHEL/CentOS 7+, Windows | Fortran | [EPA](https://www.epa.gov/scram) |
| [WRF-Chem](https://www2.mmm.ucar.edu/wrf/users/) | 大气化学 | 开源 | CentOS/RHEL 7+, Ubuntu 18.04+ | Fortran, C | [官方下载](https://www.acom.ucar.edu/wrf-chem/download.shtml) |
| [CAMx](https://www.camx.com/) | 大气化学 | 商业 | RHEL/CentOS 7+, Windows | Fortran | [官方下载](https://www.camx.com/download/source/) |
| [GEOS-Chem](http://wiki.seas.harvard.edu/geos-chem/index.php/Main_Page) | 大气化学 | 开源 | CentOS/RHEL 7+, Ubuntu 18.04+ | Fortran | [GitHub](https://github.com/geoschem/geos-chem) |
| [EPICC](https://earthlab.iap.ac.cn/) | 大气化学 | 免费申请 | CentOS/RHEL 7+ | Fortran | [官方下载](https://earthlab.iap.ac.cn/resdown/index.html) |

#### 3.2 水环境模型
| 模型名称 | 类型 | 开放类型 | 操作系统 | 编程语言 | 代码/资源链接 |
|---------|------|----------|----------|----------|--------------|
| [Visual MODFLOW Flex](https://www.waterloohydrogeologic.com/visual-modflow-flex/) | 地下水 | 商业 | Windows | C#, Fortran | [官方下载](https://www.waterloohydrogeologic.com/visual-modflow-flex/) |
| [MODFLOW 6](https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model) | 地下水 | 开源 | RHEL/CentOS 7+, Ubuntu 18.04+, Windows, Mac | Fortran | [GitHub](https://github.com/MODFLOW-USGS/modflow6) |
| [GMS](https://www.aquaveo.com/software/gms-groundwater-modeling-system-introduction) | 地下水 | 商业 | Windows | C++, Python | [官方下载](https://www.aquaveo.com/downloads-gms) |
| [TOUGH](https://tough.lbl.gov/) | 多相流 | 商业 | RHEL/CentOS 7+, Windows | Fortran | [官方下载](https://tough.lbl.gov/licensing-download/) |
| [HYDRUS](https://www.pc-progress.com/en/Default.aspx?hydrus-3d) | 土壤水 | 商业 | Windows | Fortran, C++ | [官方下载](https://www.pc-progress.com/en/Default.aspx?downloads) |
| [WASP](https://www.epa.gov/hydrowq/water-quality-analysis-simulation-program-wasp) | 水质模拟 | 开源 | Ubuntu 18.04+, Windows, Mac | Fortran | [官方下载](https://www.epa.gov/hydrowq/wasp8-download) |

#### 3.3 社会经济模型
| 模型名称 | 类型 | 开放类型 | 操作系统 | 编程语言 | 代码/资源链接 |
|---------|------|----------|----------|----------|--------------|
| [LEAP](https://leap.sei.org/) | 综合评估 | 商业 | Windows | C#, .NET | [官方下载](https://leap.sei.org/default.asp?action=download) |
| [GCAM](https://github.com/JGCRI/gcam-core) | 综合评估 | 开源 | Linux/Ubuntu 18.04+, Windows, Mac | C++, R | [GitHub](https://github.com/JGCRI/gcam-core) |


## 技术方案总结

### 1. 模型类型分类

#### 1.1 深度学习模型
- **Transformer架构**：用于全球天气预报，如Pangu-Weather、ClimaX等
- **图神经网络**：用于天气预报和气候预测，如GraphCast
- **卷积神经网络**：用于降水预报，如MetNet、DeepRain
- **傅里叶神经算子**：用于天气预报，如FourCastNet

#### 1.2 数值模拟模型
- **有限差分法**：用于大气和水文模拟，如WRF、MODFLOW
- **欧拉网格法**：用于大气污染模拟，如CMAQ
- **半拉格朗日方案**：用于大气环流模拟，如GEM


### 2. 运行环境

#### 2.1 操作系统支持
- **Linux系统**
  - Ubuntu 18.04+ / 20.04+（推荐用于深度学习模型）
  - CentOS/RHEL 7+（适用于传统数值模型）
- **Windows系统**
  - Windows 10/11（主要用于商业软件和部分开源模型）
- **macOS系统**
  - 部分模型支持

#### 2.2 硬件需求
- **深度学习模型**：
  - GPU：NVIDIA显卡（支持CUDA）
  - 内存：32GB以上推荐
- **数值模拟模型**：
  - CPU：多核处理器
  - 内存：16GB以上
- **综合评估模型**：
  - 标准配置即可

### 3. 开发语言生态

#### 3.1 深度学习开发
- **Python**：主流深度学习框架
  - PyTorch
  - TensorFlow
  - JAX
- **相关库**：
  - NumPy, Pandas（数据处理）
  - xarray（气象数据）

#### 3.2 数值模型开发
- **Fortran**：
  - 数值计算效率高
  - 适用于大规模科学计算
- **C/C++**：
  - 底层性能优化
  - 核心算法实现

#### 3.3 综合模型开发
- **R**：统计分析和数据处理
- **Python**：数据处理和可视化
- **专用语言**：
  - GAMS：经济模型
  - NetCDF：科学数据格式

### 4. 关键依赖和工具

#### 4.1 科学计算
- NetCDF (>= 4.7.0)
- HDF5 (>= 1.10.0)
- BLAS/LAPACK

#### 4.2 并行计算
- MPI：分布式计算
- OpenMP：多线程计算
- CUDA：GPU加速

#### 4.3 地理信息
- GDAL
- PyProj
- Basemap

## 模型开发语言分布

### 1. 按领域划分的主要开发语言

| 领域 | 主要编程语言 | 说明 |
|------|------------|------|
| 大气环境 | Fortran, C | 大多数传统大气模型（WRF-chem、CMAQ、CAMx等）主要使用Fortran开发，部分模块使用C语言 |
| 地表水/海洋 | C#, Fortran, C++ | 时空增益模型主要使用C#，传统水文模型使用Fortran和C++，支持河流湖库与海洋联合模拟 |
| 土壤地下水 | Fortran | MODFLOW、TOUGH、HYDRUS等主要使用Fortran开发 |
| 经济社会 | GAMS, Python, Java | CGE模型主要使用GAMS，系统动力学模型使用Python和Java |
| 生态系统 | Python, R | 新开发的生态模型主要使用Python和R |

### 2. 开发语言特点

| 编程语言 | 主要应用场景 | 优势 | 局限性 |
|---------|------------|------|--------|
| Fortran | 数值计算密集型模型 | 高性能计算，数组运算效率高 | 生态系统不完善，现代特性支持有限 |
| C/C++ | 性能关键模块 | 底层控制，高性能 | 开发效率相对较低 |
| Python | 新型模型开发，数据处理 | 生态系统丰富，开发效率高 | 计算性能相对较低 |
| GAMS | 经济模型 | 优化问题求解能力强 | 专用语言，应用范围受限 |
| Java | 系统动力学 | 跨平台，面向对象 | 数值计算性能一般 |
| C# | 水文模型 | .NET生态，开发效率高 | 跨平台支持相对较弱 |

## 参考资源

- [AI-for-Earth-Paper-List](https://github.com/OpenEarthLab/AI-for-Earth-Paper-List) - OpenEarthLab整理的AI地球科学论文列表
- [Awesome-AI-for-Atmosphere-and-Ocean](https://github.com/XiongWeiTHU/Awesome-AI-for-Atmosphere-and-Ocean) - 清华大学整理的AI大气和海洋科学论文列表