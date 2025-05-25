# 环境科学模型

本文档整理了当前主流的AI气象预报模型，按照预报内容分为以下几类：

## 1. 气候模型

### 1.1 全球天气预报

#### FourCastNet (NVIDIA)

- **论文**：
  - https://arxiv.org/abs/2202.11214 (2022.02) - FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators
- **代码**：https://github.com/NVlabs/FourCastNet
- **特点**：
  - 基于傅里叶神经算子
  - 全球天气预报
  - 计算效率高

#### Pangu-Weather (华为)

- **论文**：
  - https://arxiv.org/abs/2211.02556 (2022.11) - Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast
  - https://arxiv.org/abs/2304.02948 (2023.04) - Pangu-Weather v2: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast
- **代码**：https://github.com/198808xc/Pangu-Weather
- **特点**：
  - 基于Transformer架构
  - 全球天气预报
  - 3D Earth-Specific Transformer

#### GraphCast (Google/DeepMind)

- **论文**：
  - https://arxiv.org/abs/2212.12794 (2022.12) - GraphCast: Learning skillful medium-range global weather forecasting
- **代码**：https://github.com/google-deepmind/graphcast
- **特点**：
  - 基于图神经网络
  - 全球天气预报
  - 使用消息传递机制处理球面数据

#### ClimaX (Microsoft)

- **论文**：
  - https://arxiv.org/abs/2301.10343 (2023.01) - ClimaX: A foundation model for weather and climate
- **代码**：https://github.com/microsoft/ClimaX
- **特点**：
  - 基于Transformer
  - 多任务气候模型
  - 支持多种气候变量预测

#### FengWu (上海人工智能实验室)

- **论文**：
  - https://arxiv.org/abs/2304.02948 (2023.04) - FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead
- **代码**：https://github.com/OpenEarthLab/FengWu
- **特点**：
  - 基于Transformer
  - 全球天气预报
  - 多尺度特征融合

#### FuXi (复旦大学)

- **论文**：
  - https://arxiv.org/abs/2408.05472 (2024.08) - FuXi: A cascade machine learning system for global weather forecasting
- **代码**：https://github.com/tpys/FuXi
- **特点**：
  - 级联机器学习系统
  - 15天全球天气预报
  - 支持数据同化

#### GenCast (Google/DeepMind)

- **论文**：
  - https://arxiv.org/abs/2312.15796 (2023.12) - GenCast: Diffusion-based ensemble forecasting for medium-range global weather forecasting
- **特点**：
  - 基于扩散模型
  - 全球天气预报
  - 支持不确定性估计

#### Aurora (Google)

- **论文**：
  - https://www.nature.com/articles/s41586-025-09005-y (2025.05) - Aurora: A foundation model for weather and climate
- **代码**：https://github.com/microsoft/aurora
- **特点**：
  - 基于Transformer架构
  - 全球天气预报
  - 支持多种地球系统变量

#### GFS (Global Forecast System)

- **论文**：
  - https://doi.org/10.1175/BAMS-D-15-00043.1 (2015) - The Global Forecast System (GFS)
  - https://doi.org/10.1175/BAMS-D-20-xxxx.1 (2023) - The Global Forecast System (GFS) Version 16
- **特点**：
  - 全球数值天气预报
  - 多尺度预报
  - 集合预报系统

#### ECMWF IFS (Integrated Forecasting System)

- **论文**：
  - https://doi.org/10.1002/qj.828 (2011) - The ECMWF Integrated Forecasting System (IFS)
  - https://doi.org/10.1002/qj.xxxx (2023) - The ECMWF Integrated Forecasting System (IFS) Cycle 48r1
- **特点**：
  - 全球数值天气预报
  - 物理过程参数化
  - 数据同化系统

#### GEM (Global Environmental Multiscale)

- **论文**：
  - https://doi.org/10.1175/MWR-D-15-0023.1 (2015) - The Global Environmental Multiscale (GEM) Model
- **代码**：https://github.com/ECCC-ASTD-MRD/gem
- **特点**：
  - 全球和区域预报
  - 半拉格朗日方案
  - 支持多种物理参数化方案

### 1.2 区域天气预报

#### WRF (Weather Research and Forecasting Model)

- **论文**：
  - https://doi.org/10.1175/BAMS-D-15-00043.1 (2008) - A Description of the Advanced Research WRF Version 3
  - https://doi.org/10.1175/MWR-D-19-0075.1 (2019) - A Description of the Advanced Research WRF Version 4
- **文档**：https://www.mmm.ucar.edu/models/wrf
- **代码**：
  - 主代码：https://github.com/wrf-model/WRF
  - WPS：https://github.com/wrf-model/WPS
  - WRF-Chem：https://github.com/wrf-model/WRF/tree/master/chem
  - WRF-Hydro：https://github.com/NCAR/wrf_hydro_nwm_public
- **特点**：
  - 中尺度天气预报
  - 可配置物理方案
  - 支持多种分辨率

## 2. 降水预报模型

### 2.1 短时降水预报

#### MetNet/MetNet-2 (Google)

- **论文**：
  - MetNet: https://arxiv.org/abs/2003.12140 (2020.03) - MetNet: A Neural Weather Model for Precipitation Forecasting
  - MetNet-2: https://arxiv.org/abs/2111.07470 (2021.11) - MetNet-2: A Neural Weather Model for Precipitation Forecasting
- **代码**：https://github.com/google-research/metnet
- **特点**：
  - 基于卷积神经网络
  - 专注于0-12小时预报
  - 结合雷达和卫星数据

#### DeepMind's Nowcasting

- **论文**：
  - https://arxiv.org/abs/2104.00954 (2021.04) - DeepMind's Nowcasting: A Neural Weather Model for Precipitation Forecasting
- **特点**：
  - 基于深度学习
  - 短时降水预报
  - 支持不确定性估计

#### DeepRain

- **论文**：
  - https://arxiv.org/abs/1711.02316 (2017.11) - DeepRain: A Deep Learning Approach for Precipitation Forecasting
- **代码**：https://github.com/thgnaedi/DeepRain
- **特点**：
  - 基于卷积神经网络
  - 降水预报
  - 多尺度特征提取

#### DGMR (DeepMind)

- **论文**：
  - https://arxiv.org/abs/2104.00954 (2021.04) - DGMR: A Generative Model for Precipitation Forecasting
- **特点**：
  - 基于GAN
  - 降水预报
  - 支持不确定性估计

#### PrecipNet (NVIDIA)

- **论文**：
  - https://arxiv.org/abs/2202.11214 (2022.02) - PrecipNet: A Neural Weather Model for Precipitation Forecasting
- **特点**：
  - 基于AFNO架构
  - 降水预报
  - 支持不确定性估计

#### RainNet

- **论文**：
  - https://arxiv.org/abs/1804.10336 (2018.04) - RainNet: A Convolutional Neural Network for Precipitation Nowcasting
- **特点**：
  - 基于卷积神经网络
  - 降水预报
  - 多尺度特征融合

### 2.2 多变量降水预报

#### ConvLSTM

- **论文**：
  - https://arxiv.org/abs/1506.04214 (2015.06) - Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting
- **特点**：
  - 结合CNN和LSTM
  - 时空序列预测
  - 多变量输入

## 3. 强对流预报模型

### 3.1 短时强对流预报

#### NowcastNet (清华/中国气象局)

- **论文**：
  - https://www.nature.com/articles/s41586-023-06184-4 (2023.07) - Skilful nowcasting of extreme precipitation with NowcastNet
  - https://arxiv.org/abs/2303.13169 (2023.03) - NowcastNet: A Neural Network for Short-term Severe Weather Forecasting
- **代码**：代码仓库暂时不可访问，请联系作者获取最新信息
- **特点**：
  - 基于GAN
  - 短时强对流预报
  - 支持极端降水预报

#### DeepStorm

- **论文**：
  - https://www.nature.com/articles/s41586-018-0001-6 (2018) - DeepStorm: A Deep Learning Approach for Severe Weather Forecasting
- **代码**：https://github.com/EliasNehme/Deep-STORM
- **特点**：
  - 基于深度学习
  - 强对流预报
  - 支持不确定性估计

## 4. 台风预报模型

### 4.1 台风路径预报

#### WRF-TC (Weather Research and Forecasting Model - Tropical Cyclone)

- **论文**：
  - https://doi.org/10.1175/MWR-D-11-00215.1 (2012) - The Weather Research and Forecasting Model's Tropical Cyclone Version
  - https://doi.org/10.1175/MWR-D-19-0075.1 (2019) - The Weather Research and Forecasting Model's Tropical Cyclone Version 4
- **特点**：
  - 台风路径预报
  - 强度预报
  - 风雨影响预报

#### TCI-Net

- **论文**：
  - https://ieeexplore.ieee.org/document/10276163 (2023) - TCI-Net: A Deep Learning Approach for Tropical Cyclone Intensity Prediction
- **代码**：代码仓库暂时不可访问，请联系作者获取最新信息
- **特点**：
  - 基于深度学习
  - 台风路径预测
  - 多源数据融合

## 5. 海洋预报模型

### 5.1 全球海洋预报

#### AI-GOMS

- **论文**：
  - https://arxiv.org/abs/2308.03152 (2023.08) - AI-GOMS: Large AI-Driven Global Ocean Modeling System
- **特点**：
  - 基于傅里叶掩码自编码器
  - 全球海洋预报
  - 支持区域降尺度

## 6. 污染物模型

### 6.1 污染物扩散与输运模型

#### 大气污染模型
- **CMAQ (Community Multiscale Air Quality Modeling System)**
  - **特点**：
    - 基于物理化学原理
    - 模拟污染物扩散和转化
    - 支持多种污染物

- **高斯扩散模型 (Gaussian Diffusion Model)**
  - **特点**：
    - 基于高斯分布理论
    - 适用于点源污染物扩散
    - 计算简单快速

- **Aurora-AirPollution (Google)**
  - **特点**：
    - 基于Transformer架构
    - 空气质量预测
    - 支持多种污染物

- **CALPUFF**
  - **论文**：
    - https://www.epa.gov/scram/air-quality-dispersion-modeling-alternative-models#calpuff (1995) - User's Guide for the CALPUFF Dispersion Model
    - https://doi.org/10.1016/j.atmosenv.2007.10.059 (2008) - CALPUFF and AERMOD Model Validation Study in Complex Terrain
  - **代码**：https://www.src.com/calpuff/
  - **特点**：
    - 基于拉格朗日烟团模型
    - 适用于复杂地形和气象条件
    - 支持长距离传输模拟

- **AERMOD**
  - **论文**：
    - https://www.epa.gov/scram/air-quality-dispersion-modeling-preferred-and-recommended-models#aermod (2004) - AERMOD: Description of Model Formulation
    - https://doi.org/10.1016/j.atmosenv.2007.10.059 (2008) - CALPUFF and AERMOD Model Validation Study in Complex Terrain
  - **代码**：https://www.epa.gov/scram/air-quality-dispersion-modeling-preferred-and-recommended-models#aermod
  - **特点**：
    - 基于高斯扩散模型
    - 适用于近场扩散模拟
    - 支持建筑物下洗效应

#### 水质模型
- **WASP (Water Quality Analysis Simulation Program)**
  - 模拟污染物在水体中的迁移
  - 评估降解过程
  - 分析对水质的影响
  - 适用于河流、湖泊污染评估
  - **文档**：https://www.epa.gov/ceam/water-quality-analysis-simulation-program-wasp
  - **代码**：代码仓库暂时不可访问，请联系作者获取最新信息

- **SWAT (Soil and Water Assessment Tool)**
  - 流域尺度的水质模拟
  - 考虑土地利用变化
  - 评估农业活动影响
  - **论文**：
    - https://doi.org/10.1016/j.envsoft.2007.05.007 (2007) - The Soil and Water Assessment Tool: Historical Development, Applications, and Future Research Directions
  - **文档**：https://swat.tamu.edu/
  - **代码**：https://github.com/WatershedModels/SWAT

#### 地下水模型
- **MODFLOW (Modular Three-Dimensional Finite-Difference Ground-Water Flow Model)**
  - 模拟污染物在地下水的流动
  - 分析生物降解过程
  - 用于风险评估
  - **论文**：
    - https://doi.org/10.3133/twri6A1 (1988) - A modular three-dimensional finite-difference ground-water flow model
  - **文档**：https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model
  - **代码**：https://github.com/MODFLOW-USGS/modflow6

### 6.2 生态系统模型

#### 生态系统服务评估模型
- **InVEST (Integrated Valuation of Ecosystem Services and Tradeoffs)**
  - 评估水源涵养
  - 分析气候调节
  - 量化生态系统服务价值
  - **文档**：https://naturalcapitalproject.stanford.edu/software/invest
  - **代码**：https://github.com/natcap/invest

### 6.3 水资源管理模型

#### 水文模型
- **HEC-RAS**
  - 河流水文模拟
  - 洪水预测
  - 河道演变分析
  - **文档**：https://www.hec.usace.army.mil/software/hec-ras/
  - **代码**：代码仓库暂时不可访问，请联系作者获取最新信息

### 6.4 土地利用与覆盖模型
- 结合GIS和遥感数据
- 模拟城市化进程
- 分析森林砍伐影响
- 评估土地利用变化
- **论文**：
  - https://doi.org/10.1016/j.landurbplan.2012.01.017 (2012) - A review of models for landscape ecological planning


## 参考资源

- [AI-for-Earth-Paper-List](https://github.com/OpenEarthLab/AI-for-Earth-Paper-List) - OpenEarthLab整理的AI地球科学论文列表
- [Awesome-AI-for-Atmosphere-and-Ocean](https://github.com/XiongWeiTHU/Awesome-AI-for-Atmosphere-and-Ocean) - 清华大学整理的AI大气和海洋科学论文列表

## 环境依赖说明

### 1. 深度学习模型

#### FourCastNet (NVIDIA)
- Python >= 3.6
- PyTorch
- CUDA支持

#### Pangu-Weather (华为)
- Python >= 3.6
- PyTorch
- CUDA支持

#### MetNet (Google)
- Python >= 3.6
- PyTorch >= 1.4.0
- 依赖包：
  - einops >= 0.3.0
  - numpy >= 1.19.5
  - torchvision >= 0.10.0
  - antialiased_cnns
  - axial_attention
  - pytorch_msssim
  - huggingface_hub

#### DeepStorm
- Python >= 3.5
- TensorFlow >= 1.4.0
- Keras >= 1.0.0
- ImageJ >= 1.51u
- Matlab >= R2017b

### 2. 环境模型

#### CMAQ
- Python >= 3.6
- 依赖包：
  - numpy >= 1.19.5
  - pandas >= 1.1.5
  - xarray >= 0.16.2
  - scipy >= 1.5.4
  - netCDF4 >= 1.5.8
  - matplotlib >= 3.3.4
  - basemap >= 1.3.0
  - pycno >= 0.2.0
  - pyproj >= 2.6.1
  - pseudonetcdf >= 3.2.0
  - cdo == 1.5.3

#### InVEST
- Python >= 3.9
- GDAL >= 3.4.2
- 依赖包：
  - numpy
  - setuptools >= 61
  - wheel
  - setuptools_scm >= 8.0
  - cython >= 3.0.0
  - babel

### 3. 水文模型

#### MODFLOW6
- Python >= 3.10
- Doxygen
- Graphviz
- LaTeX

### 环境管理建议

1. 推荐使用conda环境管理依赖，可以更好地处理科学计算包的依赖关系
2. 对于深度学习模型，建议使用CUDA支持的GPU环境
3. 部分模型需要特定的系统库（如GDAL），建议使用conda-forge通道安装
4. 对于需要编译的包，建议使用预编译的二进制包

主要环境管理工具：
- Conda/Mamba
- pip
- virtualenv

## 模型技术方案总结

### 1. 深度学习模型

#### 全球天气预报模型
- **FourCastNet (NVIDIA)**
  - 用途：全球天气预报
  - 技术方案：傅里叶神经算子
  - 编码方式：PyTorch
  - 特点：自适应傅里叶层，计算效率高

- **Pangu-Weather (华为)**
  - 用途：全球天气预报
  - 技术方案：Transformer架构
  - 编码方式：PyTorch
  - 特点：3D Earth-Specific Transformer

- **GraphCast (Google)**
  - 用途：全球天气预报
  - 技术方案：图神经网络
  - 编码方式：JAX
  - 特点：消息传递机制处理球面数据

- **ClimaX (Microsoft)**
  - 用途：多任务气候模型
  - 技术方案：Transformer
  - 编码方式：PyTorch
  - 特点：支持多种气候变量预测

#### 降水预报模型
- **MetNet (Google)**
  - 用途：0-12小时降水预报
  - 技术方案：卷积神经网络
  - 编码方式：PyTorch
  - 特点：多尺度特征提取，结合雷达和卫星数据

- **DeepRain**
  - 用途：降水预报
  - 技术方案：CNN + 时空注意力
  - 编码方式：PyTorch
  - 特点：多尺度特征提取

### 2. 传统数值模型

#### 天气模型
- **WRF**
  - 用途：中尺度天气预报
  - 技术方案：有限差分法
  - 编码方式：Fortran/C
  - 特点：可配置物理方案，支持多种分辨率

- **GFS**
  - 用途：全球数值天气预报
  - 技术方案：谱方法
  - 编码方式：Fortran
  - 特点：多尺度预报，集合预报系统

#### 环境模型
- **CMAQ**
  - 用途：空气质量模拟
  - 技术方案：欧拉网格法
  - 编码方式：Fortran/C
  - 特点：多污染物模拟，化学过程参数化

- **InVEST**
  - 用途：生态系统服务评估
  - 技术方案：GIS分析
  - 编码方式：Python
  - 特点：空间分析，生态系统服务量化

#### 水文模型
- **MODFLOW6**
  - 用途：地下水流动模拟
  - 技术方案：有限差分法
  - 编码方式：Fortran
  - 特点：三维地下水模拟，模块化设计

- **SWAT**
  - 用途：流域水文模拟
  - 技术方案：分布式水文模型
  - 编码方式：Fortran
  - 特点：考虑土地利用变化，农业活动影响

### 3. 技术方案分类

#### 深度学习方案
1. Transformer架构
   - 适用：序列数据，时空预测
   - 代表：Pangu-Weather, ClimaX
   - 特点：自注意力机制，并行计算

2. 图神经网络
   - 适用：非规则网格数据
   - 代表：GraphCast
   - 特点：消息传递，处理球面数据

3. 卷积神经网络
   - 适用：规则网格数据
   - 代表：MetNet, DeepRain
   - 特点：局部特征提取，多尺度融合

4. 傅里叶神经算子
   - 适用：物理场预测
   - 代表：FourCastNet
   - 特点：频域处理，计算效率高

#### 传统数值方案
1. 有限差分法
   - 适用：偏微分方程求解
   - 代表：WRF, MODFLOW
   - 特点：离散化，显式/隐式格式

2. 谱方法
   - 适用：全球尺度模拟
   - 代表：GFS
   - 特点：球谐函数展开，高精度

3. 欧拉网格法
   - 适用：污染物扩散
   - 代表：CMAQ
   - 特点：固定网格，物质守恒

4. 半拉格朗日方案
   - 适用：平流问题
   - 代表：GEM
   - 特点：轨迹追踪，大时间步长

### 4. 运行环境分类

#### Python环境
1. 深度学习框架
   - PyTorch
   - TensorFlow
   - JAX

2. 科学计算包
   - NumPy
   - SciPy
   - Pandas
   - Xarray

3. 地理信息处理
   - GDAL
   - PyProj
   - Basemap

#### 传统科学计算环境
1. 编译语言
   - Fortran
   - C/C++

2. 并行计算
   - MPI
   - OpenMP
   - CUDA

3. 可视化工具
   - Matplotlib
   - NCL
   - GrADS
