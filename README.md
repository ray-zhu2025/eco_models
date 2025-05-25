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
  - 支持区域尺度预报

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
  - 多任务学习框架
  - 支持数据同化

#### GenCast (Google/DeepMind)

- **论文**：
  - https://arxiv.org/abs/2312.15796 (2023.12) - GenCast: Diffusion-based ensemble forecasting for medium-range global weather forecasting
- **特点**：
  - 基于扩散模型
  - 全球天气预报
  - 使用去噪架构
  - 支持不确定性估计

#### Aurora (Google)

- **论文**：
  - https://www.nature.com/articles/s41586-025-09005-y (2025.05) - Aurora: A foundation model for weather and climate
- **代码**：https://github.com/microsoft/aurora
- **特点**：
  - 基于Transformer架构
  - 全球天气预报
  - 支持多变量预测
  - 高分辨率输出
  - 支持0.1°分辨率
  - 10天预报时效
  - 13亿参数规模
  - 支持多种地球系统变量

#### GFS (Global Forecast System)

- **论文**：
  - https://doi.org/10.1175/BAMS-D-15-00043.1 (2015) - The Global Forecast System (GFS)
- **特点**：
  - 美国国家气象局
  - 全球数值天气预报
  - 多尺度预报
  - 集合预报系统

#### ECMWF IFS (Integrated Forecasting System)

- **论文**：
  - https://doi.org/10.1002/qj.828 (2011) - The ECMWF Integrated Forecasting System (IFS)
- **特点**：
  - 欧洲中期天气预报中心
  - 全球数值天气预报
  - 物理过程参数化
  - 数据同化系统

#### GEM (Global Environmental Multiscale)

- **论文**：
  - https://doi.org/10.1175/MWR-D-15-0023.1 (2015) - The Global Environmental Multiscale (GEM) Model
- **代码**：https://github.com/ECCC-ASTD-MRD/gem
- **特点**：
  - 加拿大环境部
  - 全球和区域预报
  - 半拉格朗日方案
  - 多尺度嵌套
  - 数据同化系统
  - 开源代码(LGPL-2.1)
  - 支持多种物理参数化方案
  - 适用于天气和气候研究
  - 支持多种污染物模拟
  - 支持气溶胶和化学过程

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
  - 广泛用于科研和业务

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
  - 使用雷达数据
  - 支持不确定性估计

#### DeepRain

- **论文**：
  - https://arxiv.org/abs/1711.02316 (2017.11) - DeepRain: A Deep Learning Approach for Precipitation Forecasting
- **代码**：https://github.com/thgnaedi/DeepRain
- **特点**：
  - 基于卷积神经网络
  - 降水预报
  - 多尺度特征提取
  - 时空注意力机制

#### DGMR (DeepMind)

- **论文**：
  - https://arxiv.org/abs/2104.00954 (2021.04) - DGMR: A Generative Model for Precipitation Forecasting
- **特点**：
  - 基于GAN
  - 降水预报
  - 生成对抗网络用于不确定性估计

#### PrecipNet (NVIDIA)

- **论文**：
  - https://arxiv.org/abs/2202.11214 (2022.02) - PrecipNet: A Neural Weather Model for Precipitation Forecasting
- **特点**：
  - 基于AFNO架构
  - 降水预报
  - 多尺度特征融合
  - 支持不确定性估计

#### RainNet

- **论文**：
  - https://arxiv.org/abs/1804.10336 (2018.04) - RainNet: A Convolutional Neural Network for Precipitation Nowcasting
- **特点**：
  - 基于卷积神经网络
  - 降水预报
  - 多尺度特征融合
  - 支持不确定性估计

### 2.2 多变量降水预报

#### ConvLSTM

- **论文**：
  - https://arxiv.org/abs/1506.04214 (2015.06) - Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting
- **特点**：
  - 结合CNN和LSTM
  - 时空序列预测
  - 降水预报
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
  - 多尺度特征提取和时空注意力
  - 支持极端降水预报
  - 3小时预报时效
  - 2048km×2048km空间范围
  - 结合物理演化和条件学习
  - 支持对流和辐合过程

#### DeepStorm

- **论文**：
  - https://www.nature.com/articles/s41586-018-0001-6 (2018) - DeepStorm: A Deep Learning Approach for Severe Weather Forecasting
- **代码**：https://github.com/EliasNehme/Deep-STORM
- **特点**：
  - 基于深度学习
  - 强对流预报
  - 时空特征提取
  - 支持不确定性估计
  - 基于CNN架构
  - 支持多尺度特征融合

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
  - 支持波浪解码
  - 支持生物化学耦合
  - 1/4°空间分辨率
  - 15层深度预报
  - 30天预报时效

## 6. 污染物模型

### 6.1 污染物扩散与输运模型

#### 大气污染模型
- **CMAQ (Community Multiscale Air Quality Modeling System)**
  - 基于物理化学原理
  - 模拟污染物在大气中的扩散和转化
  - 支持多种污染物（PM1、PM2.5、PM10、CO、NO、NO2、SO2、O3等）
  - **论文**：
    - https://doi.org/10.1016/j.atmosenv.2010.07.060 (2010) - Description and evaluation of the Community Multiscale Air Quality (CMAQ) modeling system version 4.7
    - https://doi.org/10.1016/j.atmosenv.2019.117116 (2019) - Description and evaluation of the Community Multiscale Air Quality (CMAQ) modeling system version 5.3
  - **文档**：https://www.epa.gov/cmaq
  - **代码**：https://github.com/USEPA/CMAQ

- **高斯扩散模型 (Gaussian Diffusion Model)**
  - 基于高斯分布理论
  - 适用于点源污染物扩散
  - 考虑风速、大气稳定度
  - 计算污染物浓度分布
  - **论文**：
    - https://doi.org/10.1016/j.atmosenv.2010.07.060 (2010) - A review of Gaussian plume models for atmospheric dispersion
  - **特点**：
    - 计算简单快速
    - 适用于稳态条件
    - 考虑地形影响
    - 支持多源叠加
    - 适用于短期预测
    - 可用于应急响应

- **Aurora-AirPollution (Google)**
  - **论文**：
    - https://www.nature.com/articles/s41586-025-09005-y (2025.05) - Aurora: A foundation model for weather and climate
  - **特点**：
    - 基于Transformer架构
    - 空气质量预测
    - 支持多种污染物
    - 包括PM1、PM2.5、PM10
    - 支持CO、NO、NO2、SO2、O3等
    - 支持静态排放源
    - 支持动态排放源

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

### 1. 深度学习框架依赖

#### PyTorch系列模型
- **FourCastNet (NVIDIA)**
  - Python >= 3.6
  - PyTorch
  - CUDA支持

- **Pangu-Weather (华为)**
  - Python >= 3.6
  - PyTorch
  - CUDA支持

- **ClimaX (Microsoft)**
  - Python >= 3.6
  - PyTorch
  - CUDA支持

- **MetNet (Google)**
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

#### TensorFlow系列模型
- **DeepStorm**
  - Python >= 3.5
  - TensorFlow >= 1.4.0
  - Keras >= 1.0.0
  - ImageJ >= 1.51u
  - Matlab >= R2017b

### 2. 科学计算环境依赖

#### 气象模型
- **WRF**
  - Fortran编译器
  - NetCDF库
  - MPI支持
  - 依赖包：
    - HDF5
    - zlib
    - libpng
    - Jasper

#### 环境模型
- **CMAQ**
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

- **InVEST**
  - Python >= 3.9
  - GDAL >= 3.4.2
  - 依赖包：
    - numpy
    - setuptools >= 61
    - wheel
    - setuptools_scm >= 8.0
    - cython >= 3.0.0
    - babel

#### 水文模型
- **SWAT**
  - Fortran编译器
  - 依赖包：
    - netCDF
    - HDF5
    - GDAL

- **MODFLOW6**
  - Python >= 3.10
  - Doxygen
  - Graphviz
  - LaTeX

### 3. 环境管理建议

1. 基础环境配置
   ```bash
   # 创建基础环境
   conda env create -f environment.yml
   
   # 激活环境
   conda activate env_science
   ```

2. 模型特定环境
   - 深度学习模型 (FourCastNet, Pangu-Weather, ClimaX, MetNet)
     - PyTorch + CUDA
     - 需要GPU支持
     - 建议使用conda-forge通道

   - 气象模型 (WRF, GFS)
     - NetCDF + MPI
     - 需要编译环境
     - 建议使用系统包管理器安装依赖

   - 环境模型 (CMAQ, InVEST)
     - GDAL + 地理信息工具
     - 需要系统级依赖
     - 建议使用conda-forge通道

   - 水文模型 (SWAT, MODFLOW6)
     - 需要Fortran编译器
     - 需要系统级依赖
     - 建议使用预编译包

3. 环境管理工具
   - Conda/Mamba：主要环境管理工具
   - pip：Python包管理
   - 系统包管理器：安装系统级依赖

4. 注意事项
   - 深度学习模型需要CUDA支持
   - 气象模型需要MPI环境
   - 环境模型需要GDAL等地理信息工具
   - 水文模型需要Fortran编译器
   - 建议使用conda-forge通道安装科学计算包
   - 部分包可能需要系统级依赖

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
  - 特点：支持多种气候变量预测，区域尺度预报

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

