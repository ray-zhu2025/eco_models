# 环境科学模型

## 模型总结

### 1. 全球天气预报模型

| 模型名称 | 开发机构 | 技术方案 | 主要特点 | 代码链接 |
|---------|---------|---------|---------|---------|
| [FourCastNet](https://arxiv.org/abs/2202.11214) | NVIDIA | 傅里叶神经算子 | 全球天气预报，计算效率高 | [GitHub](https://github.com/NVlabs/FourCastNet) |
| [Pangu-Weather](https://arxiv.org/abs/2211.02556) | 华为 | Transformer | 3D Earth-Specific，全球预报 | [GitHub](https://github.com/198808xc/Pangu-Weather) |
| [GraphCast](https://arxiv.org/abs/2212.12794) | Google/DeepMind | 图神经网络 | 消息传递机制，球面数据处理 | [GitHub](https://github.com/google-deepmind/graphcast) |
| [ClimaX](https://arxiv.org/abs/2301.10343) | Microsoft | Transformer | 多任务气候模型，多变量预测 | [GitHub](https://github.com/microsoft/ClimaX) |
| [FengWu](https://arxiv.org/abs/2304.02948) | 上海人工智能实验室 | Transformer | 多尺度特征融合，10天预报 | [GitHub](https://github.com/OpenEarthLab/FengWu) |
| [FuXi](https://arxiv.org/abs/2408.05472) | 复旦大学 | 级联机器学习 | 15天预报，支持数据同化 | [GitHub](https://github.com/tpys/FuXi) |
| [Aurora](https://www.nature.com/articles/s41586-025-09005-y) | Google | Transformer | 全球预报，多系统变量 | [GitHub](https://github.com/microsoft/aurora) |

### 2. 降水预报模型

| 模型名称 | 开发机构 | 技术方案 | 主要特点 | 代码链接 |
|---------|---------|---------|---------|---------|
| [MetNet](https://arxiv.org/abs/2003.12140) | Google | CNN | 0-12小时预报，雷达卫星数据 | [GitHub](https://github.com/google-research/metnet) |
| [DeepRain](https://arxiv.org/abs/1711.02316) | - | CNN | 多尺度特征提取 | [GitHub](https://github.com/thgnaedi/DeepRain) |
| [DeepStorm](https://www.nature.com/articles/s41586-018-0001-6) | - | 深度学习 | 强对流预报，不确定性估计 | [GitHub](https://github.com/EliasNehme/Deep-STORM) |

### 3. 环境模型

| 模型名称 | 类型 | 技术方案 | 主要特点 | 代码链接 |
|---------|------|---------|---------|---------|
| [CMAQ](https://www.epa.gov/cmaq) | 大气污染 | 欧拉网格法 | 多污染物模拟，化学过程 | [GitHub](https://github.com/USEPA/CMAQ) |
| [CALPUFF](https://www.epa.gov/scram/air-quality-dispersion-modeling-alternative-models#calpuff) | 大气污染 | 拉格朗日烟团 | 复杂地形，长距离传输 | [SRC](https://www.src.com/calpuff/) |
| [AERMOD](https://www.epa.gov/scram/air-quality-dispersion-modeling-preferred-and-recommended-models#aermod) | 大气污染 | 高斯扩散 | 近场扩散，建筑物效应 | [EPA](https://www.epa.gov/scram) |
| [SWAT](https://swat.tamu.edu/) | 水文 | 分布式水文 | 流域模拟，农业影响 | [GitHub](https://github.com/WatershedModels/SWAT) |
| [MODFLOW6](https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model) | 地下水 | 有限差分法 | 三维地下水模拟 | [GitHub](https://github.com/MODFLOW-USGS/modflow6) |
| [InVEST](https://naturalcapitalproject.stanford.edu/software/invest) | 生态系统 | GIS分析 | 生态系统服务评估 | [GitHub](https://github.com/natcap/invest) |

### 4. 技术方案分类

| 技术方案 | 适用场景 | 代表模型 | 主要特点 |
|---------|---------|---------|---------|
| Transformer | 序列数据，时空预测 | Pangu-Weather, ClimaX | 自注意力机制，并行计算 |
| 图神经网络 | 非规则网格数据 | GraphCast | 消息传递，球面数据处理 |
| CNN | 规则网格数据 | MetNet, DeepRain | 局部特征提取，多尺度融合 |
| 傅里叶神经算子 | 物理场预测 | FourCastNet | 频域处理，计算效率高 |
| 有限差分法 | 偏微分方程求解 | WRF, MODFLOW | 离散化，显式/隐式格式 |
| 欧拉网格法 | 污染物扩散 | CMAQ | 固定网格，物质守恒 |

## 技术方案总结

### 1. 深度学习方案
- Transformer：序列数据，时空预测（Pangu-Weather, ClimaX）
- 图神经网络：非规则网格数据（GraphCast）
- CNN：规则网格数据（MetNet, DeepRain）
- 傅里叶神经算子：物理场预测（FourCastNet）

### 2. 传统数值方案
- 有限差分法：偏微分方程求解（WRF, MODFLOW）
- 谱方法：全球尺度模拟（GFS）
- 欧拉网格法：污染物扩散（CMAQ）
- 半拉格朗日方案：平流问题（GEM）

### 3. 运行环境
- 操作系统：
  - Linux (推荐 Ubuntu 20.04+)
  - Windows 10/11 (部分模型支持)
  - macOS (部分模型支持)
- 编译环境：
  - Fortran (gfortran >= 9.0)
  - C/C++ (gcc/g++ >= 9.0)
  - CMake (>= 3.10)
- 深度学习框架：PyTorch, TensorFlow, JAX
- 科学计算：NumPy, SciPy, Pandas, Xarray
- 地理信息：GDAL, PyProj, Basemap
- 并行计算：MPI, OpenMP, CUDA
- 数值计算：
  - NetCDF (>= 4.7.0)
  - HDF5 (>= 1.10.0)
  - OpenMPI (>= 4.0.0)
  - BLAS/LAPACK

## 参考资源

- [AI-for-Earth-Paper-List](https://github.com/OpenEarthLab/AI-for-Earth-Paper-List) - OpenEarthLab整理的AI地球科学论文列表
- [Awesome-AI-for-Atmosphere-and-Ocean](https://github.com/XiongWeiTHU/Awesome-AI-for-Atmosphere-and-Ocean) - 清华大学整理的AI大气和海洋科学论文列表
