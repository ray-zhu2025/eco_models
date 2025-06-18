# 环境科学模型

## 1. 气象模型

### 1.1 基本信息

| 模型名称 | 开发机构 | 开放类型 | 编程语言 | 数据源 | 核心算法架构 | 代码链接 |
|---------|---------|:---------|----------|--------|------------|----------|
| [FourCastNet](https://arxiv.org/abs/2202.11214) | NVIDIA | 开源 | Python, CUDA | ERA5（40年） | ViT + 适应性傅里叶神经算子（AFNO） | [GitHub](https://github.com/NVlabs/FourCastNet) |
| [盘古](https://arxiv.org/abs/2211.02556) | 华为 | 开源 | Python, PyTorch | ERA5（39年） | Transformer | [GitHub](https://github.com/198808xc/Pangu-Weather) |
| [GraphCast](https://arxiv.org/abs/2212.12794) | Google/DeepMind | 开源 | Python, JAX | ERA5（39年） | 图神经网络（GNN） | [GitHub](https://github.com/google-deepmind/graphcast) |
| [ClimaX](https://arxiv.org/abs/2301.10343) | Microsoft | 开源 | Python, PyTorch | ERA5, CMIP6 | Transformer | [GitHub](https://github.com/microsoft/ClimaX) |
| [风乌](https://arxiv.org/abs/2304.02948) | 上海人工智能实验室 | 开源 | Python, PyTorch | ERA5（39年） | Transformer | [GitHub](https://github.com/OpenEarthLab/FengWu) |
| [伏羲](https://arxiv.org/abs/2408.05472) | 复旦大学 | 开源 | Python, PyTorch | ERA5（39年） | U-Transformer + Cascade级联 | [GitHub](https://github.com/tpys/FuXi) |
| [Aurora](https://www.nature.com/articles/s41586-025-09005-y) | Microsoft | 开源 | Python, JAX | ERA5, MARS | Transformer | [GitHub](https://github.com/microsoft/aurora) |

### 1.2 预报能力

| 模型 | 预报时效 | 有效预报时效（Z500 ACC>0.6） | 分辨率与预报范围 | 模型内核 | 预报要素 |
|------|---------|----------------------------|----------------|---------|---------|
| 伏羲 | 中期（15天） | 约10.8天 | 0.25°×0.25°，6h，全球 | 数据驱动 | 5个地表变量（含降水）+ 5个大气变量 |
| 风乌 | 中期（10.75天） | 10.75天 | 0.25°×0.25°，6h，全球 | 数据驱动 | 4个地表变量（不含降水）+ 5个大气变量 |
| GraphCast | 中期（10天） | 9.75天 | 0.25°×0.25°，6h，全球 | 数据驱动 | 5个地表变量（含降水）+ 6个大气变量 |
| Pangu-Weather | 短/中期（1h-7天） | 超过7天 | 0.25°×0.25°，6h，全球 | 数据驱动 | 4个地表变量（不含降水）+ 5个大气变量 |
| FourCastNet | 短/中期 | 不到7天 | 0.25°×0.25°，6h，全球 | 数据驱动 | 5个地表变量（含降水）+ 4个大气变量 |
| DLWP | 短/中期 | 不到6天 | 1.9°×1.9°，6h，全球 | 数据驱动 | 4个地表变量（不含降水） |

### 1.3 硬件需求和预测速度

| 模型 | 显卡配置 | 预测速度 |
|------|---------|---------|
| 伏羲 | 8个NVIDIA A100 GPU<br>• 预训练：约30小时<br>• 微调：2天 | （数据缺失） |
| 风乌 | 32个NVIDIA A100 GPU<br>• PyTorch开发<br>• 总训练时间：17天 | • 10天预测生成时间：<30秒<br>• 能耗：12 kJ（比IFS模型低2000倍） |
| GraphCast | 32个Cloud TPU v4<br>• 训练时间：21天 | • 10天预测生成时间：60秒（35GB数据） |
| Pangu-Weather | 192块NVIDIA V100 GPU<br>• 训练配置：小时采样，100 epoch<br>• 训练时间：16天 | • 24小时全球预报：1.4秒（单V100）<br>• 比传统数值预报快10000倍 |
| FourCastNet | 4个NVIDIA A100 GPU | • 100成员24小时预报：7秒（30km分辨率）<br>• 每个GPU处理25个batch |
| DLWP | 单块NVIDIA Tesla V100 GPU<br>• 训练时间：2-3天 | • 全球28天预测：<0.2秒<br>• 1000成员月预报集合：约3分钟 |

## 2. 降水预报模型

| 模型名称 | 开发机构 | 开放类型 | 编程语言 | 数据源 | 核心算法架构 | 代码链接 |
|---------|---------|----------|----------|--------|------------|----------|
| [MetNet](https://arxiv.org/abs/2003.12140) | Google | 开源 | Python, TensorFlow | MRMS, GOES-16 | U-Net | [GitHub](https://github.com/google-research/metnet) |
| [DeepRain](https://arxiv.org/abs/1711.02316) | 韩国科学技术信息研究院 | 开源 | Python, PyTorch | 雷达数据 | CNN | 
| [NowcastNet](https://www.nature.com/articles/s41586-023-06184-4) | 清华大学 | 开源 | Python, PyTorch | 雷达观测（6年） | U-Net | [GitHub](https://github.com/google-deepmind/deepmind-research/tree/master/nowcasting) |

## 3. 环境模型

### 3.1 大气环境模型
| 模型名称 | 开发机构 | 类型 | 开放类型 | 编程语言 | 代码/资源链接 |
|---------|---------|------|----------|----------|--------------|
| [CMAQ](https://www.epa.gov/cmaq) | 美国环保署 | 大气污染 | 开源 | Fortran | [GitHub](https://github.com/USEPA/CMAQ) |
| [CALPUFF](https://www.epa.gov/scram/air-quality-dispersion-modeling-alternative-models#calpuff) | 美国环保署 | 大气污染 | 商业 | Fortran | [SRC](https://www.src.com/calpuff/) |
| [AERMOD](https://www.epa.gov/scram/air-quality-dispersion-modeling-preferred-and-recommended-models#aermod) | 美国环保署 | 大气污染 | 商业 | Fortran | [EPA](https://www.epa.gov/scram) |
| [WRF-Chem](https://www2.mmm.ucar.edu/wrf/users/) | 美国国家大气研究中心 | 大气化学 | 开源 | Fortran, C | [官方下载](https://www.acom.ucar.edu/wrf-chem/download.shtml) |
| [CAMx](https://www.camx.com/) | ENVIRON | 大气化学 | 商业 | Fortran | [官方下载](https://www.camx.com/download/source/) |
| [GEOS-Chem](http://wiki.seas.harvard.edu/geos-chem/index.php/Main_Page) | 哈佛大学 | 大气化学 | 开源 | Fortran | [GitHub](https://github.com/geoschem/geos-chem) |
| [EPICC](https://earthlab.iap.ac.cn/) | 中科院大气物理研究所 | 大气化学 | 免费申请 | Fortran | [官方下载](https://earthlab.iap.ac.cn/resdown/index.html) |

### 3.2 水环境模型
| 模型名称 | 开发机构 | 类型 | 开放类型 | 编程语言 | 代码/资源链接 |
|---------|---------|------|----------|----------|--------------|
| [Visual MODFLOW Flex](https://www.waterloohydrogeologic.com/visual-modflow-flex/) | Waterloo Hydrogeologic | 地下水 | 商业 | C#, Fortran | [官方下载](https://www.waterloohydrogeologic.com/visual-modflow-flex/) |
| [MODFLOW 6](https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model) | 美国地质调查局 | 地下水 | 开源 | Fortran | [GitHub](https://github.com/MODFLOW-USGS/modflow6) |
| [GMS](https://www.aquaveo.com/software/gms-groundwater-modeling-system-introduction) | Aquaveo | 地下水 | 商业 | C++, Python | [官方下载](https://www.aquaveo.com/downloads-gms) |
| [TOUGH](https://tough.lbl.gov/) | 劳伦斯伯克利国家实验室 | 多相流 | 商业 | Fortran | [官方下载](https://tough.lbl.gov/licensing-download/) |
| [HYDRUS](https://www.pc-progress.com/en/Default.aspx?hydrus-3d) | PC-Progress | 土壤水 | 商业 | Fortran, C++ | [官方下载](https://www.pc-progress.com/en/Default.aspx?downloads) |
| [WASP](https://www.epa.gov/hydrowq/water-quality-analysis-simulation-program-wasp) | 美国环保署 | 水质模拟 | 开源 | Fortran | [官方下载](https://www.epa.gov/hydrowq/wasp8-download) |

### 3.3 社会经济模型
| 模型名称 | 开发机构 | 类型 | 开放类型 | 编程语言 | 代码/资源链接 |
|---------|---------|------|----------|----------|--------------|
| [LEAP](https://leap.sei.org/) | 斯德哥尔摩环境研究所 | 综合评估 | 商业 | C#, .NET | [官方下载](https://leap.sei.org/default.asp?action=download) |
| [GCAM](https://github.com/JGCRI/gcam-core) | 联合全球变化研究所 | 综合评估 | 开源 | C++, R | [GitHub](https://github.com/JGCRI/gcam-core) |

## 4. 技术方案总结

### 4.1 模型类型分类

#### 4.1.1 深度学习模型
- **Transformer架构**：用于全球天气预报，如Pangu-Weather、ClimaX等
- **图神经网络**：用于天气预报和气候预测，如GraphCast
- **卷积神经网络**：用于降水预报，如MetNet、DeepRain
- **傅里叶神经算子**：用于天气预报，如FourCastNet

#### 4.1.2 数值模拟模型
- **有限差分法**：用于大气和水文模拟，如WRF、MODFLOW
- **欧拉网格法**：用于大气污染模拟，如CMAQ
- **半拉格朗日方案**：用于大气环流模拟，如GEM

### 4.2 开发语言分布

#### 4.2.1 按领域划分的主要开发语言

| 领域 | 主要编程语言 | 说明 |
|------|------------|------|
| 大气环境 | Fortran, C | 大多数传统大气模型（WRF-chem、CMAQ、CAMx等）主要使用Fortran开发，部分模块使用C语言 |
| 地表水/海洋 | C#, Fortran, C++ | 时空增益模型主要使用C#，传统水文模型使用Fortran和C++，支持河流湖库与海洋联合模拟 |
| 土壤地下水 | Fortran | MODFLOW、TOUGH、HYDRUS等主要使用Fortran开发 |
| 经济社会 | GAMS, Python, Java | CGE模型主要使用GAMS，系统动力学模型使用Python和Java |
| 生态系统 | Python, R | 新开发的生态模型主要使用Python和R |

#### 4.2.2 开发语言特点

| 编程语言 | 主要应用场景 | 优势 | 局限性 |
|---------|------------|------|--------|
| Fortran | 数值计算密集型模型 | 高性能计算，数组运算效率高 | 生态系统不完善，现代特性支持有限 |
| C/C++ | 性能关键模块 | 底层控制，高性能 | 开发效率相对较低 |
| Python | 新型模型开发，数据处理 | 生态系统丰富，开发效率高 | 计算性能相对较低 |
| GAMS | 经济模型 | 优化问题求解能力强 | 专用语言，应用范围受限 |
| Java | 系统动力学 | 跨平台，面向对象 | 数值计算性能一般 |
| C# | 水文模型 | .NET生态，开发效率高 | 跨平台支持相对较弱 |

### 4.3 开发建议规范

#### 4.3.1 操作系统版本

传统数据模型：推荐使用centos 7+，稳定性优先。
深度学习相关模型开发：Ubuntu 22.04 LTS（CUDA支持完善），兼容docker部署。

#### 4.3.2 模型开发语言推荐

| 编程语言 | 兼容性推荐版本 | 新项目建议版本 | 关键说明 |
|---------|--------------|--------------|---------|
| Fortran | Fortran 90/95 | Fortran 2008+ | • 90/95：气象模式（如WRF）广泛使用，兼容性好<br>• 2008+：支持面向对象、并行计算增强，适合新项目开发 |
| C | C11 | C17/C23 | C11平衡现代性与兼容性，新项目可选C17（稳定）或C23（前沿）。 |
| C++ | C++17 | C++20/C++23 | C++17工业级稳定，C++20引入模块化/协程等现代特性。 |
| Python | Python 3.9/3.10 (LTS) | Python 3.11+ | Python 2已淘汰，新项目必选3.x，3.10+支持最新库特性。 |

#### 4.3.3 数据服务和开发语言

1. **数据采集与存储**
   - Kafka + Flume：日志收集
   - HBase + HDFS：数据持久化
   - Elasticsearch：实时检索

2. **数据处理与分析**
   - Spark + MLlib：机器学习
   - Flink + CEP：复杂事件处理
   - Hive + Presto：交互式查询

3. **数据应用与服务**
   - Spring Cloud：微服务架构
   - Zookeeper：服务注册与发现
   - Kubernetes：容器编排部署

### 4.4 开发环境管理方案

#### 4.4.1 核心工具链建议

1. **代码管理**  
   推荐使用 **Git**，相关网站，国内团队可选Gitee，支持高效分支管理和多人协作，以及开发代码进度管理。

2. **依赖管理**  
   - Python项目：优先用 **Conda**（尤其适合数据科学），能管理多语言依赖和预编译包，包括很多数据计算的包。
   - Java项目：使用 **Maven**，提供中央仓库，支持依赖传递，构建流程标准化。
   - C/C++项目：推荐 **Conan** 或 **vcpkg**，管理第三方库依赖，跨平台支持。
   - Fortran项目：使用 **FPM**（Fortran Package Manager）管理依赖，或通过 **CMake** 配合其他包管理工具。

3. **开发部署**  
   建议 **Docker** 容器化部署，确保环境一致性，便于移植和扩展。跨语言的复杂环境，建议使用dockerfile进行管理。

## 参考资源

- [AI-for-Earth-Paper-List](https://github.com/OpenEarthLab/AI-for-Earth-Paper-List) - OpenEarthLab整理的AI地球科学论文列表
- [Awesome-AI-for-Atmosphere-and-Ocean](https://github.com/XiongWeiTHU/Awesome-AI-for-Atmosphere-and-Ocean) - 清华大学整理的AI大气和海洋科学论文列表
- [AI气象大模型比较](https://mp.weixin.qq.com/s?__biz=MzA5OTQ5Njk5Nw==&mid=2247491456&idx=1&sn=e1b20651efd5fecec1b146fdb3bc61b2&chksm=912ec6bf38b28f00fe1eb26be80c711ea099317c6d43c6b7f1f3fef8d33e3ce59969e9c822f1) - AI气象大模型比较