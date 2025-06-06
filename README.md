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


## 3. 技术方案总结

### 3.1 模型类型分类

#### 3.1.1 深度学习模型
- **Transformer架构**：用于全球天气预报，如Pangu-Weather、ClimaX等
- **图神经网络**：用于天气预报和气候预测，如GraphCast
- **卷积神经网络**：用于降水预报，如MetNet、DeepRain
- **傅里叶神经算子**：用于天气预报，如FourCastNet

#### 3.1.2 数值模拟模型
- **有限差分法**：用于大气和水文模拟，如WRF、MODFLOW
- **欧拉网格法**：用于大气污染模拟，如CMAQ
- **半拉格朗日方案**：用于大气环流模拟，如GEM


### 3.2 开发语言分布

#### 3.2.1 按领域划分的主要开发语言

| 领域 | 主要编程语言 | 说明 |
|------|------------|------|
| 大气环境 | Fortran, C | 大多数传统大气模型（WRF-chem、CMAQ、CAMx等）主要使用Fortran开发，部分模块使用C语言 |
| 地表水/海洋 | C#, Fortran, C++ | 时空增益模型主要使用C#，传统水文模型使用Fortran和C++，支持河流湖库与海洋联合模拟 |
| 土壤地下水 | Fortran | MODFLOW、TOUGH、HYDRUS等主要使用Fortran开发 |
| 经济社会 | GAMS, Python, Java | CGE模型主要使用GAMS，系统动力学模型使用Python和Java |
| 生态系统 | Python, R | 新开发的生态模型主要使用Python和R |

### 3.2.2 开发语言特点

| 编程语言 | 主要应用场景 | 优势 | 局限性 |
|---------|------------|------|--------|
| Fortran | 数值计算密集型模型 | 高性能计算，数组运算效率高 | 生态系统不完善，现代特性支持有限 |
| C/C++ | 性能关键模块 | 底层控制，高性能 | 开发效率相对较低 |
| Python | 新型模型开发，数据处理 | 生态系统丰富，开发效率高 | 计算性能相对较低 |
| GAMS | 经济模型 | 优化问题求解能力强 | 专用语言，应用范围受限 |
| Java | 系统动力学 | 跨平台，面向对象 | 数值计算性能一般 |
| C# | 水文模型 | .NET生态，开发效率高 | 跨平台支持相对较弱 |

## 4. 开发建议规范

### 4.1 操作系统版本

传统数据模型：推荐使用centos 7+，稳定性优先。
深度学习相关模型开发：Ubuntu 22.04 LTS（CUDA支持完善），兼容docker部署。


### 4.2 模型开发语言推荐
#### 4.2.1 模型开发语言

| 编程语言 | 兼容性推荐版本 | 新项目建议版本 | 关键说明 |
|---------|--------------|--------------|---------|
| Fortran | Fortran 90/95 | Fortran 2008+ | • 90/95：气象模式（如WRF）广泛使用，兼容性好<br>• 2008+：支持面向对象、并行计算增强，适合新项目开发 |
| C | C11 | C17/C23 | C11平衡现代性与兼容性，新项目可选C17（稳定）或C23（前沿）。 |
| C++ | C++17 | C++20/C++23 | C++17工业级稳定，C++20引入模块化/协程等现代特性。 |
| Python | Python 3.9/3.10 (LTS) | Python 3.11+ | Python 2已淘汰，新项目必选3.x，3.10+支持最新库特性。 |


### 4.3 数据服务和开发语言

#### 4.3.1 Java生态系统
推荐使用JAVA:

| 特性 | 优势 | 应用场景 |
|------|------|---------|
| 高性能 | JVM优化、JIT编译、垃圾回收机制 | 大规模数据处理、实时计算 |
| 跨平台 | Write Once, Run Anywhere | 分布式系统部署、云平台服务 |
| 生态成熟 | 丰富的开源框架和工具 | 企业级应用开发、微服务架构 |
| 并发处理 | 线程池、虚拟线程、响应式编程 | 高并发服务、异步处理 |
| 安全特性 | 类型安全、安全管理器、加密API | 金融系统、安全敏感应用 |
| 开发效率 | 强大IDE支持、丰富调试工具 | 快速迭代、团队协作 |

Java版本选择：

| 编程语言 | 兼容性推荐版本 | 新项目建议版本 | 关键说明 |
|---------|--------------|--------------|---------|
| Java | Java 11 LTS | Java 17 LTS | • Java 11：长期支持至2026，HTTP Client API完善，模块化系统成熟，主流框架兼容性好，容器化支持完整<br>• Java 17：长期支持至2029，性能优化显著，密封类增强，Pattern Matching特性，更完善的空指针提示，适合云原生开发 |


#### 4.3.2 Java生态大数据构件

| 构件类型 | 框架/工具 | 主要功能 | 应用场景 |
|---------|----------|---------|----------|
| 分布式计算 | Apache Hadoop | 分布式存储和计算框架 | 大规模数据批处理 |
| 分布式计算 | Apache Spark | 内存计算、流处理、机器学习 | 实时分析、数据挖掘 |
| 流计算引擎 | Apache Flink | 有状态计算、事件处理 | 实时计算、复杂事件处理 |
| 消息中间件 | Apache Kafka | 高吞吐消息队列 | 日志收集、流式处理 |
| 数据仓库 | Apache Hive | SQL查询、数据仓库 | 数据分析、报表生成 |
| 数据存储 | HBase | 分布式列式数据库 | 大规模数据存储 |
| 资源调度 | YARN | 集群资源管理 | 任务调度、资源分配 |
| 工作流调度 | Apache Airflow | DAG工作流管理 | 任务编排、调度控制 |

#### 4.3.3 构件集成应用

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