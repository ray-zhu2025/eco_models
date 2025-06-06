# WRF Docker 部署说明

## 环境要求
- Docker
- Docker Compose

## 使用方法

1. 构建镜像：
```bash
docker-compose build
```

2. 启动容器：
```bash
docker-compose up -d
```

3. 进入容器：
```bash
docker-compose exec wrf bash
```

4. 在容器内运行WRF：
```bash
cd WRF/test/em_real
./wrf.exe
```

## 数据目录
- 输入数据放在 `./data` 目录下
- 输出结果也会保存在 `./data` 目录中

## 注意事项
- 首次运行需要编译WRF，可能需要较长时间
- 确保有足够的磁盘空间用于编译和运行
- 建议使用SSD以提高性能 