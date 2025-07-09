# 实时点云姿态估计与STL模型可视化程序

这个程序集成了两个现有的姿态估计算法，实现了实时点云获取、姿态估计和STL模型可视化。

## 重要提示

⚠️ **此程序需要真实的Mecheye相机设备才能运行，不支持虚拟数据模式。**

## 功能特点

- **实时点云获取**: 从Mecheye相机实时获取点云数据
- **粗配准**: 使用PyTorch实现的基于优化的粗配准
- **精配准**: 使用PyTorch实现的ICP算法进行精确配准
- **实时可视化**: 实时显示场景点云和匹配后的STL模型
- **配置文件支持**: 通过YAML文件灵活配置所有参数

## 文件说明

- `realtime_pose_estimation_with_stl_visualization.py` - 主程序
- `realtime_pose_config.yaml` - 配置文件
- `run_realtime_pose_estimation.py` - 启动脚本
- `stp/part2_rude.STL` - 目标STL模型文件

## 依赖要求

### 必需依赖
```bash
pip install open3d torch numpy scipy
```

### 可选依赖
```bash
pip install pyyaml  # 用于配置文件支持
```

### Mecheye SDK（必需）
**必须安装Mecheye Python SDK并连接相机才能运行此程序。**
程序不支持虚拟数据模式，需要真实的相机设备。

## 使用方法

### 1. 快速启动
```bash
python run_realtime_pose_estimation.py
```

### 2. 直接运行主程序
```bash
python realtime_pose_estimation_with_stl_visualization.py
```

## 配置参数

编辑 `realtime_pose_config.yaml` 文件来调整参数：

### 主要参数说明

- **model_file**: STL模型文件路径
- **pose_estimation_interval**: 姿态估计间隔（秒）
- **icp.threshold**: ICP对应点距离阈值
- **coarse_alignment.iterations**: 粗配准迭代次数
- **visualization.window_width/height**: 可视化窗口大小

## 程序流程

1. **初始化**: 加载配置、连接相机、加载STL模型
2. **实时循环**:
   - 从相机获取点云数据
   - 定期进行姿态估计（粗配准 → ICP精配准）
   - 更新可视化显示（场景点云 + 变换后的STL模型）
3. **退出**: 关闭可视化窗口或按Ctrl+C

## 操作说明

### 运行时操作
- **关闭窗口**: 程序退出
- **Ctrl+C**: 强制退出程序

### 可视化说明
- **蓝色点云**: 实时获取的场景点云
- **绿色模型**: 经过姿态估计变换后的STL模型
- **坐标轴**: 显示世界坐标系

## 算法说明

### 粗配准
- 使用PyTorch实现的基于优化的方法
- 通过最小化最近点距离进行配准
- 支持GPU加速

### ICP精配准
- 点到点ICP算法
- 使用SVD求解最优变换
- 支持收敛检测

## 性能优化

### 点云预处理
- 体素下采样减少计算量
- 统计离群点移除提高鲁棒性
- 最远点采样保持点云分布

### 算法优化
- 使用PyTorch GPU加速
- 自适应收敛检测
- 分层配准策略（粗配准 → 精配准）

## 故障排除

### 常见问题

1. **相机连接失败**
   - 检查Mecheye SDK安装
   - 程序会自动使用虚拟数据继续运行

2. **STL文件未找到**
   - 检查文件路径是否正确
   - 确保 `stp/part2_rude.STL` 文件存在

3. **配置文件加载失败**
   - 检查YAML语法
   - 程序会使用默认参数继续运行

4. **性能问题**
   - 调整点云预处理参数
   - 减少姿态估计频率
   - 使用GPU加速

### 调试信息
程序会输出详细的运行信息，包括：
- 配准迭代过程
- 配准质量指标（Fitness、RMSE）
- 性能统计信息

## 扩展功能

### 自定义模型
替换 `model_file` 参数来使用不同的STL模型。

### 参数调优
根据具体应用场景调整配置文件中的参数：
- 提高精度：增加迭代次数、减小收敛阈值
- 提高速度：减少点云数量、降低姿态估计频率

### 结果保存
可以扩展程序来保存姿态估计结果到文件。

## 技术支持

如有问题，请检查：
1. 依赖是否正确安装
2. 配置文件格式是否正确
3. STL文件是否存在
4. 相机连接状态 