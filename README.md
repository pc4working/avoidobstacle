# avoidobstacle

基于 ZED 深度相机的单帧避障模块，面向 RTK 巡航场景。

这个目录实现了一个可独立维护的小仓库，目标是把原有导航系统中的通行性分析和扇形轨迹评分逻辑拆出来，供 `rtk_cors_4g.py` 之类的 RTK 导航脚本按周期调用。

## 目录结构

```text
avoidobstacle/
  __init__.py
  config.py
  coordinator.py
  planner.py
  tests/
  zazzy-foraging-sedgewick.md
```

## 核心模块

- `config.py`
  - 定义 `AvoidanceConfig` 和 `AvoidanceResult`
- `coordinator.py`
  - 定义 `ZedFrameCoordinator`
  - 负责在 ZED 采集线程和避障规划器之间共享同一台相机
  - 内部把 `RIGHT_HANDED_Z_UP` 坐标系转换为规划侧使用的 `X forward / Y left / Z up`
- `planner.py`
  - 定义 `ObstacleAvoidancePlanner`
  - 负责单帧点云下采样、高程图更新、近场缺失深度封锁、强制可通行区域应用，以及扇形轨迹评分

## 公开 API

```python
from avoidobstacle import (
    AvoidanceConfig,
    AvoidanceResult,
    FrameData,
    ObstacleAvoidancePlanner,
    ZedFrameCoordinator,
)
```

典型使用方式：

1. 在 ZED 初始化完成后创建 `ZedFrameCoordinator(camera)`
2. 在 ZED 工作线程里循环调用 `coordinator.grab_and_get_pose()`
3. 创建 `ObstacleAvoidancePlanner(config, coordinator)` 并调用 `setup()`
4. 在导航循环中调用 `plan_avoidance(goal_relative_angle_deg=...)`

## 依赖关系

- 直接复用 `../navigation/realtimetravgen_v1_1.py` 中的高程图和通行性处理函数
- 相机安装外参默认读取 `../ysc_rtk/unitree_zed/CAM_TO_BASE_PARAMS`
- 设计说明见 `zazzy-foraging-sedgewick.md`

## 测试

当前目录包含纯逻辑测试，主要覆盖：

- 坐标系转换
- 点云坐标转换
- 扇形轨迹评分的目标偏置行为

运行方式：

```bash
PYTHONPATH=/home/pc/code/robotnav/robotnav python3 -m pytest tests/test_coordinator.py -q
```

## 当前边界

- 本目录只维护避障模块本身，不包含完整的机器人运行环境
- `rtk_cors_4g.py` 的接入代码位于上级目录 `../ysc_rtk/`
- 实机运行仍依赖 ZED SDK、`elevation_mapping_cupy_core` 以及机器人控制侧环境
