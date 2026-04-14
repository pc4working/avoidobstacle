# 避障模块设计方案：RTK导航中的单帧通行性分析与轨迹评分

## Context

`rtk_cors_4g.py` 目前只有纯GPS/RTK打点导航，没有任何障碍物检测和规避能力。而 `lynxnav_v1.5.py` 拥有成熟的基于ZED深度相机的通行性分析和扇形轨迹评分算法，但它与VIO累积、道路分割、多帧融合等深度绑定，无法直接被RTK导航调用。

**目标**：将通行性分析（不带道路分割）和轨迹评分逻辑提取为独立模块，RTK导航在巡航时周期性调用，获取最佳避障轨迹方向和安全距离。

---

## 模块结构

```
avoidobstacle/
  __init__.py           # 公开API导出
  config.py             # AvoidanceConfig 参数配置 + AvoidanceResult 结果
  planner.py            # ObstacleAvoidancePlanner 核心规划类
  coordinator.py        # ZedFrameCoordinator ZED相机帧共享协调器
```

---

## 文件设计

### 1. `config.py` — 参数与结果

**`AvoidanceConfig` dataclass**：镜像 `realtimetravgen_v1_1.py` 的相关参数，设默认值：

| 参数 | 默认值 | 来源 |
|------|--------|------|
| resolution | 0.04 | 地图单元格分辨率(m) |
| map_length | 8.0 | 地图边长(m) |
| downsample_stride | 8 | 点云下采样步长 |
| camera_pitch_deg | 45.0 | 相机俯仰角 |
| camera_offset_forward/left/up_m | 0.38/0.065/0.18 | 来自 CAM_TO_BASE_PARAMS |
| blocked_threshold | 0.5 | 通行性阈值 |
| near_depth_block_distance | 1.5 | 近场缺失深度封锁距离(m) |
| forced_traversable_forward_distance | 0.12 | 强制可通行前方距离(m) |
| forced_traversable_half_width | 0.6 | 强制可通行半宽(m) |
| trajectory_length | 2.0 | 轨迹标称长度(m) |
| trajectory_min_length | 0.3 | 最短有效轨迹(m) |
| trajectory_width_px | 8 | 走廊宽度(像素) |
| trajectory_angle_max_deg | 75.0 | 扇形最大半角 |
| trajectory_angle_step_deg | 3.0 | 角度步进 |
| band_max_width_px | 8 | 方向性波段修复参数 |
| traversable_base_cost | None | 可选：固定通行代价 |
| preferred_base_cost | None | 可选：偏好区域代价 |
| depth_mode | "NEURAL_PLUS" | ZED深度模式 |
| avoidance_check_interval | 0.5 | 避障检查间隔(s) |

**`AvoidanceResult` dataclass**：

```python
@dataclass
class AvoidanceResult:
    status: str                    # "ok" / "no_path" / "error"
    recommended_angle_deg: float   # 相对机体前方的推荐转向角（正=左，CCW俯视）
    safe_distance_m: float         # 最佳轨迹安全距离(m)
    max_reachable_distance_m: float  # 所有候选中最远可达距离
    score: float                   # 最佳轨迹评分
    raw_result: dict | None        # 完整trajectory_result字典
    timing_ms: dict                # 各阶段耗时
    error_message: str = ""
```

---

### 2. `coordinator.py` — ZED相机帧共享

**核心问题**：ZED SDK 对同一设备只允许一个 `Camera` 对象。`rtk_cors_4g.py` 的 `task_zed()` 线程已占用相机。避障模块不能再创建第二个。

**方案**：`ZedFrameCoordinator` 包装已有的 `sl.Camera` 对象，用请求-响应模式在ZED线程和导航线程间安全共享帧数据。

```python
class ZedFrameCoordinator:
    """
    包装 pyzed.sl.Camera，让 task_zed() 和避障规划器共享同一相机。
    
    工作流：
    1. rtk_cors_4g.py 创建 ZedFrameCoordinator(camera) 传入已初始化的相机
    2. task_zed() 每帧调 coordinator.grab_and_get_pose() 获取位姿（同原流程）
    3. 规划器调 coordinator.request_frame() 阻塞等待下一帧的点云+位姿数据
    """
```

**关键设计**：

- **按需获取深度**：只在 `_frame_requested` 标志为True时才调 `camera.retrieve_measure(MEASURE.XYZ)` 获取点云，避免每帧都做深度计算
- **线程同步**：`threading.Event` 通知机制，`threading.Lock` 保护共享数据
- **坐标系转换**：`rtk_cors_4g.py` 用 `RIGHT_HANDED_Z_UP`（X=右,Y=前,Z=上），而 `realtimetravgen_v1_1.py` 的函数假设 `RIGHT_HANDED_Z_UP_X_FWD`（X=前,Y=左,Z=上）。协调器内部做旋转变换：

```python
# RIGHT_HANDED_Z_UP → RIGHT_HANDED_Z_UP_X_FWD 的变换矩阵
R_CONVERT = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)

# 对位姿和点云分别变换：
rotation_xfwd = R_CONVERT @ rotation_zup @ R_CONVERT.T
translation_xfwd = R_CONVERT @ translation_zup
xyz_xfwd[:, 0], xyz_xfwd[:, 1] = xyz_zup[:, 1], -xyz_zup[:, 0]  # 逐点变换
```

- **`cam_extrinsic` 不受影响**：协调器将原始 `RIGHT_HANDED_Z_UP` 位姿传回给 `task_zed()` 更新 `cam_extrinsic`，仅在提供给规划器的数据上做变换

**接口**：

```python
def grab_and_get_pose(self) -> tuple[int, float, float, float, float, float, float] | None:
    """task_zed()每帧调用。返回 (tracking_state, qx, qy, qz, qw, tx, ty)。
    如果有规划请求待处理，顺便获取点云存入缓冲区。"""

def request_frame(self, timeout_sec=1.0) -> FrameData | None:
    """规划器调用，阻塞等待下一帧的点云+位姿数据（已转为X_FWD坐标系）。"""
```

**`FrameData` namedtuple**：rotation(3x3), translation(3,), confidence, xyz_image(HxWx3), intrinsics(fx,fy,cx,cy)

---

### 3. `planner.py` — 核心规划器

**复用已有函数**（通过 `sys.path` 添加 `navigation/` 目录后导入）：

| 函数 | 来源文件 | 用途 |
|------|----------|------|
| `init_elevation_map(args)` | realtimetravgen_v1_1.py | 创建高程图 |
| `build_traversability_postprocessor(args)` | realtimetravgen_v1_1.py | 通行性后处理器 |
| `downsample_pointcloud(xyz, stride)` | realtimetravgen_v1_1.py | 点云下采样 |
| `build_near_depth_block_mask(...)` | realtimetravgen_v1_1.py | 近场深度封锁 |
| `build_traversability_masks(...)` | realtimetravgen_v1_1.py | 通行性掩码组合 |
| `build_forced_traversable_mask(...)` | realtimetravgen_v1_1.py | 强制可通行区域 |
| `apply_forced_traversable_mask(...)` | realtimetravgen_v1_1.py | 应用强制掩码 |
| `plan_scored_fan_trajectory_with_goal_bias(...)` | lynxnav_v1.5.py:524 | 核心轨迹评分 |
| `extract_base_pose_with_translation_offset(...)` | lynxnav_v1.5.py | 带偏移的基座位姿 |
| `get_cached_local_grid_xy(...)` | lynxnav_v1.5.py | 缓存网格坐标 |

**`ObstacleAvoidancePlanner` 类**：

```python
class ObstacleAvoidancePlanner:
    def __init__(self, config: AvoidanceConfig, coordinator: ZedFrameCoordinator): ...
    def setup(self) -> None: ...          # 初始化高程图、后处理器
    def plan_avoidance(self, goal_relative_angle_deg: float | None = None) -> AvoidanceResult: ...
    def close(self) -> None: ...
```

**`plan_avoidance()` 单帧管线**（每次调用 ~40-100ms）：

```
1. coordinator.request_frame()            # 获取当前帧点云+位姿（已转X_FWD坐标系）
2. extract_base_pose_with_translation_offset()  # 相机位姿→机体位姿
3. elmap.clear()                          # 清空高程图（单帧模式的关键）
4. downsample_pointcloud()                # 下采样点云
5. elmap.move_to() + elmap.input_pointcloud()  # 更新高程图
6. elmap.get_map_with_name_ref("traversability")  # 提取通行性
7. trav_postprocessor()                   # 方向性波段修复
8. build_near_depth_block_mask()          # 近场缺失深度封锁
9. build_traversability_masks()           # 组合阈值（road_map传全零，不用道路分割）
10. build_forced_traversable_mask() + apply  # 机体正前方强制可通行
11. plan_scored_fan_trajectory_with_goal_bias()  # 扇形轨迹评分 + 目标偏置
12. 封装 AvoidanceResult 返回
```

---

### 4. 坐标系转换详解

**角度约定对照**：

| 系统 | 正方向 | 0度指向 |
|------|--------|---------|
| ZED base_link frame | 左=CCW俯视 | 机体前方 |
| RTK 航向 | CW 顺时针 | 真北 |
| `best_angle_deg` 输出 | +值=左(CCW) | 机体前方 |
| RTK `error = normalize_angle(heading - target)` | +值=需右转 | — |

**RTK调用时的转换**：

```python
# rtk_cors_4g.py navigate_to() 中:
# 1. 计算目标在机体坐标系中的相对角度
#    error = normalize_angle(current_heading - target_heading)  # +值=目标在右
#    goal_relative_angle_deg = -error  # 转为base_link坐标（+值=左）

# 2. 调用避障
result = planner.plan_avoidance(goal_relative_angle_deg=-error)

# 3. 使用结果
if result.status == "ok":
    # recommended_angle_deg: +值=建议左转(CCW)
    # RTK航向调整: 左转=减航向
    adjusted_target_heading = (current_heading - result.recommended_angle_deg) % 360
    error = normalize_angle(current_heading - adjusted_target_heading)
```

**轨迹终点输出**（如需要绝对坐标）：

`best_centerline_world_xy` 是ZED世界坐标系的XY坐标。对于RTK导航，通常不需要这个绝对坐标，只需要 `recommended_angle_deg`（方向）和 `safe_distance_m`（距离）。如果需要绝对坐标，可通过当前RTK位置 + 方向 + 距离计算UTM坐标。

---

### 5. `rtk_cors_4g.py` 改动

#### 5.1 ZED初始化改动

```python
# 原来 task_zed() 中:
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
# ...
zed.open(init_params)

# 改为:
from avoidobstacle import ZedFrameCoordinator, AvoidanceConfig, ObstacleAvoidancePlanner

zed = sl.Camera()
init_params = sl.InitParameters()
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS  # 新增：启用深度
# ...
zed.open(init_params)
zed_coordinator = ZedFrameCoordinator(zed)  # 包装相机
```

#### 5.2 task_zed() 循环改动

```python
# 原来:
if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
    pose = sl.Pose()
    zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
    qx, qy, qz, qw = pose.get_orientation().get()
    tx = pose.get_translation().get()[0]
    ty = pose.get_translation().get()[1]
    cam_extrinsic.update(qx, qy, qz, qw, tx, ty)

# 改为:
pose_data = zed_coordinator.grab_and_get_pose()
if pose_data is not None:
    _, qx, qy, qz, qw, tx, ty = pose_data
    cam_extrinsic.update(qx, qy, qz, qw, tx, ty)
```

#### 5.3 navigate_to() 中加入避障

在远距离阶段（dist >= 0.7m）的循环中，周期性调用避障：

```python
AVOIDANCE_CHECK_INTERVAL = 0.5  # 每0.5秒检查一次
last_avoidance_time = 0.0

# 在 navigate_to 循环内，计算完 target_heading 和 error 后：
if avoidance_planner and dist >= NAV_CLOSE_STAGE_DIST:
    now = time.time()
    if now - last_avoidance_time > AVOIDANCE_CHECK_INTERVAL:
        result = avoidance_planner.plan_avoidance(
            goal_relative_angle_deg=-error  # RTK error → base_link角度
        )
        last_avoidance_time = now
        if result.status == "ok" and result.recommended_angle_deg is not None:
            adjusted_heading = (current_heading - result.recommended_angle_deg) % 360
            error = normalize_angle(current_heading - adjusted_heading)
            # 更新 dist 为 min(到目标距离, 安全距离)
            # 让速度控制不超过安全距离
```

#### 5.4 主函数初始化

```python
# 在校准完成后、导航循环之前：
avoidance_config = AvoidanceConfig()
avoidance_planner = ObstacleAvoidancePlanner(avoidance_config, zed_coordinator)
avoidance_planner.setup()
```

---

### 6. 需要修改的文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `avoidobstacle/__init__.py` | **新建** | 导出公开API |
| `avoidobstacle/config.py` | **新建** | 配置和结果数据类 |
| `avoidobstacle/coordinator.py` | **新建** | ZED帧共享协调器 |
| `avoidobstacle/planner.py` | **新建** | 核心避障规划器 |
| `ysc_rtk/rtk_cors_4g.py` | **修改** | 集成避障模块 |

**不修改**的文件（只读依赖）：
- `navigation/realtimetravgen_v1_1.py` — 直接import其函数
- `navigation/lynxnav_v1.5.py` — 直接import其函数
- `ysc_rtk/unitree_zed/extrinsic.py` — 保持不变

---

### 7. 验证方案

1. **单元测试**：构造模拟点云数据（平面+障碍物方块），验证 `plan_avoidance()` 返回正确角度
2. **集成测试**：
   - 启动 `rtk_cors_4g.py`，确认ZED航向校准仍正常（`cam_extrinsic`不受影响）
   - 在无障碍场景下RTK导航，确认避障模块返回 `recommended_angle_deg ≈ 0`（与目标方向一致）
   - 在有障碍场景下，确认 `recommended_angle_deg` 偏离障碍方向，狗能绕行
3. **性能测试**：测量单次 `plan_avoidance()` 耗时，预期 40-100ms，不超过 `avoidance_check_interval`
