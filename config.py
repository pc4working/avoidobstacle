from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace


_DEFAULT_CAM_TO_BASE = {
    "pitch_deg": 45.0,
    "tx": 0.38,
    "ty": 0.065,
    "tz": 0.18,
}

_YSC_RTK_DIR = Path(__file__).resolve().parent.parent / "ysc_rtk"
if str(_YSC_RTK_DIR) not in sys.path:
    sys.path.insert(0, str(_YSC_RTK_DIR))

try:
    from unitree_zed import CAM_TO_BASE_PARAMS as _CAM_TO_BASE_PARAMS
except Exception:
    _CAM_TO_BASE_PARAMS = _DEFAULT_CAM_TO_BASE


@dataclass(slots=True)
class AvoidanceConfig:
    resolution: float = 0.04
    map_length: float = 8.0
    downsample_stride: int = 8
    camera_pitch_deg: float = float(_CAM_TO_BASE_PARAMS.get("pitch_deg", 45.0))
    camera_offset_forward_m: float = float(_CAM_TO_BASE_PARAMS.get("tx", 0.38))
    camera_offset_left_m: float = float(_CAM_TO_BASE_PARAMS.get("ty", 0.065))
    camera_offset_up_m: float = float(_CAM_TO_BASE_PARAMS.get("tz", 0.18))
    blocked_threshold: float = 0.5
    near_depth_block_distance: float = 1.5
    forced_traversable_forward_distance: float = 0.12
    forced_traversable_half_width: float = 0.6
    trajectory_length: float = 2.0
    trajectory_min_length: float = 0.3
    trajectory_width_px: int = 8
    trajectory_angle_max_deg: float = 75.0
    trajectory_angle_step_deg: float = 3.0
    band_max_width_px: int = 8
    traversable_base_cost: float | None = None
    preferred_base_cost: float | None = None
    depth_mode: str = "NEURAL_PLUS"
    avoidance_check_interval: float = 0.5
    band_min_lateral_span_px: int = 6
    band_min_aspect_ratio: float = 1.5
    band_border_blend_radius_px: int = 1

    def to_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(**asdict(self))


@dataclass(slots=True)
class AvoidanceResult:
    status: str
    recommended_angle_deg: float
    safe_distance_m: float
    max_reachable_distance_m: float
    score: float
    raw_result: dict | None
    timing_ms: dict[str, float]
    error_message: str = ""
