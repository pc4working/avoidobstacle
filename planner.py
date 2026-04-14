from __future__ import annotations

import importlib
import math
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np

from .config import AvoidanceConfig, AvoidanceResult
from .coordinator import ZedFrameCoordinator


_MODULE_CACHE: dict[str, ModuleType] = {}
_REPO_ROOT = Path(__file__).resolve().parent.parent
_NAVIGATION_DIR = _REPO_ROOT / "navigation"
_YSC_RTK_DIR = _REPO_ROOT / "ysc_rtk"
_TRAJECTORY_GRID_LOCAL_XY_CACHE: dict[tuple[tuple[int, int], int, float], np.ndarray] = {}


def _ensure_sys_path() -> None:
    for path in (_NAVIGATION_DIR, _YSC_RTK_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_rt_module() -> ModuleType:
    _ensure_sys_path()
    cached = _MODULE_CACHE.get("rt")
    if cached is not None:
        return cached

    module = importlib.import_module("realtimetravgen_v1_1")
    _MODULE_CACHE["rt"] = module
    return module


def _normalize_angle(angle_deg: float) -> float:
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


# `lynxnav_v1.5.py` is structured as a runnable script and pulls in optional
# runtime-only dependencies. Keep the pure math helpers local so the planner
# can stay importable in lighter environments.
def _get_cached_local_grid_xy(rt: ModuleType, traj_shape: tuple[int, int], cell_n: int, resolution: float) -> np.ndarray:
    cache_key = (tuple(int(v) for v in traj_shape), int(cell_n), float(resolution))
    cached = _TRAJECTORY_GRID_LOCAL_XY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rows, cols = np.indices(traj_shape, dtype=np.int32)
    vis_indices = np.stack([rows, cols], axis=-1).reshape(-1, 2)
    local_xy = rt.vis_indices_to_world_xy(
        vis_indices,
        np.zeros(2, dtype=np.float32),
        int(cell_n),
        float(resolution),
    ).astype(np.float32, copy=False)
    _TRAJECTORY_GRID_LOCAL_XY_CACHE[cache_key] = local_xy
    return local_xy


def _extract_base_pose_with_translation_offset(
    rt: ModuleType,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_pitch_deg: float,
    camera_offset_forward_m: float,
    camera_offset_left_m: float,
    camera_offset_up_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    camera_to_base_rotation = rt.rotation_matrix_y(camera_pitch_deg)
    base_rotation = rotation @ camera_to_base_rotation.T
    camera_offset_in_base = np.array(
        [
            float(camera_offset_forward_m),
            float(camera_offset_left_m),
            float(camera_offset_up_m),
        ],
        dtype=np.float32,
    )
    base_translation = translation - base_rotation @ camera_offset_in_base
    return base_rotation.astype(np.float32, copy=False), base_translation.astype(np.float32, copy=False)


def _plan_scored_fan_trajectory_with_goal_bias(
    rt: ModuleType,
    trav_data: np.ndarray,
    blocked_mask: np.ndarray,
    preferred_mask: np.ndarray | None,
    camera_xy: np.ndarray,
    base_rotation: np.ndarray,
    map_center_xy: np.ndarray,
    cell_n: int,
    resolution: float,
    trajectory_length_m: float,
    trajectory_min_length_m: float,
    trajectory_width_px: int,
    trajectory_angle_max_deg: float,
    trajectory_angle_step_deg: float,
    traversable_base_cost: float | None = None,
    preferred_base_cost: float | None = None,
    goal_relative_angle_deg: float | None = None,
) -> dict:
    angles_deg = rt.build_trajectory_angles(trajectory_angle_max_deg, trajectory_angle_step_deg)
    result = rt.make_empty_trajectory_result(
        traj_shape=trav_data.shape,
        target_length_m=trajectory_length_m,
        min_length_m=trajectory_min_length_m,
        width_px=trajectory_width_px,
        resolution=resolution,
        angle_max_deg=trajectory_angle_max_deg,
        angle_step_deg=trajectory_angle_step_deg,
        candidate_count=int(angles_deg.shape[0]),
    )
    result["best_raw_score"] = 0.0
    result["best_goal_weight"] = 1.0
    result["best_goal_angle_error_deg"] = None
    result["goal_relative_angle_deg"] = None if goal_relative_angle_deg is None else float(goal_relative_angle_deg)

    forward_axis = np.asarray(base_rotation[:2, 0], dtype=np.float32)
    lateral_axis = np.asarray(base_rotation[:2, 1], dtype=np.float32)
    forward_norm = float(np.linalg.norm(forward_axis))
    lateral_norm = float(np.linalg.norm(lateral_axis))
    if forward_norm <= 1e-6 or lateral_norm <= 1e-6:
        return result

    forward_axis /= forward_norm
    lateral_axis /= lateral_norm

    height, width = trav_data.shape
    local_xy = _get_cached_local_grid_xy(rt, (height, width), cell_n, resolution)
    relative_xy = local_xy + (
        np.asarray(map_center_xy, dtype=np.float32) - np.asarray(camera_xy, dtype=np.float32)
    ).reshape(1, 2)

    trav_flat = np.asarray(trav_data, dtype=np.float32).reshape(-1)
    valid_flat = np.isfinite(trav_flat)
    hard_blocked_flat = np.asarray(blocked_mask, dtype=bool).reshape(-1) | ~valid_flat
    trav_cost_flat = 1.0 - np.clip(np.nan_to_num(trav_flat, nan=0.0), 0.0, 1.0)
    if traversable_base_cost is not None:
        normal_cost = float(np.clip(traversable_base_cost, 0.0, 1.0))
        traversable_valid_flat = valid_flat & ~hard_blocked_flat
        trav_cost_flat[traversable_valid_flat] = normal_cost
    if preferred_mask is not None:
        if preferred_mask.shape != trav_data.shape:
            raise RuntimeError(
                f"Preferred mask shape {preferred_mask.shape} does not match traversability shape {trav_data.shape}"
            )
        preferred_flat = np.asarray(preferred_mask, dtype=bool).reshape(-1)
        road_cost = float(
            np.clip(
                rt.PREFERRED_BASE_COST if preferred_base_cost is None else preferred_base_cost,
                0.0,
                1.0,
            )
        )
        trav_cost_flat[preferred_flat & ~hard_blocked_flat] = road_cost

    step_length_m = max(float(resolution), 1e-6)
    step_count = max(1, int(np.ceil(float(trajectory_length_m) / step_length_m)))
    corridor_half_width_m = 0.5 * float(trajectory_width_px) * float(resolution)
    max_angle_rad = math.radians(float(trajectory_angle_max_deg))
    max_lateral_extent_m = float(trajectory_length_m) * abs(math.sin(max_angle_rad)) + corridor_half_width_m
    forward_distance_all = (relative_xy @ forward_axis).astype(np.float32, copy=False)
    lateral_distance_all = (relative_xy @ lateral_axis).astype(np.float32, copy=False)
    candidate_domain_mask = (
        (forward_distance_all >= 0.0)
        & (forward_distance_all <= float(trajectory_length_m))
        & (np.abs(lateral_distance_all) <= max_lateral_extent_m)
    )
    candidate_domain_ids = np.flatnonzero(candidate_domain_mask)
    if candidate_domain_ids.size <= 0:
        return result

    forward_distance = forward_distance_all[candidate_domain_ids]
    lateral_distance = lateral_distance_all[candidate_domain_ids]
    hard_blocked = hard_blocked_flat[candidate_domain_ids]
    trav_cost = trav_cost_flat[candidate_domain_ids]
    angles_rad = np.deg2rad(angles_deg.astype(np.float32, copy=False))
    cos_angles = np.cos(angles_rad).astype(np.float32, copy=False)
    sin_angles = np.sin(angles_rad).astype(np.float32, copy=False)
    best_key = None
    best_safe_flat_ids = None
    best_end_xy = None

    for angle_idx, angle_deg in enumerate(angles_deg):
        cos_angle = float(cos_angles[angle_idx])
        sin_angle = float(sin_angles[angle_idx])
        direction = cos_angle * forward_axis + sin_angle * lateral_axis
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-6:
            continue
        direction /= direction_norm

        if abs(float(angle_deg)) <= 1e-6:
            progress = forward_distance
            corridor_offset = lateral_distance
        else:
            if cos_angle <= 1e-6:
                continue
            progress = forward_distance / cos_angle
            corridor_offset = lateral_distance - progress * sin_angle

        inside_candidate = (progress <= float(trajectory_length_m)) & (
            np.abs(corridor_offset) <= corridor_half_width_m
        )
        if not np.any(inside_candidate):
            continue

        inside_ids = np.flatnonzero(inside_candidate)
        progress_inside = progress[inside_ids]
        step_ids = np.floor(progress_inside / step_length_m).astype(np.int32, copy=False)
        np.minimum(step_ids, step_count - 1, out=step_ids)

        step_population = np.bincount(step_ids, minlength=step_count)
        empty_step_ids = np.flatnonzero(step_population == 0)
        first_empty_step = step_count if empty_step_ids.size <= 0 else int(empty_step_ids[0])

        blocked_inside = hard_blocked[inside_ids]
        if np.any(blocked_inside):
            blocked_step_hits = np.bincount(step_ids[blocked_inside], minlength=step_count)
            blocked_step_ids = np.flatnonzero(blocked_step_hits > 0)
            first_blocked_step = step_count if blocked_step_ids.size <= 0 else int(blocked_step_ids[0])
        else:
            first_blocked_step = step_count

        safe_step_count = min(first_empty_step, first_blocked_step)

        traversable_length_m = min(float(trajectory_length_m), safe_step_count * step_length_m)
        result["max_reachable_length_m"] = max(result["max_reachable_length_m"], traversable_length_m)
        if safe_step_count <= 0:
            continue

        safe_inside = step_ids < safe_step_count
        if not np.any(safe_inside):
            continue

        safe_local_ids = inside_ids[safe_inside]
        safe_costs = trav_cost[safe_local_ids]
        safe_pixel_count = int(safe_local_ids.size)
        if safe_pixel_count <= 0:
            continue

        mean_cost = float(np.mean(safe_costs))
        length_penalty = max(0.0, float(trajectory_length_m) - traversable_length_m) / max(
            float(trajectory_length_m), 1e-6
        )
        base_score = mean_cost + length_penalty
        unit_score = 1.0 / (float(np.sqrt(max(base_score, 0.0))) + 0.5)
        raw_score = unit_score * float(safe_pixel_count)
        if traversable_length_m <= float(trajectory_min_length_m):
            continue

        goal_angle_error_deg = None
        goal_weight = 1.0
        corrected_score = raw_score
        tie_angle_deg = float(angle_deg)
        if goal_relative_angle_deg is not None:
            goal_angle_error_deg = _normalize_angle(float(angle_deg) - float(goal_relative_angle_deg))
            goal_weight = math.sqrt(max(0.0, math.cos(math.radians(goal_angle_error_deg)))) + 1.0
            corrected_score = raw_score * goal_weight
            tie_angle_deg = goal_angle_error_deg

        result["valid_candidate_count"] += 1
        candidate_key = (corrected_score, raw_score, traversable_length_m, -abs(float(tie_angle_deg)))
        if best_key is not None and candidate_key <= best_key:
            continue

        best_key = candidate_key
        end_xy = camera_xy + direction * traversable_length_m
        result["status"] = "ok"
        result["message"] = "ok"
        result["best_angle_deg"] = float(angle_deg)
        result["best_score"] = corrected_score
        result["best_raw_score"] = raw_score
        result["best_unit_score"] = unit_score
        result["best_goal_weight"] = goal_weight
        result["best_goal_angle_error_deg"] = goal_angle_error_deg
        result["best_base_score"] = base_score
        result["best_safe_pixel_count"] = safe_pixel_count
        result["best_mean_cost"] = mean_cost
        result["best_length_penalty"] = length_penalty
        result["best_length_m"] = traversable_length_m
        best_safe_flat_ids = candidate_domain_ids[safe_local_ids]
        best_end_xy = end_xy.astype(np.float32, copy=False)

    if best_safe_flat_ids is not None:
        best_mask = np.zeros(height * width, dtype=bool)
        best_mask[best_safe_flat_ids] = True
        result["best_mask"] = best_mask.reshape(trav_data.shape)
        result["best_centerline_world_xy"] = np.stack(
            [
                np.asarray(camera_xy, dtype=np.float32),
                np.asarray(best_end_xy, dtype=np.float32),
            ],
            axis=0,
        )

    return result


class ObstacleAvoidancePlanner:
    def __init__(self, config: AvoidanceConfig, coordinator: ZedFrameCoordinator):
        self.config = config
        self.coordinator = coordinator

        self._rt: ModuleType | None = None
        self._args = config.to_namespace()

        self.elmap = None
        self.trav_postprocessor = None
        self.trav_data: np.ndarray | None = None
        self.elev_data: np.ndarray | None = None

    def setup(self) -> None:
        self._rt = _load_rt_module()

        if self._rt.cp is None or self._rt.Parameter is None or self._rt.ElevationMap is None:
            raise RuntimeError(
                "elevation_mapping_cupy_core dependencies are unavailable. "
                f"Expected root: {self._rt.DEFAULT_ELEVATION_MAPPING_ROOT}"
            )

        self.elmap = self._rt.init_elevation_map(self._args)
        map_size = int(self.elmap.cell_n) - 2
        self.trav_data = np.zeros((map_size, map_size), dtype=np.float32)
        self.elev_data = np.zeros((map_size, map_size), dtype=np.float32)
        self.trav_postprocessor = self._rt.build_traversability_postprocessor(self._args)

    def close(self) -> None:
        self.elmap = None
        self.trav_postprocessor = None
        self.trav_data = None
        self.elev_data = None

    def plan_avoidance(self, goal_relative_angle_deg: float | None = None) -> AvoidanceResult:
        if self.elmap is None:
            self.setup()

        timing_ms: dict[str, float] = {}
        plan_start = time.perf_counter()

        try:
            perf_start = time.perf_counter()
            frame = self.coordinator.request_frame(timeout_sec=max(1.0, self.config.avoidance_check_interval * 2.0))
            timing_ms["frame_wait_ms"] = (time.perf_counter() - perf_start) * 1000.0
            if frame is None:
                return self._make_result(
                    status="error",
                    recommended_angle_deg=0.0,
                    safe_distance_m=0.0,
                    max_reachable_distance_m=0.0,
                    score=0.0,
                    raw_result=None,
                    timing_ms=timing_ms,
                    error_message="Timed out waiting for a fresh ZED frame.",
                )

            perf_start = time.perf_counter()
            base_rotation, base_translation = _extract_base_pose_with_translation_offset(
                self._rt,
                frame.rotation,
                frame.translation,
                self.config.camera_pitch_deg,
                self.config.camera_offset_forward_m,
                self.config.camera_offset_left_m,
                self.config.camera_offset_up_m,
            )
            timing_ms["base_pose_ms"] = (time.perf_counter() - perf_start) * 1000.0

            perf_start = time.perf_counter()
            self.elmap.clear()
            points = self._rt.downsample_pointcloud(frame.xyz_image, self.config.downsample_stride)
            timing_ms["downsample_ms"] = (time.perf_counter() - perf_start) * 1000.0
            if int(points.shape[0]) <= 0:
                return self._make_result(
                    status="no_path",
                    recommended_angle_deg=0.0,
                    safe_distance_m=0.0,
                    max_reachable_distance_m=0.0,
                    score=0.0,
                    raw_result=None,
                    timing_ms=timing_ms,
                    error_message="Point cloud did not contain any valid samples.",
                )

            perf_start = time.perf_counter()
            base_rotation_cp = self._rt.cp.asarray(base_rotation)
            self.elmap.move_to(base_translation, base_rotation_cp)

            rotation_cp = self._rt.cp.asarray(frame.rotation)
            translation_cp = self._rt.cp.asarray(frame.translation)
            pos_noise = max(0.01, 1.0 - frame.confidence / 100.0)
            ori_noise = max(0.01, 1.0 - frame.confidence / 100.0)
            self.elmap.input_pointcloud(points, ["x", "y", "z"], rotation_cp, translation_cp, pos_noise, ori_noise)
            self.elmap.get_map_with_name_ref("traversability", self.trav_data)
            self.elmap.get_map_with_name_ref("elevation", self.elev_data)
            nav_trav_data = self.trav_data
            if self.trav_postprocessor is not None:
                nav_trav_data = self.trav_postprocessor(self.trav_data, base_rotation)
            timing_ms["elmap_ms"] = (time.perf_counter() - perf_start) * 1000.0

            map_center_xy = self._rt.cp.asnumpy(self.elmap.center[:2]).astype(np.float32, copy=False)
            camera_xy = base_translation[:2].astype(np.float32, copy=False)

            perf_start = time.perf_counter()
            near_depth_block_mask = self._rt.build_near_depth_block_mask(
                trav_data=nav_trav_data,
                base_rotation=base_rotation,
                base_translation=base_translation,
                rotation=frame.rotation,
                translation=frame.translation,
                map_center_xy=map_center_xy,
                cell_n=self.elmap.cell_n,
                resolution=self.elmap.resolution,
                xyz_image=frame.xyz_image,
                intrinsics=frame.intrinsics,
                near_distance_m=self.config.near_depth_block_distance,
            )
            timing_ms["near_block_ms"] = (time.perf_counter() - perf_start) * 1000.0

            perf_start = time.perf_counter()
            road_map = np.zeros_like(nav_trav_data, dtype=bool)
            blocked_mask, road_priority_map = self._rt.build_traversability_masks(
                trav_data=nav_trav_data,
                road_map=road_map,
                blocked_threshold=self.config.blocked_threshold,
                extra_blocked_mask=near_depth_block_mask,
            )
            forced_traversable_mask = self._rt.build_forced_traversable_mask(
                trav_shape=nav_trav_data.shape,
                camera_xy=camera_xy,
                base_rotation=base_rotation,
                map_center_xy=map_center_xy,
                cell_n=self.elmap.cell_n,
                resolution=self.elmap.resolution,
                forward_distance_m=self.config.forced_traversable_forward_distance,
                half_width_m=self.config.forced_traversable_half_width,
            )
            nav_trav_data, blocked_mask = self._rt.apply_forced_traversable_mask(
                trav_data=nav_trav_data,
                blocked_mask=blocked_mask,
                forced_mask=forced_traversable_mask,
            )
            preferred_priority_map = (road_priority_map | forced_traversable_mask) & ~blocked_mask
            timing_ms["trav_fusion_ms"] = (time.perf_counter() - perf_start) * 1000.0

            perf_start = time.perf_counter()
            trajectory_result = _plan_scored_fan_trajectory_with_goal_bias(
                self._rt,
                trav_data=nav_trav_data,
                blocked_mask=blocked_mask,
                preferred_mask=preferred_priority_map,
                camera_xy=camera_xy,
                base_rotation=base_rotation,
                map_center_xy=map_center_xy,
                cell_n=self.elmap.cell_n,
                resolution=self.elmap.resolution,
                trajectory_length_m=self.config.trajectory_length,
                trajectory_min_length_m=self.config.trajectory_min_length,
                trajectory_width_px=self.config.trajectory_width_px,
                trajectory_angle_max_deg=self.config.trajectory_angle_max_deg,
                trajectory_angle_step_deg=self.config.trajectory_angle_step_deg,
                traversable_base_cost=self.config.traversable_base_cost,
                preferred_base_cost=self.config.preferred_base_cost,
                goal_relative_angle_deg=goal_relative_angle_deg,
            )
            timing_ms["trajectory_ms"] = (time.perf_counter() - perf_start) * 1000.0
            timing_ms["plan_total_ms"] = (time.perf_counter() - plan_start) * 1000.0

            result_status = str(trajectory_result.get("status", "no_path"))
            return self._make_result(
                status=result_status,
                recommended_angle_deg=float(trajectory_result.get("best_angle_deg") or 0.0),
                safe_distance_m=float(trajectory_result.get("best_length_m") or 0.0),
                max_reachable_distance_m=float(trajectory_result.get("max_reachable_length_m") or 0.0),
                score=float(trajectory_result.get("best_score") or 0.0),
                raw_result=trajectory_result,
                timing_ms=timing_ms,
                error_message="" if result_status != "error" else str(trajectory_result.get("message", "")),
            )
        except Exception as exc:
            timing_ms["plan_total_ms"] = (time.perf_counter() - plan_start) * 1000.0
            return self._make_result(
                status="error",
                recommended_angle_deg=0.0,
                safe_distance_m=0.0,
                max_reachable_distance_m=0.0,
                score=0.0,
                raw_result=None,
                timing_ms=timing_ms,
                error_message=str(exc),
            )

    @staticmethod
    def _make_result(
        status: str,
        recommended_angle_deg: float,
        safe_distance_m: float,
        max_reachable_distance_m: float,
        score: float,
        raw_result: dict | None,
        timing_ms: dict[str, float],
        error_message: str,
    ) -> AvoidanceResult:
        return AvoidanceResult(
            status=status,
            recommended_angle_deg=float(recommended_angle_deg),
            safe_distance_m=float(safe_distance_m),
            max_reachable_distance_m=float(max_reachable_distance_m),
            score=float(score),
            raw_result=raw_result,
            timing_ms={str(key): float(value) for key, value in timing_ms.items()},
            error_message=error_message,
        )
