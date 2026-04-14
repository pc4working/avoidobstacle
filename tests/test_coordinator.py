from __future__ import annotations

import math
import numpy as np

from avoidobstacle.coordinator import R_CONVERT_ZUP_TO_XFWD, ZedFrameCoordinator
from avoidobstacle.planner import _load_rt_module, _plan_scored_fan_trajectory_with_goal_bias


def test_convert_translation_to_xfwd():
    translation_zup = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    converted = ZedFrameCoordinator._convert_translation_to_xfwd(translation_zup)
    np.testing.assert_allclose(converted, np.array([2.0, -1.0, 3.0], dtype=np.float32))


def test_convert_xyz_image_to_xfwd():
    xyz_image_zup = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ],
        dtype=np.float32,
    )
    converted = ZedFrameCoordinator._convert_xyz_image_to_xfwd(xyz_image_zup)
    expected = np.array(
        [
            [[2.0, -1.0, 3.0], [5.0, -4.0, 6.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(converted, expected)


def test_convert_rotation_to_xfwd_is_consistent_with_basis_change():
    rotation_zup = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    vector_zup = np.array([0.2, -0.4, 0.6], dtype=np.float32)

    converted_rotation = ZedFrameCoordinator._convert_rotation_to_xfwd(rotation_zup)
    converted_vector = R_CONVERT_ZUP_TO_XFWD @ vector_zup

    expected = R_CONVERT_ZUP_TO_XFWD @ (rotation_zup @ vector_zup)
    actual = converted_rotation @ converted_vector
    np.testing.assert_allclose(actual, expected)


def test_plan_scored_fan_trajectory_prefers_open_goal_biased_corridor():
    rt = _load_rt_module()
    shape = (40, 40)
    cell_n = 42
    resolution = 0.1

    trav_data = np.ones(shape, dtype=np.float32)
    rows, cols = np.indices(shape, dtype=np.int32)
    vis_indices = np.stack([rows, cols], axis=-1).reshape(-1, 2)
    world_xy = rt.vis_indices_to_world_xy(
        vis_indices,
        np.zeros(2, dtype=np.float32),
        cell_n,
        resolution,
    ).astype(np.float32)

    target_angle_deg = 30.0
    target_angle_rad = math.radians(target_angle_deg)
    progress = world_xy[:, 0] / math.cos(target_angle_rad)
    offset = world_xy[:, 1] - progress * math.sin(target_angle_rad)
    open_mask = (
        (progress >= 0.0)
        & (progress <= 1.5)
        & (np.abs(offset) <= 0.2)
    )
    blocked_mask = (~open_mask).reshape(shape)

    result = _plan_scored_fan_trajectory_with_goal_bias(
        rt,
        trav_data=trav_data,
        blocked_mask=blocked_mask,
        preferred_mask=None,
        camera_xy=np.array([0.0, 0.0], dtype=np.float32),
        base_rotation=np.eye(3, dtype=np.float32),
        map_center_xy=np.array([0.0, 0.0], dtype=np.float32),
        cell_n=cell_n,
        resolution=resolution,
        trajectory_length_m=1.5,
        trajectory_min_length_m=0.3,
        trajectory_width_px=4,
        trajectory_angle_max_deg=45.0,
        trajectory_angle_step_deg=15.0,
        goal_relative_angle_deg=target_angle_deg,
    )

    assert result["status"] == "ok"
    assert result["best_angle_deg"] == target_angle_deg
    assert result["best_goal_angle_error_deg"] == 0.0
