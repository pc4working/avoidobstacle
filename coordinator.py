from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

try:
    import pyzed.sl as sl
except ImportError:
    sl = None


R_CONVERT_ZUP_TO_XFWD = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


@dataclass(slots=True)
class FrameData:
    rotation: np.ndarray
    translation: np.ndarray
    confidence: float
    xyz_image: np.ndarray
    intrinsics: tuple[float, float, float, float]


class ZedFrameCoordinator:
    """
    Share one ZED camera between the ZED worker thread and the avoidance planner.

    `task_zed()` owns all SDK calls. The planner only requests that the next
    successful frame also include XYZ data and the pose transformed into the
    X-forward frame expected by the traversability code.
    """

    def __init__(self, camera):
        if sl is None:
            raise RuntimeError("pyzed.sl not found. Please install ZED SDK Python bindings.")

        self.camera = camera
        self.pose = sl.Pose()
        self.point_cloud_mat = sl.Mat()

        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._frame_requested = False
        self._closed = False
        self._frame_buffer: FrameData | None = None
        self._intrinsics: tuple[float, float, float, float] | None = None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._frame_requested = False
            self._frame_buffer = None
            self._frame_ready.set()

    def grab_and_get_pose(self) -> tuple[int, float, float, float, float, float, float] | None:
        err = self.camera.grab()
        if err != sl.ERROR_CODE.SUCCESS:
            return None

        tracking_state = self.camera.get_position(self.pose, sl.REFERENCE_FRAME.WORLD)
        if tracking_state != sl.POSITIONAL_TRACKING_STATE.OK:
            return None

        orientation = self.pose.get_orientation()
        qx, qy, qz, qw = (float(value) for value in orientation.get())
        translation = self.pose.get_translation()
        tx, ty = (float(value) for value in translation.get()[:2])

        should_capture = False
        with self._lock:
            should_capture = self._frame_requested and not self._closed

        if should_capture:
            transform = self.pose.pose_data().m
            rotation_zup = np.asarray(transform[:3, :3], dtype=np.float32)
            translation_zup = np.asarray(transform[:3, 3], dtype=np.float32)
            xyz_image_zup = self._extract_xyz_image()
            frame = FrameData(
                rotation=self._convert_rotation_to_xfwd(rotation_zup),
                translation=self._convert_translation_to_xfwd(translation_zup),
                confidence=float(self.pose.pose_confidence),
                xyz_image=self._convert_xyz_image_to_xfwd(xyz_image_zup),
                intrinsics=self._get_intrinsics(),
            )
            with self._lock:
                if self._frame_requested and not self._closed:
                    self._frame_buffer = frame
                    self._frame_requested = False
                    self._frame_ready.set()

        return int(tracking_state), qx, qy, qz, qw, tx, ty

    def request_frame(self, timeout_sec: float = 1.0) -> FrameData | None:
        timeout_sec = max(0.0, float(timeout_sec))
        with self._lock:
            if self._closed:
                return None
            self._frame_buffer = None
            self._frame_requested = True
            self._frame_ready.clear()

        frame_ready = self._frame_ready.wait(timeout_sec)
        with self._lock:
            if not frame_ready:
                self._frame_requested = False
                self._frame_buffer = None
                return None

            frame = self._frame_buffer
            self._frame_buffer = None
            self._frame_ready.clear()
            return frame

    def _extract_xyz_image(self) -> np.ndarray:
        self.camera.retrieve_measure(self.point_cloud_mat, sl.MEASURE.XYZ)
        xyz = self.point_cloud_mat.get_data().copy()
        return np.asarray(xyz[:, :, :3], dtype=np.float32)

    def _get_intrinsics(self) -> tuple[float, float, float, float]:
        if self._intrinsics is not None:
            return self._intrinsics

        calib = self.camera.get_camera_information().camera_configuration.calibration_parameters.left_cam
        self._intrinsics = (
            float(calib.fx),
            float(calib.fy),
            float(calib.cx),
            float(calib.cy),
        )
        return self._intrinsics

    @staticmethod
    def _convert_rotation_to_xfwd(rotation_zup: np.ndarray) -> np.ndarray:
        rotation_zup = np.asarray(rotation_zup, dtype=np.float32)
        return (
            R_CONVERT_ZUP_TO_XFWD
            @ rotation_zup
            @ R_CONVERT_ZUP_TO_XFWD.T
        ).astype(np.float32, copy=False)

    @staticmethod
    def _convert_translation_to_xfwd(translation_zup: np.ndarray) -> np.ndarray:
        translation_zup = np.asarray(translation_zup, dtype=np.float32)
        return (R_CONVERT_ZUP_TO_XFWD @ translation_zup).astype(np.float32, copy=False)

    @staticmethod
    def _convert_xyz_image_to_xfwd(xyz_image_zup: np.ndarray) -> np.ndarray:
        xyz_image_zup = np.asarray(xyz_image_zup, dtype=np.float32)
        converted = np.empty_like(xyz_image_zup, dtype=np.float32)
        converted[..., 0] = xyz_image_zup[..., 1]
        converted[..., 1] = -xyz_image_zup[..., 0]
        converted[..., 2] = xyz_image_zup[..., 2]
        return converted
