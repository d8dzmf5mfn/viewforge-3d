from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from face3d.errors import fail


def _numpy(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    if hasattr(value, "r"):
        value = value.r
    if hasattr(value, "toarray"):
        value = value.toarray()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
            fail("asset-invalid", f"模型 JSON 根对象不是 mapping: {path.name}", stage="assets")
        if path.suffix.lower() in (".npy", ".npz"):
            loaded = np.load(path, allow_pickle=True, encoding="latin1")
            if isinstance(loaded, np.lib.npyio.NpzFile):
                return {key: loaded[key] for key in loaded.files}
            value = loaded.item() if loaded.shape == () else loaded
            if isinstance(value, dict):
                return value
            fail(
                "asset-invalid",
                f"模型数组根对象不是 mapping: {path.name}",
                stage="assets",
            )
        with path.open("rb") as handle:
            value = pickle.load(handle, encoding="latin1")  # noqa: S301 - user-authorized model
        if not isinstance(value, dict):
            fail("asset-invalid", f"模型 pickle 根对象不是 mapping: {path.name}", stage="assets")
        return value
    except ModuleNotFoundError as exc:
        fail(
            "asset-invalid",
            f"模型依赖无法反序列化: {exc.name}",
            stage="assets",
            details={
                "path": str(path),
                "hint": "请使用 FLAME 2023 Open 的 Python 3/NumPy 资产，或先转换为 NPZ",
            },
        )
    except Exception as exc:
        fail(
            "asset-invalid",
            f"无法读取模型资产 {path.name}: {exc}",
            stage="assets",
            details={"path": str(path)},
        )


def _key(mapping: dict[str, Any], *names: str) -> Any:
    normalized = {
        key.decode("utf-8") if isinstance(key, bytes) else str(key): value
        for key, value in mapping.items()
    }
    for name in names:
        if name in normalized:
            return normalized[name]
    fail(
        "asset-invalid",
        f"模型缺少字段: {' / '.join(names)}",
        stage="assets",
        details={"available": sorted(normalized)},
    )


@dataclass(slots=True)
class LandmarkEmbedding:
    static_faces: np.ndarray
    static_barycentric: np.ndarray
    dynamic_faces: np.ndarray | None = None
    dynamic_barycentric: np.ndarray | None = None

    @property
    def supports_ibug68(self) -> bool:
        return self.static_faces.shape[-1] == 68 or (
            self.static_faces.shape[-1] == 51
            and self.dynamic_faces is not None
            and self.dynamic_faces.shape[-1] == 17
        )

    def indices_for_yaw(self, yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
        if self.static_faces.shape[-1] == 68:
            return self.static_faces.reshape(-1), self.static_barycentric.reshape(-1, 3)
        if (
            not self.supports_ibug68
            or self.dynamic_faces is None
            or self.dynamic_barycentric is None
        ):
            fail(
                "asset-invalid",
                "FLAME landmark embedding 必须提供 68 点，或 17 动态轮廓 + 51 静态点",
                stage="assets",
            )
        rounded = int(np.rint(yaw_deg))
        if rounded < -39:
            row = 78
        elif rounded < 0:
            row = 39 - rounded
        else:
            row = min(rounded, 39)
        dynamic_faces = self.dynamic_faces[row].reshape(-1)
        dynamic_barycentric = self.dynamic_barycentric[row].reshape(-1, 3)
        return (
            np.concatenate((dynamic_faces, self.static_faces.reshape(-1))),
            np.concatenate((dynamic_barycentric, self.static_barycentric.reshape(-1, 3))),
        )


@dataclass(slots=True)
class FlameModel:
    vertices_template: np.ndarray
    shape_directions: np.ndarray
    faces: np.ndarray
    landmarks: LandmarkEmbedding
    symmetry_pairs: np.ndarray
    symmetry_center_x: float
    joint_regressor: np.ndarray | None = None

    @classmethod
    def load(
        cls,
        model_path: Path,
        landmark_path: Path,
        shape_coefficients: int,
    ) -> FlameModel:
        model = _load_mapping(model_path)
        vertices = _numpy(_key(model, "v_template"), dtype=np.float64)
        faces = _numpy(_key(model, "f", "faces"), dtype=np.int64)
        shape_directions = _numpy(_key(model, "shapedirs"), dtype=np.float64)
        normalized_model = {
            key.decode("utf-8") if isinstance(key, bytes) else str(key): value
            for key, value in model.items()
        }
        joint_value = normalized_model.get("J_regressor")
        joint_regressor = (
            _numpy(joint_value, dtype=np.float64) if joint_value is not None else None
        )
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            fail("asset-invalid", "FLAME v_template 形状必须为 [V,3]", stage="assets")
        if faces.ndim != 2 or faces.shape[1] != 3:
            fail("asset-invalid", "FLAME faces 形状必须为 [F,3]", stage="assets")
        if shape_directions.ndim == 2 and shape_directions.shape[0] == vertices.size:
            shape_directions = shape_directions.reshape(vertices.shape[0], 3, -1)
        if shape_directions.ndim != 3 or shape_directions.shape[:2] != vertices.shape:
            fail("asset-invalid", "FLAME shapedirs 形状必须为 [V,3,K]", stage="assets")
        if shape_directions.shape[2] < shape_coefficients:
            fail(
                "asset-invalid",
                "FLAME shape basis 维度不足",
                stage="assets",
                details={
                    "available": int(shape_directions.shape[2]),
                    "requested": shape_coefficients,
                },
            )
        if joint_regressor is not None and (
            joint_regressor.ndim != 2 or joint_regressor.shape[1] != len(vertices)
        ):
            fail(
                "asset-invalid",
                "FLAME J_regressor 形状必须为 [J,V]",
                stage="assets",
                details={"shape": list(joint_regressor.shape)},
            )
        landmark_mapping = _load_mapping(landmark_path)
        embedding = cls._load_landmarks(landmark_mapping)
        if not embedding.supports_ibug68:
            fail(
                "asset-invalid",
                "landmark embedding 不支持 ibug 68 点",
                stage="assets",
                details={"staticCount": int(embedding.static_faces.shape[-1])},
            )
        center_x = float(np.median(vertices[:, 0]))
        mirrored = vertices.copy()
        mirrored[:, 0] = 2 * center_x - mirrored[:, 0]
        distances, neighbors = cKDTree(vertices).query(mirrored, k=1)
        span = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        candidates = np.flatnonzero(distances < max(span * 0.006, 1e-8))
        pairs = np.stack((candidates, neighbors[candidates]), axis=1)
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
        if len(pairs) > 1800:
            indices = np.linspace(0, len(pairs) - 1, 1800, dtype=int)
            pairs = pairs[indices]
        return cls(
            vertices_template=np.ascontiguousarray(vertices),
            shape_directions=np.ascontiguousarray(shape_directions[:, :, :shape_coefficients]),
            faces=np.ascontiguousarray(faces),
            landmarks=embedding,
            symmetry_pairs=np.ascontiguousarray(pairs),
            symmetry_center_x=center_x,
            joint_regressor=(
                np.ascontiguousarray(joint_regressor) if joint_regressor is not None else None
            ),
        )

    @staticmethod
    def _load_landmarks(mapping: dict[str, Any]) -> LandmarkEmbedding:
        keys = {
            key.decode("utf-8") if isinstance(key, bytes) else str(key): value
            for key, value in mapping.items()
        }
        if "lmk_face_idx" in keys or "lmk_faces_idx" in keys:
            faces = keys.get("lmk_face_idx", keys.get("lmk_faces_idx"))
            barycentric = keys.get("lmk_b_coords", keys.get("lmk_bary_coords"))
            if barycentric is None:
                fail("asset-invalid", "landmark embedding 缺少 barycentric 坐标", stage="assets")
            return LandmarkEmbedding(
                _numpy(faces, dtype=np.int64),
                _numpy(barycentric, dtype=np.float64),
            )
        static_faces = _numpy(
            _key(keys, "static_lmk_faces_idx", "static_lmk_face_idx"), dtype=np.int64
        )
        static_barycentric = _numpy(
            _key(keys, "static_lmk_bary_coords", "static_lmk_b_coords"), dtype=np.float64
        )
        dynamic_faces = keys.get("dynamic_lmk_faces_idx", keys.get("dynamic_lmk_face_idx"))
        dynamic_barycentric = keys.get("dynamic_lmk_bary_coords", keys.get("dynamic_lmk_b_coords"))
        return LandmarkEmbedding(
            static_faces,
            static_barycentric,
            _numpy(dynamic_faces, dtype=np.int64) if dynamic_faces is not None else None,
            _numpy(dynamic_barycentric, dtype=np.float64)
            if dynamic_barycentric is not None
            else None,
        )

    def shaped_vertices(self, coefficients: np.ndarray) -> np.ndarray:
        return self.vertices_template + np.einsum(
            "vck,k->vc", self.shape_directions, coefficients, optimize=True
        )

    def landmark_vertices(self, vertices: np.ndarray, yaw_deg: float) -> np.ndarray:
        face_indices, barycentric = self.landmarks.indices_for_yaw(yaw_deg)
        triangles = vertices[self.faces[face_indices]]
        return np.einsum("lvc,lv->lc", triangles, barycentric, optimize=True)

    def joint_centers(self, vertices: np.ndarray) -> np.ndarray:
        if self.joint_regressor is None or self.joint_regressor.shape[0] < 5:
            fail(
                "asset-invalid",
                "Face v2 需要至少包含颈、下颌和双眼关节的 FLAME J_regressor",
                stage="assets",
            )
        return np.asarray(self.joint_regressor @ np.asarray(vertices), dtype=np.float64)

    def eye_centers(self, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joints = self.joint_centers(vertices)
        eyes = np.asarray(joints[3:5], dtype=np.float64)
        order = np.argsort(eyes[:, 0])
        return eyes[order[0]], eyes[order[1]]


@dataclass(slots=True)
class FlameRegionMasks:
    left_ear: np.ndarray
    right_ear: np.ndarray
    left_eyelid: np.ndarray
    right_eyelid: np.ndarray
    neck: np.ndarray

    @classmethod
    def load(cls, path: Path, vertices: np.ndarray) -> FlameRegionMasks:
        payload = {
            (key.decode("utf-8") if isinstance(key, bytes) else str(key))
            .lower()
            .replace("-", "_"): value
            for key, value in _load_mapping(path).items()
        }
        vertex_count = len(vertices)

        def indices(*aliases: str) -> np.ndarray | None:
            for alias in aliases:
                key = alias.lower().replace("-", "_")
                if key not in payload:
                    continue
                value = np.asarray(payload[key])
                if value.dtype == bool and value.size == vertex_count:
                    result = np.flatnonzero(value.reshape(-1))
                else:
                    result = value.astype(np.int64, copy=False).reshape(-1)
                result = np.unique(result)
                if len(result) and (result[0] < 0 or result[-1] >= vertex_count):
                    fail(
                        "asset-invalid",
                        f"FLAME 区域 {alias} 包含越界顶点",
                        stage="assets",
                    )
                return result
            return None

        center_x = float(np.median(vertices[:, 0]))
        ears = indices("ears", "ear")
        left_ear = indices("left_ear", "ear_left", "leftear")
        right_ear = indices("right_ear", "ear_right", "rightear")
        if ears is not None:
            left_ear = ears[vertices[ears, 0] <= center_x] if left_ear is None else left_ear
            right_ear = ears[vertices[ears, 0] > center_x] if right_ear is None else right_ear

        eyes = indices("eye_region", "eyes", "eyelids")
        left_eyelid = indices("left_eyelid", "left_eye_region", "eye_region_left")
        right_eyelid = indices("right_eyelid", "right_eye_region", "eye_region_right")
        if eyes is not None:
            left_eyelid = (
                eyes[vertices[eyes, 0] <= center_x] if left_eyelid is None else left_eyelid
            )
            right_eyelid = (
                eyes[vertices[eyes, 0] > center_x] if right_eyelid is None else right_eyelid
            )

        missing = [
            name
            for name, value in (
                ("left_ear", left_ear),
                ("right_ear", right_ear),
                ("left_eyelid", left_eyelid),
                ("right_eyelid", right_eyelid),
            )
            if value is None or len(value) < 4
        ]
        if missing:
            fail(
                "asset-invalid",
                "FLAME 区域 masks 不完整",
                stage="assets",
                details={"missing": missing, "available": sorted(payload)},
            )
        neck = indices("neck", "neck_region")
        if neck is None:
            y_limit = float(np.quantile(vertices[:, 1], 0.12))
            neck = np.flatnonzero(vertices[:, 1] <= y_limit)
        return cls(
            left_ear=np.asarray(left_ear, dtype=np.int64),
            right_ear=np.asarray(right_ear, dtype=np.int64),
            left_eyelid=np.asarray(left_eyelid, dtype=np.int64),
            right_eyelid=np.asarray(right_eyelid, dtype=np.int64),
            neck=np.asarray(neck, dtype=np.int64),
        )

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "left_ear": self.left_ear,
            "right_ear": self.right_ear,
            "left_eyelid": self.left_eyelid,
            "right_eyelid": self.right_eyelid,
            "neck": self.neck,
        }
