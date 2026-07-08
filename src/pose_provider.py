# 포즈 출처 추상화 — 1단계 GT(TUM), 2단계 LiDAR SLAM 세션(TUM 동일 형식이라 상속).
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class PoseProvider(ABC):
    @abstractmethod
    def pose_at(self, t: float):
        """시각 t의 T_world_cam(4x4) 반환. 커버 범위 밖/공백 구간이면 None."""


class GTPoseProvider(PoseProvider):
    """TUM 형식(t x y z qx qy qz qw) 궤적. 회전=Slerp, 이동=선형 보간.

    범위 밖 외삽은 하지 않는다(None) — 틀린 포즈를 자신 있게 주는 것이 최악.
    """

    def __init__(self, path: str, max_gap: float = 0.5):
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data[None, :]
        self.ts = data[:, 0]
        self.xyz = data[:, 1:4]
        self.rots = Rotation.from_quat(data[:, 4:8])  # (qx,qy,qz,qw)
        self.slerp = Slerp(self.ts, self.rots) if len(self.ts) > 1 else None
        self.max_gap = max_gap

    def pose_at(self, t: float):
        if t < self.ts[0] or t > self.ts[-1]:
            return None
        if self.slerp is None:  # 포즈 1개뿐
            T = np.eye(4)
            T[:3, :3] = self.rots.as_matrix()[0]
            T[:3, 3] = self.xyz[0]
            return T
        i = int(np.searchsorted(self.ts, t))
        i = min(max(i, 1), len(self.ts) - 1)
        if self.ts[i] - self.ts[i - 1] > self.max_gap:
            return None  # GT 공백 — 보간 신뢰 불가
        a = (t - self.ts[i - 1]) / (self.ts[i] - self.ts[i - 1])
        T = np.eye(4)
        T[:3, :3] = self.slerp([t]).as_matrix()[0]
        T[:3, 3] = (1 - a) * self.xyz[i - 1] + a * self.xyz[i]
        return T


class SessionPoseProvider(GTPoseProvider):
    """LiDAR SLAM(Cpp_SLAM)이 내보낸 slam_poses_tum.txt.

    형식을 TUM으로 통일했기 때문에 상속만으로 끝난다 — PoseProvider 추상화의 목적.
    """
