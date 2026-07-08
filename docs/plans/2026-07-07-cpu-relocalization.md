# CPU 비주얼 리로컬라이징 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.
> **커밋은 항상 사용자가 직접** — 계획의 "커밋" 스텝은 사용자 몫으로 표기한다.

**Goal:** Hilti 데이터셋으로 "정확한 포즈 + 이미지 태깅 DB → 쿼리 이미지 1장 → CPU만으로 6DOF 위치 반환" 파이프라인을 완성하고 성공률을 숫자로 측정한다.

**Architecture:** build_db.py(오프라인 DB 구축)와 localize.py(쿼리 측위) 두 프로그램. 포즈 출처는 PoseProvider 인터페이스로 추상화(1단계 GT → 2단계 Cpp_SLAM 세션). 특징=XFeat, 검색=MegaLoc(폴백 CosPlace), 매칭=XFeat 동반 LighterGlue, 위치=OpenCV PnP+RANSAC.

**Tech Stack:** Python 3.10+, PyTorch(CPU), OpenCV, numpy, scipy, rosbags(순수 파이썬 bag 리더 — ROS 설치 불필요), torch.hub 모델(XFeat/MegaLoc/CosPlace).

---

## 파일 구조 (최종)

```
VSLAM_repository/
├── requirements.txt
├── config.yaml              # 토픽명/캘리브 경로/모델 선택 등 실행 설정
├── src/
│   ├── __init__.py
│   ├── extract.py           # bag → 이미지(+타임스탬프) 추출
│   ├── pose_provider.py     # PoseProvider ABC + GT(TUM) / Session 구현
│   ├── features.py          # XFeat 래퍼
│   ├── retrieval.py         # MegaLoc/CosPlace 래퍼 + 코사인 검색
│   ├── matcher.py           # XFeat LighterGlue 매칭 래퍼
│   ├── lift3d.py            # LiDAR 점 투영 → 특징점 3D 부여
│   └── pnp.py               # PnP+RANSAC 래퍼
├── build_db.py
├── localize.py
├── eval.py
├── tools/visualize_lift3d.py  # LiDAR 투영 오버레이 (lift3d 검증)
└── tests/                     # pytest
```

**모델 조달 경로 (전부 torch.hub, 첫 실행 시 자동 다운로드):**
- XFeat: `torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True)`
- 매칭: XFeat 객체의 `match_lighterglue()` (verlab이 XFeat용으로 학습한 LighterGlue)
  — 주의: 공식 cvg/LightGlue는 XFeat 디스크립터 미지원이라 **XFeat 동반 매처를 사용**
- MegaLoc: `torch.hub.load('gmberton/MegaLoc', 'get_trained_model')`
- CosPlace(경량 폴백): `torch.hub.load('gmberton/cosplace', 'get_trained_model', backbone='ResNet18', fc_output_dim=512)`

---

### Task 1: 프로젝트 골격 + 환경

**Files:** Create: `requirements.txt`, `config.yaml`, `src/__init__.py`, `tests/__init__.py`, `.gitignore`, `README.md`

- [ ] **Step 1: 파일 생성**

`requirements.txt`:
```
torch
torchvision
opencv-python
numpy
scipy
rosbags
pyyaml
pytest
tqdm
```

`config.yaml`:
```yaml
# Task 2에서 실제 bag 검사 후 값 채움
bag_path: ""            # Hilti 시퀀스 bag 경로
image_topic: ""         # 예: /alphasense/cam0/image_raw
lidar_topic: ""         # 예: /os_cloud_node/points
gt_path: ""             # GT 궤적 파일 (TUM 형식: t x y z qx qy qz qw)
calib:
  intrinsics: [0, 0, 0, 0]   # fx fy cx cy
  dist_coeffs: []            # 왜곡 계수 (모델에 맞게)
  dist_model: "radtan"       # radtan | equidistant(fisheye)
  T_cam_lidar: []            # 4x4 row-major 16개 값
retrieval_model: "megaloc"   # megaloc | cosplace
top_k: 5
keyframe_stride: 2           # 짝수 인덱스=DB, 홀수=쿼리 (eval 분할과 일치)
db_dir: "db"
```

`.gitignore`:
```
__pycache__/
*.pyc
db/
data/
.venv/
```

`README.md`: 스펙 요약 5줄 + 실행법 자리(뒤 태스크에서 채움).

- [ ] **Step 2: venv 구성 및 설치 확인**

```bash
cd /Users/deepfine/C++Project/VSLAM_repository
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch, cv2, rosbags, scipy; print('env ok')"
```
Expected: `env ok`

- [ ] **Step 3: 커밋 (사용자)**

---

### Task 2: Hilti 데이터 준비 + bag 검사 스크립트

**Files:** Create: `tools/inspect_bag.py`

- [ ] **Step 1: 데이터 다운로드 (사용자)**

Hilti SLAM Challenge 사이트(hilti-challenge.com)에서 실내 시퀀스 1개(rosbag)와
해당 캘리브 파일, GT 궤적을 `data/` 아래에 받는다. (2022 시퀀스 권장 — Alphasense
5캠 + LiDAR + IMU 동기, GT 제공 시퀀스 선택)

- [ ] **Step 2: 검사 스크립트 작성**

```python
# tools/inspect_bag.py — bag의 토픽/타입/메시지수/시간범위를 출력
import sys
from rosbags.highlevel import AnyReader
from pathlib import Path

def main(bag_path: str) -> None:
    with AnyReader([Path(bag_path)]) as reader:
        print(f"duration: {(reader.end_time - reader.start_time)/1e9:.1f}s")
        for c in reader.connections:
            print(f"{c.topic:40s} {c.msgtype:45s} {c.msgcount:7d} msgs")

if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 3: 실행해 실제 토픽명 확인 (사용자 실행, 결과 공유)**

```bash
python tools/inspect_bag.py data/<시퀀스>.bag
```
Expected: 이미지 토픽(sensor_msgs/Image 또는 CompressedImage)과 LiDAR 토픽
(sensor_msgs/PointCloud2)이 보임 → `config.yaml`의 topic/경로/캘리브 값 채움.
캘리브 파일(yaml)에서 intrinsics·T_cam_lidar를 옮겨 적는다.

- [ ] **Step 4: 커밋 (사용자)**

---

### Task 3: extract.py — bag에서 이미지+타임스탬프 추출

**Files:** Create: `src/extract.py`, `tests/test_extract.py`

- [ ] **Step 1: 실패 테스트 작성 (합성 bag을 만들어 왕복 검증)**

```python
# tests/test_extract.py
import numpy as np
from pathlib import Path
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore
from src.extract import extract_images

def _make_bag(path: Path, n: int = 3) -> None:
    ts = get_typestore(Stores.ROS1_NOETIC)
    Image = ts.types['sensor_msgs/msg/Image']
    Header = ts.types['std_msgs/msg/Header']
    Time = ts.types['builtin_interfaces/msg/Time']
    with Writer(path) as w:
        conn = w.add_connection('/cam0/image_raw', Image.__msgtype__, typestore=ts)
        for i in range(n):
            img = np.full((4, 5), i, dtype=np.uint8)
            msg = Image(
                header=Header(seq=i, stamp=Time(sec=100 + i, nanosec=0), frame_id='cam'),
                height=4, width=5, encoding='mono8', is_bigendian=0, step=5,
                data=img.reshape(-1))
            w.write(conn, (100 + i) * 10**9, ts.serialize_ros1(msg, Image.__msgtype__))

def test_extract_images(tmp_path):
    bag = tmp_path / 'tiny.bag'
    _make_bag(bag)
    out = tmp_path / 'imgs'
    entries = extract_images(str(bag), '/cam0/image_raw', str(out))
    assert len(entries) == 3
    assert abs(entries[0]['t'] - 100.0) < 1e-6
    assert Path(entries[0]['path']).exists()
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_extract.py -v`
Expected: FAIL (`extract_images` 없음)

- [ ] **Step 3: 구현**

```python
# src/extract.py — bag의 이미지 토픽을 PNG로 저장하고 [{'t','path'}] 목록 반환
from pathlib import Path
import cv2
import numpy as np
from rosbags.highlevel import AnyReader

def _decode(msg) -> np.ndarray:
    # raw Image (mono8/bgr8/rgb8) + CompressedImage 지원
    if hasattr(msg, 'format'):  # CompressedImage
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    buf = np.frombuffer(msg.data, np.uint8)
    if msg.encoding == 'mono8':
        return buf.reshape(msg.height, msg.width)
    img = buf.reshape(msg.height, msg.width, -1)
    if msg.encoding == 'rgb8':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def extract_images(bag_path: str, topic: str, out_dir: str, stride: int = 1):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    entries, idx = [], 0
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise ValueError(f'토픽 없음: {topic}')
        for conn, timestamp, raw in reader.messages(connections=conns):
            if idx % stride == 0:
                msg = reader.deserialize(raw, conn.msgtype)
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                path = out / f'{idx:06d}.png'
                cv2.imwrite(str(path), _decode(msg))
                entries.append({'t': t, 'path': str(path)})
            idx += 1
    return entries
```

- [ ] **Step 4: 통과 확인** — Run: `pytest tests/test_extract.py -v` → PASS
- [ ] **Step 5: 커밋 (사용자)**

---

### Task 4: pose_provider.py — GT 포즈 (TUM) + 타임스탬프 보간

**Files:** Create: `src/pose_provider.py`, `tests/test_pose_provider.py`

- [ ] **Step 1: 실패 테스트**

```python
# tests/test_pose_provider.py
import numpy as np
from src.pose_provider import GTPoseProvider

def test_interpolation(tmp_path):
    # t=0에 원점, t=2에 x=2 (무회전): t=1이면 x=1이어야 함
    gt = tmp_path / 'gt.txt'
    gt.write_text('0 0 0 0 0 0 0 1\n2 2 0 0 0 0 0 1\n')
    p = GTPoseProvider(str(gt))
    T = p.pose_at(1.0)              # 4x4
    assert np.allclose(T[:3, 3], [1, 0, 0], atol=1e-6)
    assert np.allclose(T[:3, :3], np.eye(3), atol=1e-6)

def test_out_of_range_returns_none(tmp_path):
    gt = tmp_path / 'gt.txt'
    gt.write_text('0 0 0 0 0 0 0 1\n2 2 0 0 0 0 0 1\n')
    p = GTPoseProvider(str(gt))
    assert p.pose_at(5.0) is None   # 범위 밖 외삽 금지 — 실패를 실패로
```

- [ ] **Step 2: 실패 확인** — `pytest tests/test_pose_provider.py -v` → FAIL

- [ ] **Step 3: 구현**

```python
# src/pose_provider.py — 포즈 출처 추상화. 1단계 GT(TUM), 2단계 Cpp_SLAM 세션.
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

class PoseProvider(ABC):
    @abstractmethod
    def pose_at(self, t: float):
        """시각 t의 T_world_cam(4x4) 반환. 커버 범위 밖이면 None."""

class GTPoseProvider(PoseProvider):
    """TUM 형식(t x y z qx qy qz qw) 궤적 파일. 회전=Slerp, 이동=선형 보간."""
    def __init__(self, path: str, max_gap: float = 0.5):
        data = np.loadtxt(path)
        self.ts = data[:, 0]
        self.xyz = data[:, 1:4]
        self.rots = Rotation.from_quat(data[:, 4:8])  # (qx,qy,qz,qw)
        self.slerp = Slerp(self.ts, self.rots)
        self.max_gap = max_gap

    def pose_at(self, t: float):
        if t < self.ts[0] or t > self.ts[-1]:
            return None
        i = int(np.searchsorted(self.ts, t))
        i = min(max(i, 1), len(self.ts) - 1)
        if self.ts[i] - self.ts[i - 1] > self.max_gap:
            return None  # GT 공백 구간 — 보간 신뢰 불가
        a = (t - self.ts[i - 1]) / (self.ts[i] - self.ts[i - 1])
        T = np.eye(4)
        T[:3, :3] = self.slerp([t]).as_matrix()[0]
        T[:3, 3] = (1 - a) * self.xyz[i - 1] + a * self.xyz[i]
        return T
```

- [ ] **Step 4: 통과 확인** → PASS
- [ ] **Step 5: 커밋 (사용자)**

---

### Task 5: pnp.py — 합성 데이터로 완전 검증 (파이프라인의 수학 코어)

**Files:** Create: `src/pnp.py`, `tests/test_pnp.py`

- [ ] **Step 1: 실패 테스트 (알려진 포즈로 투영 → 복원 일치 확인)**

```python
# tests/test_pnp.py
import numpy as np
import cv2
from src.pnp import solve_pnp

def test_recovers_known_pose():
    rng = np.random.default_rng(0)
    pts3d = rng.uniform([-2, -2, 4], [2, 2, 8], (50, 3))     # 카메라 앞 3D 점
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], float)
    rvec_true = np.array([0.1, -0.2, 0.05])
    tvec_true = np.array([0.3, -0.1, 0.5])
    pts2d, _ = cv2.projectPoints(pts3d, rvec_true, tvec_true, K, None)
    res = solve_pnp(pts2d.reshape(-1, 2), pts3d, K, dist=None)
    assert res is not None
    T = res['T_cam_world']
    R_true, _ = cv2.Rodrigues(rvec_true)
    assert np.allclose(T[:3, :3], R_true, atol=1e-4)
    assert np.allclose(T[:3, 3], tvec_true, atol=1e-3)
    assert res['inliers'] >= 45

def test_too_few_points_returns_none():
    K = np.eye(3)
    assert solve_pnp(np.zeros((3, 2)), np.zeros((3, 3)), K, None) is None
```

- [ ] **Step 2: 실패 확인** → FAIL

- [ ] **Step 3: 구현**

```python
# src/pnp.py — 2D-3D 대응 → 6DOF. 실패는 None (틀린 pose 자신있게 반환 금지)
import cv2
import numpy as np

MIN_POINTS = 6
MIN_INLIERS = 10

def solve_pnp(pts2d, pts3d, K, dist, min_inliers: int = MIN_INLIERS):
    if len(pts2d) < max(MIN_POINTS, 4):
        return None
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        np.asarray(pts3d, np.float64), np.asarray(pts2d, np.float64),
        np.asarray(K, np.float64), dist,
        iterationsCount=1000, reprojectionError=4.0, confidence=0.999,
        flags=cv2.SOLVEPNP_SQPNP)
    if not ok or inl is None or len(inl) < min_inliers:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return {'T_cam_world': T, 'inliers': int(len(inl)),
            'inlier_idx': inl.ravel()}
```

(주: `T_cam_world`는 "월드 점 → 카메라 좌표" 변환. 카메라의 월드 위치는
`T_world_cam = inv(T_cam_world)` — localize.py에서 뒤집어 반환.)

- [ ] **Step 4: 통과 확인** → PASS
- [ ] **Step 5: 커밋 (사용자)**

---

### Task 6: lift3d.py — LiDAR 투영으로 특징점에 3D 부여

**Files:** Create: `src/lift3d.py`, `tests/test_lift3d.py`, `tools/visualize_lift3d.py`

- [ ] **Step 1: 실패 테스트 (합성 — 정면 평면 점들)**

```python
# tests/test_lift3d.py
import numpy as np
from src.lift3d import project_lidar, assign_depth

K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], float)

def test_project_center():
    # 카메라 정면 5m 점 → 주점(50,50)에 투영
    pts_cam = np.array([[0, 0, 5.0]])
    uv, z, idx = project_lidar(pts_cam, K, (100, 100))
    assert np.allclose(uv[0], [50, 50], atol=1e-6) and z[0] == 5.0

def test_assign_depth_nearest():
    # 투영점 (50,50,5m)가 있을 때, 키포인트 (51,50)은 그 3D를 받아야 함
    pts_cam = np.array([[0, 0, 5.0]])
    kpts = np.array([[51.0, 50.0], [90.0, 90.0]])
    p3d, valid = assign_depth(kpts, pts_cam, K, (100, 100), radius=3.0)
    assert valid[0] and not valid[1]          # 두 번째는 근처에 라이다 점 없음
    assert np.allclose(p3d[0], [0, 0, 5.0], atol=1e-6)
```

- [ ] **Step 2: 실패 확인** → FAIL

- [ ] **Step 3: 구현**

```python
# src/lift3d.py — 카메라좌표계 LiDAR 점을 이미지에 투영, 키포인트에 최근접 3D 부여
import numpy as np
from scipy.spatial import cKDTree

def project_lidar(pts_cam, K, img_hw):
    """pts_cam(N,3, 카메라좌표) → 화면 안(z>0) 점의 (uv, z, 원본 idx)"""
    z = pts_cam[:, 2]
    front = z > 0.1
    p = pts_cam[front]
    uv = (K @ (p / p[:, 2:3]).T).T[:, :2]
    h, w = img_hw
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    idx = np.flatnonzero(front)[inside]
    return uv[inside], p[inside, 2], idx

def assign_depth(kpts, pts_cam, K, img_hw, radius: float = 4.0):
    """각 키포인트에 반경 radius(픽셀) 내 최근접 LiDAR 투영점의 3D(카메라좌표)를 부여.
    반환: (p3d_cam (M,3), valid (M,) bool)"""
    p3d = np.zeros((len(kpts), 3))
    valid = np.zeros(len(kpts), bool)
    uv, _, idx = project_lidar(pts_cam, K, img_hw)
    if len(uv) == 0:
        return p3d, valid
    tree = cKDTree(uv)
    dist, j = tree.query(kpts, k=1)
    ok = dist <= radius
    p3d[ok] = pts_cam[idx[j[ok]]]
    valid[ok] = True
    return p3d, valid
```

- [ ] **Step 4: 통과 확인** → PASS

- [ ] **Step 5: 시각화 도구 (실데이터 검증용 — M2 게이트)**

```python
# tools/visualize_lift3d.py — 이미지 위에 LiDAR 투영점을 깊이 색으로 오버레이.
# 벽·기둥 윤곽과 점이 일치하면 캘리브/좌표계가 맞는 것. 어긋나면 T_cam_lidar 재확인.
import sys, yaml
import cv2
import numpy as np
from src.lift3d import project_lidar

def main(img_path, lidar_npy, cfg_path='config.yaml'):
    cfg = yaml.safe_load(open(cfg_path))
    fx, fy, cx, cy = cfg['calib']['intrinsics']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    T = np.array(cfg['calib']['T_cam_lidar'], float).reshape(4, 4)
    img = cv2.imread(img_path)
    pts_l = np.load(lidar_npy)                       # (N,3) LiDAR 좌표
    pts_c = (T[:3, :3] @ pts_l.T).T + T[:3, 3]
    uv, z, _ = project_lidar(pts_c, K, img.shape[:2])
    zn = np.clip((z - 1) / 20, 0, 1)
    for (u, v), d in zip(uv.astype(int), zn):
        cv2.circle(img, (u, v), 1, (int(255 * (1 - d)), 64, int(255 * d)), -1)
    cv2.imwrite('lift3d_overlay.png', img)
    print('saved lift3d_overlay.png — 구조 윤곽과 점이 겹치는지 육안 확인')

if __name__ == '__main__':
    main(*sys.argv[1:])
```

- [ ] **Step 6: 커밋 (사용자)**

---

### Task 7: features.py / retrieval.py / matcher.py — 모델 래퍼

**Files:** Create: `src/features.py`, `src/retrieval.py`, `src/matcher.py`, `tests/test_models_smoke.py`

- [ ] **Step 1: 구현 (래퍼 3종 — 모델 다운로드가 필요하므로 테스트는 스모크로)**

```python
# src/features.py — XFeat 특징 추출 (CPU)
import numpy as np
import torch

class XFeatExtractor:
    def __init__(self, top_k: int = 2048):
        self.model = torch.hub.load('verlab/accelerated_features', 'XFeat',
                                    pretrained=True, top_k=top_k)
    def extract(self, img_bgr: np.ndarray) -> dict:
        out = self.model.detectAndCompute(
            torch.from_numpy(img_bgr).permute(2, 0, 1).float()[None] / 255.0,
            top_k=None)[0]
        return {'kpts': out['keypoints'].cpu().numpy(),
                'desc': out['descriptors'].cpu().numpy()}
```

```python
# src/retrieval.py — 전역 디스크립터 + 코사인 유사도 검색
import numpy as np
import torch
import torchvision.transforms.functional as TF

class Retriever:
    def __init__(self, model_name: str = 'megaloc'):
        if model_name == 'megaloc':
            self.model = torch.hub.load('gmberton/MegaLoc', 'get_trained_model')
        else:  # cosplace — CPU 경량 폴백
            self.model = torch.hub.load('gmberton/cosplace', 'get_trained_model',
                                        backbone='ResNet18', fc_output_dim=512)
        self.model.eval()

    @torch.no_grad()
    def describe(self, img_bgr: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(img_bgr[..., ::-1].copy()).permute(2, 0, 1).float() / 255
        t = TF.resize(t, [322, 322], antialias=True)[None]
        t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        d = self.model(t)[0].cpu().numpy()
        return d / (np.linalg.norm(d) + 1e-9)

def search(query_vec: np.ndarray, db_vecs: np.ndarray, k: int):
    """정규화된 벡터의 내적 = 코사인 유사도. 상위 k 인덱스."""
    sims = db_vecs @ query_vec
    order = np.argsort(-sims)[:k]
    return order, sims[order]
```

```python
# src/matcher.py — XFeat 동반 LighterGlue 매칭
import numpy as np
import torch

class Matcher:
    def __init__(self, xfeat_model):
        self.xfeat = xfeat_model    # features.XFeatExtractor.model 재사용

    def match(self, featsA: dict, featsB: dict):
        """반환: A/B 매칭 인덱스 (idxA, idxB)"""
        dA = {'keypoints': torch.from_numpy(featsA['kpts'])[None],
              'descriptors': torch.from_numpy(featsA['desc'])[None]}
        dB = {'keypoints': torch.from_numpy(featsB['kpts'])[None],
              'descriptors': torch.from_numpy(featsB['desc'])[None]}
        idx = self.xfeat.match_lighterglue(dA, dB)   # (M,2)
        idx = idx.cpu().numpy() if torch.is_tensor(idx) else np.asarray(idx)
        return idx[:, 0], idx[:, 1]
```

(주: verlab API의 정확한 시그니처는 첫 실행에서 확인 후 이 래퍼 내부만 조정.
래퍼 밖 인터페이스(`extract/describe/match`)는 고정 — 그래서 래퍼를 두는 것.)

- [ ] **Step 2: 스모크 테스트**

```python
# tests/test_models_smoke.py — 모델 다운로드 필요. 오프라인이면 skip.
import numpy as np
import pytest

@pytest.mark.slow
def test_xfeat_and_matcher_smoke():
    torch = pytest.importorskip('torch')
    from src.features import XFeatExtractor
    from src.matcher import Matcher
    rng = np.random.default_rng(0)
    img = (rng.uniform(0, 255, (240, 320, 3))).astype('uint8')
    ex = XFeatExtractor(top_k=512)
    f1 = ex.extract(img)
    assert f1['kpts'].shape[1] == 2 and len(f1['kpts']) > 0
    m = Matcher(ex.model)
    ia, ib = m.match(f1, f1)          # 자기 자신과 매칭 → 다수 일치
    assert len(ia) > 50

@pytest.mark.slow
def test_retrieval_smoke():
    from src.retrieval import Retriever, search
    import numpy as np
    r = Retriever('cosplace')          # 스모크는 경량 모델로
    img = np.zeros((240, 320, 3), np.uint8)
    v = r.describe(img)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-3
    idx, sims = search(v, np.stack([v, -v]), k=2)
    assert idx[0] == 0
```

- [ ] **Step 3: 실행** — `pytest tests/test_models_smoke.py -v -m slow`
Expected: PASS (첫 실행은 모델 다운로드 시간 소요). API 시그니처 다르면 래퍼 내부 수정.
- [ ] **Step 4: 커밋 (사용자)**

---

### Task 8: build_db.py — DB 구축 조립

**Files:** Create: `build_db.py`

- [ ] **Step 1: 구현**

```python
# build_db.py — bag → (짝수 프레임) 특징/검색벡터/3D태깅 DB
# 사용: python build_db.py config.yaml
import sys, json
from pathlib import Path
import numpy as np
import yaml
from tqdm import tqdm
from rosbags.highlevel import AnyReader
from src.extract import extract_images
from src.pose_provider import GTPoseProvider
from src.features import XFeatExtractor
from src.retrieval import Retriever
from src.lift3d import assign_depth
import cv2

def load_lidar_scans(bag_path, topic):
    """LiDAR 스캔을 (t, (N,3) xyz) 목록으로 로드 (PointCloud2 x,y,z float32 가정)"""
    scans = []
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            step = msg.point_step
            buf = np.frombuffer(msg.data, np.uint8).reshape(-1, step)
            xyz = buf[:, :12].copy().view(np.float32).reshape(-1, 3)
            xyz = xyz[np.isfinite(xyz).all(1)]
            scans.append((t, xyz))
    return scans

def nearest_scan(scans, t):
    ts = np.array([s[0] for s in scans])
    i = int(np.argmin(np.abs(ts - t)))
    return scans[i] if abs(ts[i] - t) < 0.15 else (None, None)

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    fx, fy, cx, cy = cfg['calib']['intrinsics']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    T_cl = np.array(cfg['calib']['T_cam_lidar'], float).reshape(4, 4)
    db = Path(cfg['db_dir']); (db / 'frames').mkdir(parents=True, exist_ok=True)

    entries = extract_images(cfg['bag_path'], cfg['image_topic'], str(db / 'images'))
    poses = GTPoseProvider(cfg['gt_path'])
    scans = load_lidar_scans(cfg['bag_path'], cfg['lidar_topic'])
    ex = XFeatExtractor(); ret = Retriever(cfg['retrieval_model'])

    meta, vecs = [], []
    for i, e in enumerate(tqdm(entries)):
        if i % cfg['keyframe_stride'] != 0:   # 짝수=DB (홀수는 eval 쿼리)
            continue
        T_wc = poses.pose_at(e['t'])
        st, pts_l = nearest_scan(scans, e['t'])
        if T_wc is None or pts_l is None:
            continue
        img = cv2.imread(e['path'])
        feats = ex.extract(img)
        pts_cam = (T_cl[:3, :3] @ pts_l.T).T + T_cl[:3, 3]
        p3d_cam, valid = assign_depth(feats['kpts'], pts_cam, K, img.shape[:2])
        if valid.sum() < 30:
            continue
        # 카메라좌표 3D → 월드좌표 (T_wc = T_world_cam)
        p3d_w = (T_wc[:3, :3] @ p3d_cam[valid].T).T + T_wc[:3, 3]
        np.savez(db / 'frames' / f'{i:06d}.npz',
                 kpts=feats['kpts'][valid], desc=feats['desc'][valid],
                 pts3d_w=p3d_w, T_wc=T_wc)
        vecs.append(ret.describe(img))
        meta.append({'idx': i, 't': e['t'], 'img': e['path'],
                     'n3d': int(valid.sum())})
    np.save(db / 'retrieval.npy', np.stack(vecs))
    json.dump(meta, open(db / 'meta.json', 'w'), indent=1)
    print(f'DB 완성: {len(meta)} 키프레임')

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'config.yaml')
```

- [ ] **Step 2: 실데이터 실행 (사용자)**

```bash
python build_db.py config.yaml
```
Expected: `DB 완성: N 키프레임` (N > 100 기대). 실패 시 keyframes 단계별 원인
(포즈 None / 스캔 없음 / valid<30) 카운트를 보고 진단.

- [ ] **Step 3: lift3d 육안 검증 (M2 게이트, 사용자)**

```bash
# 아무 키프레임 하나의 LiDAR를 npy로 저장하도록 build_db에서 임시 덤프하거나
# tools/visualize_lift3d.py로 오버레이 생성 → 벽 윤곽과 점이 겹치는지 확인
python tools/visualize_lift3d.py db/images/000000.png <scan.npy>
```
Expected: 구조 윤곽과 투영점 일치. 어긋나면 T_cam_lidar 부호/역행렬 여부 재확인.

- [ ] **Step 4: 커밋 (사용자)**

---

### Task 9: localize.py — 쿼리 측위 조립

**Files:** Create: `localize.py`

- [ ] **Step 1: 구현**

```python
# localize.py — 쿼리 이미지 1장 → 6DOF pose (실패 시 명시적 실패)
# 사용: python localize.py <query.png> [config.yaml]
import sys, json
from pathlib import Path
import numpy as np
import yaml, cv2
from src.features import XFeatExtractor
from src.retrieval import Retriever, search
from src.matcher import Matcher
from src.pnp import solve_pnp

def localize(img_bgr, cfg, ex=None, ret=None):
    db = Path(cfg['db_dir'])
    meta = json.load(open(db / 'meta.json'))
    vecs = np.load(db / 'retrieval.npy')
    ex = ex or XFeatExtractor(); ret = ret or Retriever(cfg['retrieval_model'])
    matcher = Matcher(ex.model)
    fx, fy, cx, cy = cfg['calib']['intrinsics']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    q_feats = ex.extract(img_bgr)
    q_vec = ret.describe(img_bgr)
    cand, sims = search(q_vec, vecs, cfg['top_k'])

    best = None
    for ci in cand:
        f = np.load(db / 'frames' / f"{meta[ci]['idx']:06d}.npz")
        qi, di = matcher.match(q_feats, {'kpts': f['kpts'], 'desc': f['desc']})
        if len(qi) < 15:
            continue
        res = solve_pnp(q_feats['kpts'][qi], f['pts3d_w'][di], K, None)
        if res and (best is None or res['inliers'] > best['inliers']):
            best = {**res, 'ref_idx': int(meta[ci]['idx'])}
    if best is None:
        return None
    T_wc = np.linalg.inv(best['T_cam_world'])   # 카메라의 월드 pose
    return {'T_world_cam': T_wc, 'inliers': best['inliers'],
            'ref_idx': best['ref_idx']}

if __name__ == '__main__':
    cfg = yaml.safe_load(open(sys.argv[2] if len(sys.argv) > 2 else 'config.yaml'))
    img = cv2.imread(sys.argv[1])
    out = localize(img, cfg)
    if out is None:
        print('측위 실패 (후보/매칭/inlier 부족)'); sys.exit(1)
    t = out['T_world_cam'][:3, 3]
    print(f"pose: x={t[0]:.3f} y={t[1]:.3f} z={t[2]:.3f}  "
          f"inliers={out['inliers']}  ref={out['ref_idx']}")
```

- [ ] **Step 2: 실데이터 1장 실행 (사용자, M3 게이트)**

```bash
python localize.py db/images/000001.png   # 홀수 프레임(쿼리용) 하나
```
Expected: pose 출력 + inliers ≥ 10. GT와 대략 일치하는 위치인지 육안 확인.
- [ ] **Step 3: 커밋 (사용자)**

---

### Task 10: eval.py — 성공률 자동 측정 (M4 게이트)

**Files:** Create: `eval.py`

- [ ] **Step 1: 구현**

```python
# eval.py — 홀수 프레임 전체를 쿼리로 → 성공률/오차/시간 집계
# 성공 기준: 위치 오차 < 0.25m AND 회전 오차 < 5도
import sys, json, time
from pathlib import Path
import numpy as np
import yaml, cv2
from scipy.spatial.transform import Rotation
from src.pose_provider import GTPoseProvider
from src.features import XFeatExtractor
from src.retrieval import Retriever
from localize import localize
from src.extract import extract_images  # 이미지 목록 재사용 위해 meta 재활용 가능

def rot_err_deg(Ra, Rb):
    return np.degrees(np.abs(Rotation.from_matrix(Ra.T @ Rb).magnitude()))

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    db = Path(cfg['db_dir'])
    entries = json.load(open(db / 'images_index.json'))  # build_db가 저장(아래 주)
    poses = GTPoseProvider(cfg['gt_path'])
    ex = XFeatExtractor(); ret = Retriever(cfg['retrieval_model'])

    n, ok, fails, terrs, rerrs, times = 0, 0, 0, [], [], []
    for i, e in enumerate(entries):
        if i % cfg['keyframe_stride'] == 0:   # 짝수=DB에 들어감 → 스킵
            continue
        T_gt = poses.pose_at(e['t'])
        if T_gt is None:
            continue
        n += 1
        t0 = time.time()
        out = localize(cv2.imread(e['path']), cfg, ex, ret)
        times.append(time.time() - t0)
        if out is None:
            fails += 1; continue
        te = np.linalg.norm(out['T_world_cam'][:3, 3] - T_gt[:3, 3])
        re = rot_err_deg(out['T_world_cam'][:3, :3], T_gt[:3, :3])
        terrs.append(te); rerrs.append(re)
        if te < 0.25 and re < 5.0:
            ok += 1
    print(f"쿼리 {n}  성공 {ok} ({100*ok/max(n,1):.1f}%)  실패반환 {fails}")
    if terrs:
        print(f"위치오차 중앙값 {np.median(terrs):.3f}m  회전 {np.median(rerrs):.2f}°")
    print(f"쿼리당 시간 중앙값 {np.median(times):.2f}s (CPU)")

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'config.yaml')
```

(주: build_db.py Step 1 구현에 `json.dump(entries, open(db/'images_index.json','w'))`
한 줄을 추가해 전체 이미지 목록을 저장한다 — eval이 재추출 없이 재사용.)

- [ ] **Step 2: 실행 (사용자, M4 게이트)**

```bash
python eval.py config.yaml
```
Expected: 성공률/오차/시간 출력. **이 숫자가 1단계 완성의 증거.**
(참고 기대치: 같은 시퀀스 홀짝 분할이라 성공률이 높게 나오는 게 정상 — 여기서
낮으면 파이프라인 버그. 조명/시점 강건성 평가는 다른 시퀀스 교차 평가로 후속.)

- [ ] **Step 3: 커밋 (사용자)**

---

### Task 11 (M5, 2단계): SessionPoseProvider — Cpp_SLAM 연동

**Files:** Modify: `/Users/deepfine/C++Project/Cpp_SLAM/src/main.cpp` (TUM 포즈 내보내기), Create: `src/pose_provider.py`에 클래스 추가, `tests/test_session_provider.py`

- [ ] **Step 1: Cpp_SLAM에 전 프레임 TUM 궤적 내보내기 추가**

main.cpp의 최종 저장부(finalPoses 확보 지점)에 추가:

```cpp
// output/slam_poses_tum.txt — t x y z qx qy qz qw (리로컬라이징 연동용)
{
    std::ofstream tum(outputDir + "slam_poses_tum.txt");
    tum << std::setprecision(9) << std::fixed;
    for (size_t i = 0; i < finalPoses.size() && i < frameTimestamps.size(); ++i)
    {
        const auto& P = finalPoses[i];
        // 회전행렬 → 쿼터니언 (GTSAM Rot3 사용)
        gtsam::Rot3 R(P.R[0],P.R[1],P.R[2], P.R[3],P.R[4],P.R[5], P.R[6],P.R[7],P.R[8]);
        auto q = R.toQuaternion();  // w,x,y,z
        tum << frameTimestamps[i] << " " << P.t[0] << " " << P.t[1] << " " << P.t[2]
            << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }
}
```
(전제: 프레임별 타임스탬프 벡터 `frameTimestamps`를 메인 루프에서 수집 —
BagParser가 프레임 시각을 이미 알고 있으므로 push_back 한 줄.)

- [ ] **Step 2: SessionPoseProvider (TUM 파일이므로 GTPoseProvider 재사용)**

```python
# src/pose_provider.py에 추가
class SessionPoseProvider(GTPoseProvider):
    """Cpp_SLAM이 내보낸 slam_poses_tum.txt — 형식이 TUM과 동일하므로 상속으로 끝.
    (형식을 TUM으로 맞춘 이유가 바로 이 재사용)"""
    pass
```

- [ ] **Step 3: 비교 실험 (사용자)**

같은 Hilti bag을 (a) GT 포즈로 DB 구축 → eval, (b) Cpp_SLAM 포즈로 DB 구축 → eval.
두 성공률·오차를 비교 → "내 SLAM 포즈의 품질이 측위에 주는 영향"을 숫자로 확보.
(주의: Cpp_SLAM 좌표계와 GT 좌표계는 원점이 다름 — eval은 GT 기준이므로 (b)는
Cpp_SLAM 궤적을 GT에 정렬(Umeyama)한 뒤 평가하거나, 자기일관성(DB pose 대비
반환 pose 오차)으로 평가. 구현 시 후자 우선 — 정렬 이슈 분리.)

- [ ] **Step 4: 커밋 (사용자)**

---

## Self-Review 결과

- 스펙 커버리지: 스택/데이터/아키텍처/검증/마일스톤(M1=T2-4, M2=T6-8, M3=T9, M4=T10, M5=T11) 모두 매핑 ✓
- 실패 처리: pnp/localize/pose_at 모두 명시적 None/실패 반환 ✓
- 타입 일관성: PoseProvider.pose_at→4x4 or None / features dict {kpts,desc} / matcher→(idxA,idxB) 통일 ✓
- 알려진 유연 지점: torch.hub API 시그니처(Task 7 주석), Hilti 토픽·캘리브(Task 2에서 확정) — 래퍼/설정으로 격리됨 ✓
