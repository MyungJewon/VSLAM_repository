# CPU 비주얼 리로컬라이징 설계

작성일: 2026-07-07

## 목표

라이다(또는 GT)가 만든 정확한 포즈에 이미지를 태깅해 DB를 만들고, **쿼리 이미지 한 장이
오면 CPU만으로 6DOF 위치를 반환**한다.

배경: 모노 카메라 단독 SLAM은 스케일 드리프트·환경 변화에 취약(잘 알려진 한계). 정확한
기하는 라이다가 제공하고 카메라는 "위치를 알아보는 얼굴 사진" 역할만 하는 분업 구조.
카메라만으로 3D를 역산(SFM/MVS)할 필요가 없어지므로 CUDA 필수 부품이 사라져
**CPU-only가 자연스러운 아키텍처**가 된다.

레퍼런스: 일반적인 비주얼 SFM 파이프라인(hloc + COLMAP 계열) 구조를 참고하되,
본 프로젝트는 처음부터 새로 작성한다.

## 비목표 (이번 스코프에서 제외)

- 모드 B: RGB-only SFM 매핑 (COLMAP CPU + OpenMVS) — 향후 확장
- 실시간 스트리밍 측위 (지금은 요청당 1회 배치 처리)
- GPU 경로, 동적객체 필터링, 다조건(낮/밤) DB

## 기술 스택 (CPU-only)

| 슬롯 | 선택 | 이유 |
|---|---|---|
| 로컬 특징점 | **XFeat** | CPU 실시간 목표로 설계된 모델, 자유 라이선스 (SuperPoint는 비상업 제한+무거움) |
| 전역 검색 | **MegaLoc** (대안 SALAD/CosPlace) | NetVLAD(2016) 후속 세대, 조명·계절 변화 강건성↑ |
| 매칭 | **LightGlue** | 경량 설계 + Apache 2.0. XFeat와 조합 가능 |
| 위치 계산 | **PnP+RANSAC** (OpenCV) | 신경망 아님, CPU 네이티브. COLMAP 불필요(3D는 라이다 제공) |
| 추론 런타임 | PyTorch CPU (필요시 ONNX Runtime) | 프로토 단계 우선순위 = 조립 속도 |
| 언어 | Python (프로토) | 부품 생태계가 Python. 검증 후 병목만 C++화 |

## 데이터

- **Hilti-Trimble SLAM Challenge 2026**: 앞/뒤 200° 피시아이 2대 + IMU(1kHz), ROS2 bag.
  GT 궤적은 TUM 형식(LiDAR 기반 제작 = metric·고정밀), Kalibr 캘리브 제공.
  **주의: LiDAR는 GT 제작에만 쓰였고 배포 bag에는 없음.**
- 검증 분할: 한 시퀀스의 프레임을 짝수(DB용)/홀수(쿼리용)로 분할 → 쿼리의 정답 포즈 확보.

### 3D 태깅 방식 (플랜 C — 시간적 삼각측량)

배포 데이터에 LiDAR가 없으므로 lift3d를 "LiDAR 투영"이 아니라 **GT 포즈 기반 두-뷰
삼각측량**으로 구현한다: 포즈를 아는(GT) 인접 키프레임 두 장에서 같은 특징점을 매칭하면
광선 교차로 3D가 직접 나온다. SFM처럼 포즈를 추정할 필요가 없고(GT가 제공), GT가
LiDAR 기반이라 **스케일이 metric**이다 — "정확한 포즈는 외부에서, 카메라는 인식용"이라는
설계 철학은 그대로 유지된다. 저품질 3D 방지 게이트: 시차(baseline) 부족·재투영 오차
초과·카메라 뒤 점은 기각.

### 피시아이 처리

200° 피시아이는 그대로 특징 추출에 쓰기 부적합 → Kalibr equidistant 캘리브로
**중앙 영역을 핀홀로 펴는(rectify) 전처리** 단계를 둔다 (OpenCV fisheye 모듈).
DB 구축·쿼리 양쪽에 동일 적용.

## 아키텍처 — 2개의 독립 프로그램

```
[1] build_db.py  (오프라인, 매핑 1회)
    Hilti bag → 이미지 추출(+타임스탬프)
    포즈 소스 → PoseProvider 인터페이스 ★
                 ├─ GTPoseProvider      (1단계: Hilti GT 궤적 파일)
                 └─ SessionPoseProvider (2단계: Cpp_SLAM --save-session 출력)
    키프레임마다: 이미지 ←(타임스탬프 보간 매칭)→ 포즈
    이미지마다: XFeat 특징점 + MegaLoc 요약벡터
    특징점마다: 카메라-라이다 extrinsic으로 라이다 점을 이미지에 투영해 3D 좌표 부여
    → DB 저장

[2] localize.py  (측위, 쿼리마다)
    쿼리 이미지 → XFeat + MegaLoc
    → MegaLoc 벡터로 DB 검색 → top-K 후보
    → LightGlue: 쿼리 ↔ 후보 특징점 매칭
    → 2D-3D 대응 + 카메라 intrinsic → PnP+RANSAC
    → 6DOF pose + 신뢰도(inlier 수)
```

★ **PoseProvider가 접근안 C의 핵심**: 포즈 출처만 추상화하면 GT↔Cpp_SLAM 교체가
클래스 교체 하나로 끝난다. "GT 포즈 대비 내 SLAM 포즈의 측위 정확도 차이" 비교 실험이
2단계의 산출물.

## 디렉터리 구조

```
VSLAM_repository/
├── README.md
├── requirements.txt        # torch(cpu), opencv-python, numpy, rosbags, ...
├── src/
│   ├── extract.py          # bag → 이미지+타임스탬프 (rosbags 라이브러리)
│   ├── pose_provider.py    # PoseProvider ABC + GT/Session 구현
│   ├── features.py         # XFeat 래퍼 (CPU)
│   ├── retrieval.py        # MegaLoc 래퍼 + 벡터 검색(넘파이 내적/FAISS)
│   ├── matcher.py          # LightGlue 래퍼
│   ├── lift3d.py           # 라이다 점 투영 → 특징점 3D 부여
│   └── pnp.py              # PnP+RANSAC (OpenCV solvePnPRansac)
├── build_db.py
├── localize.py
├── eval.py                 # 성공률/오차/시간 자동 측정
└── docs/specs/
```

## 검증 (성공 기준)

- **지표**: 측위 성공률 = 위치 오차 < 0.25m AND 회전 오차 < 5° 비율 (비주얼 로컬라이제이션
  표준 임계값). 보조: 실패율, 중앙값 오차, 쿼리당 CPU 처리 시간.
- eval.py가 홀수 프레임 전체를 쿼리로 돌려 자동 집계 — 숫자로 판정.

## 마일스톤

1. **M1**: Hilti bag에서 이미지·GT 포즈 추출 동작 (extract + GTPoseProvider)
2. **M2**: DB 구축 완주 (특징 + 검색벡터 + 3D 태깅 저장) — lift3d를 투영 시각화로 검증
3. **M3**: 쿼리 1장 → pose 반환 (end-to-end 최초 성공)
4. **M4**: eval.py 성공률 확보 — **1단계 완성**
5. **M5**: SessionPoseProvider로 Cpp_SLAM 세션 연결, GT 대비 성능 비교 — **2단계 완성**

## 리스크

- **lift3d(2D→3D 태깅)가 최대 고비**: 라이다 점의 이미지 투영에서 캘리브·좌표계·시간
  동기 실수가 나기 쉬움 → M2에서 "라이다 점을 이미지 위에 오버레이"하는 시각화로 검증.
- MegaLoc/XFeat CPU 추론 속도 미지수 → M2에서 실측, 과도하면 CosPlace/ORB로 슬롯 교체
  (슬롯형 설계라 교체 비용 낮음).
- Hilti 카메라가 fisheye일 경우 왜곡 모델 처리 필요 (OpenCV fisheye 모듈).

## 에러 처리 원칙

- 검색 후보 없음 / 매칭 부족 / PnP inlier 부족 → "측위 실패"를 명시적으로 반환 (틀린
  pose를 자신 있게 반환하는 것이 최악 — 실패를 실패로 보고).
