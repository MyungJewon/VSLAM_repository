# CPU Visual Relocalization

정확한 포즈(LiDAR SLAM 또는 GT)에 이미지를 태깅해 DB를 만들고,
쿼리 이미지 한 장으로 **CPU만으로 6DOF 위치를 반환**하는 파이프라인.

- 특징: XFeat · 검색: MegaLoc(폴백 CosPlace) · 매칭: LighterGlue · 측위: PnP+RANSAC
- 데이터: Hilti SLAM Challenge (LiDAR+카메라+IMU 동기 rosbag)
- 설계: `docs/specs/`, 계획: `docs/plans/`

## 실행 (예정)
```
python build_db.py config.yaml     # DB 구축 (오프라인 1회)
python localize.py query.png       # 쿼리 측위
python eval.py config.yaml         # 성공률 측정
```
