# DB 구축 (플랜 C) — bag 이미지 + GT 포즈로 "3D 태깅된 이미지 DB"를 만든다.
#
# 파이프라인 (db_cameras × view_yaws의 각 가상 뷰마다):
#   1. bag에서 stride 간격으로 키프레임 추출(원본 보존) → 뷰별 rectify 저장
#   2. GT 포즈 부여 — GT는 cam0 포즈이므로:
#      T_w_view = T_w_cam0 · T_cam0_imu · T_camX_imu⁻¹ · T_camX_view
#      (200° 어안 1장 → yaw −65/0/+65° 세 핀홀 뷰 = 시야를 버리지 않음)
#   3. XFeat 추출 → 이웃 키프레임 다중 쌍 삼각측량 + 일관성 병합 → 월드 3D
#   4. retrieval 벡터(MegaLoc) 계산
#   5. db_dir에 프레임별 npz + 전역 index.npz 저장 (뷰 통합 연번)
#
# 사용: python -m src.build_db [config.yaml]
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, '.')
from src.extract import extract_images
from src.features import XFeat
from src.pose_provider import GTPoseProvider
from src.rectify import Rectifier, load_kalibr
from src.retrieval import Retrieval
from src.triangulate import merge_multiview, triangulate_pairs


def build(cfg_path: str = 'config.yaml'):
    cfg = yaml.safe_load(open(cfg_path))
    db_dir = Path(cfg['db_dir'])
    db_dir.mkdir(parents=True, exist_ok=True)

    gt = GTPoseProvider(cfg['gt_path'])
    tri = cfg['triangulation']
    offsets = tri.get('pair_offsets', [-2, -1, 1, 2])
    yaws = cfg['calib'].get('view_yaws', [0])
    xf = XFeat()
    ret = Retrieval(cfg['retrieval_model'])

    kalibr = cfg['calib']['kalibr_yaml']
    calib0 = load_kalibr(kalibr, cfg['calib']['cam'])   # GT 기준 카메라(cam0)
    cameras = cfg.get('db_cameras',
                      [{'cam': cfg['calib']['cam'],
                        'topic': cfg['image_topic']}])

    vecs, ts, base = [], [], 0
    for cam_cfg in cameras:
        name = cam_cfg['cam']
        calib = load_kalibr(kalibr, name)
        # GT(cam0 포즈) → 이 카메라 포즈: T_cam0_camX = T_c0_imu · T_cX_imu⁻¹
        T_c0_cX = calib0['T_cam_imu'] @ np.linalg.inv(calib['T_cam_imu'])

        print(f'[{name}] 이미지 추출 중...')
        entries = extract_images(cfg['bag_path'], cam_cfg['topic'],
                                 str(db_dir / 'images' / name),
                                 stride=cfg['keyframe_stride'])
        frames = []
        for e in entries:
            T0 = gt.pose_at(e['t'] + calib['timeshift'])
            if T0 is not None:
                frames.append({**e, 'T_cam': T0 @ T_c0_cX})
        print(f'[{name}] 키프레임 {len(frames)}/{len(entries)}, 뷰 {yaws}')

        for yaw in yaws:
            rect = Rectifier(calib['K'], calib['D'], calib['size'],
                             out_size=tuple(cfg['calib']['rect_size']),
                             fov_scale=cfg['calib']['fov_scale'], yaw_deg=yaw)
            K = rect.K_new
            view_dir = db_dir / 'images' / f'{name}_y{yaw:+d}'
            view_dir.mkdir(parents=True, exist_ok=True)

            feats, imgs, poses, paths = [], [], [], []
            for f in frames:
                img = rect.rectify(cv2.imread(f['path']))
                p = view_dir / Path(f['path']).name
                cv2.imwrite(str(p), img)
                imgs.append(img)
                paths.append(str(p))
                poses.append(f['T_cam'] @ rect.T_cam_view)
                feats.append(xf.extract(img))

            n_tagged = 0
            for i, f in enumerate(frames):
                # 여러 이웃 쌍에서 삼각측량 → 추정 목록 → 일관성 병합
                ests = {}
                for o in offsets:
                    j = i + o
                    if not 0 <= j < len(frames):
                        continue
                    ia, ib = xf.match(feats[i], feats[j])
                    p3d, valid = triangulate_pairs(
                        feats[i]['keypoints'][ia], feats[j]['keypoints'][ib],
                        K, poses[i], poses[j],
                        min_baseline=tri['min_baseline'],
                        max_reproj_px=tri['max_reproj_px'],
                        max_depth_ratio=tri['max_depth_ratio'])
                    for k, p in zip(ia[valid], p3d[valid]):
                        ests.setdefault(int(k), []).append(p)
                keep, p3d_m = merge_multiview(ests, poses[i][:3, 3])
                np.savez(db_dir / f'frame_{base + i:05d}.npz',
                         t=f['t'], T_wc=poses[i], cam=name, yaw=yaw,
                         img=paths[i],
                         keypoints=feats[i]['keypoints'][keep],
                         descriptors=feats[i]['descriptors'][keep],
                         pts3d_w=p3d_m)
                vecs.append(ret.embed(imgs[i]))
                ts.append(f['t'])
                n_tagged += len(keep)
            print(f'[{name} yaw{yaw:+d}] 프레임 {len(frames)}, 3D점 {n_tagged}')
            base += len(frames)

    # 쿼리 카메라(cam0) 정면 뷰의 K를 index에 저장 — localize의 PnP에서 사용
    rect0 = Rectifier(calib0['K'], calib0['D'], calib0['size'],
                      out_size=tuple(cfg['calib']['rect_size']),
                      fov_scale=cfg['calib']['fov_scale'])
    np.savez(db_dir / 'index.npz',
             vecs=np.stack(vecs), K=rect0.K_new, ts=np.array(ts))
    print(f'전체 완료: 프레임 {base} → {db_dir}/')


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else 'config.yaml')
