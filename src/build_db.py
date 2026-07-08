# DB 구축 (플랜 C) — bag 이미지 + GT 포즈로 "3D 태깅된 이미지 DB"를 만든다.
#
# 파이프라인:
#   1. bag에서 stride 간격으로 키프레임 추출 → 피시아이 rectify
#   2. 각 키프레임에 GT 포즈(T_world_cam) 보간 부여 (포즈 없는 프레임은 버림)
#   3. XFeat 추출 → 인접 키프레임 매칭 → GT 포즈로 삼각측량 → 특징점별 월드 3D
#   4. retrieval 벡터(MegaLoc) 계산
#   5. db_dir에 프레임별 npz + 전역 index.npz 저장
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
from src.triangulate import triangulate_pairs


def build(cfg_path: str = 'config.yaml'):
    cfg = yaml.safe_load(open(cfg_path))
    db_dir = Path(cfg['db_dir'])
    db_dir.mkdir(parents=True, exist_ok=True)

    calib = load_kalibr(cfg['calib']['kalibr_yaml'], cfg['calib']['cam'])
    rect = Rectifier(calib['K'], calib['D'], calib['size'],
                     out_size=tuple(cfg['calib']['rect_size']),
                     fov_scale=cfg['calib']['fov_scale'])
    K = rect.K_new
    gt = GTPoseProvider(cfg['gt_path'])
    tri = cfg['triangulation']

    # 1. 키프레임 추출 (+2. GT 포즈 부여 — 카메라 timeshift 보정 포함)
    print('이미지 추출 중...')
    entries = extract_images(cfg['bag_path'], cfg['image_topic'],
                             str(db_dir / 'images'),
                             stride=cfg['keyframe_stride'])
    frames = []
    for e in entries:
        T = gt.pose_at(e['t'] + calib['timeshift'])
        if T is not None:
            frames.append({**e, 'T_wc': T})
    print(f'키프레임 {len(frames)}/{len(entries)} (GT 포즈 있는 것만)')

    # 3~4. 특징 + 삼각측량 + retrieval
    xf = XFeat()
    ret = Retrieval(cfg['retrieval_model'])
    gap = tri['pair_gap']
    feats, imgs = [], []
    for f in frames:
        img = rect.rectify(cv2.imread(f['path']))
        cv2.imwrite(f['path'], img)          # rectified로 교체 저장
        imgs.append(img)
        feats.append(xf.extract(img))

    vecs, n_tagged = [], 0
    for i, f in enumerate(frames):
        j = i + gap if i + gap < len(frames) else i - gap
        ia, ib = xf.match(feats[i], feats[j])
        kpts_i = feats[i]['keypoints'][ia]
        kpts_j = feats[j]['keypoints'][ib]
        p3d, valid = triangulate_pairs(
            kpts_i, kpts_j, K, f['T_wc'], frames[j]['T_wc'],
            min_baseline=tri['min_baseline'],
            max_reproj_px=tri['max_reproj_px'],
            max_depth=tri['max_depth'])
        # 유효 3D를 가진 특징점만 저장 (PnP 재료)
        keep = ia[valid]
        np.savez(db_dir / f'frame_{i:05d}.npz',
                 t=f['t'], T_wc=f['T_wc'],
                 keypoints=feats[i]['keypoints'][keep],
                 descriptors=feats[i]['descriptors'][keep],
                 pts3d_w=p3d[valid])
        vecs.append(ret.embed(imgs[i]))
        n_tagged += int(valid.sum())
        if i % 20 == 0:
            print(f'  [{i}/{len(frames)}] 3D점 {valid.sum()}/{len(ia)}')

    np.savez(db_dir / 'index.npz',
             vecs=np.stack(vecs), K=K,
             ts=np.array([f['t'] for f in frames]))
    print(f'완료: 프레임 {len(frames)}, 총 3D점 {n_tagged}, → {db_dir}/')


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else 'config.yaml')
