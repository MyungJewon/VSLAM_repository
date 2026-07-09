# 평가 — 성공 기준: 위치 오차 < 0.25m AND 회전 오차 < 5°.
#
# 두 모드:
#  · 기본(같은 bag): DB 키프레임 "사이" 중간 프레임을 쿼리로 (held-out)
#  · 크로스-run: --query-bag/--query-gt로 다른 주행의 이미지를 쿼리로 —
#    다른 날/조명/경로에서 run1 맵을 찾아가는 진짜 리로컬라이징 검증
#
# 사용: python -m src.eval [config.yaml] [--limit N]
#                          [--query-bag <폴더> --query-gt <TUM파일>]
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, '.')
from src.extract import extract_images
from src.features import XFeat
from src.localize import Localizer
from src.pose_provider import GTPoseProvider
from src.rectify import Rectifier, load_kalibr
from src.retrieval import Retrieval


def rot_err_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def main(cfg_path='config.yaml', limit=0, query_bag=None, query_gt=None):
    cfg = yaml.safe_load(open(cfg_path))
    calib = load_kalibr(cfg['calib']['kalibr_yaml'], cfg['calib']['cam'])
    # 쿼리도 다중 뷰: 좌/정면/우 각각 시도 후 인라이어 최다 뷰 채택
    q_yaws = cfg['calib'].get('query_view_yaws',
                              cfg['calib'].get('view_yaws', [0]))
    rects = [Rectifier(calib['K'], calib['D'], calib['size'],
                       out_size=tuple(cfg['calib']['rect_size']),
                       fov_scale=cfg['calib']['fov_scale'], yaw_deg=y)
             for y in q_yaws]
    gt = GTPoseProvider(query_gt or cfg['gt_path'])
    # 궤적이 카메라 포즈가 아닐 때(LiDAR SLAM 궤적 등) 카메라 포즈로 환원 —
    # build_db와 동일한 변환. 없으면 항등 (궤적 = cam0 포즈인 GT 모드).
    T_traj_cam0 = np.eye(4)
    if 'T_cam0_traj' in cfg['calib']:
        T_traj_cam0 = np.linalg.inv(
            np.array(cfg['calib']['T_cam0_traj'], float).reshape(4, 4))
    loc = Localizer(cfg['db_dir'], XFeat(), Retrieval(cfg['retrieval_model']),
                    k=cfg['top_k'])

    stride = cfg['keyframe_stride']
    if query_bag:
        # 크로스-run: 다른 bag이라 DB와 겹칠 일이 없음 → stride 간격 그대로 사용
        q_dir = Path(cfg['db_dir']) / 'query_crossrun'
        entries = extract_images(query_bag, cfg['image_topic'], str(q_dir),
                                 stride=stride, limit=limit * stride if limit else 0)
        step = 1
    else:
        # 같은 bag: DB 키프레임(idx%stride==0) "사이" 중간 프레임만 쿼리로.
        # stride/2 간격으로 뽑은 뒤 홀수번째(원본 idx ≡ stride/2 mod stride)만.
        half = max(stride // 2, 1)
        q_dir = Path(cfg['db_dir']) / 'query_images'
        entries = extract_images(cfg['bag_path'], cfg['image_topic'], str(q_dir),
                                 stride=half, limit=2 * limit * half if limit else 0)
        step = 2
    results, records, n_ok = [], [], 0   # records: 시각화용 (t, gt_xyz, est_xyz, ok)
    for i, e in enumerate(entries):
        if step == 2 and i % 2 == 0:   # 같은 bag 모드: DB와 동일 프레임 건너뜀
            continue
        T_gt = gt.pose_at(e['t'] + calib['timeshift'])
        if T_gt is None:
            continue
        T_gt = T_gt @ T_traj_cam0   # 궤적 포즈 → cam0 포즈
        raw = cv2.imread(e['path'])
        r = None
        for rect in rects:                      # 뷰별 시도, 인라이어 최다 채택
            ri = loc.localize(rect.rectify(raw))
            if ri and (r is None or ri['inliers'] > r['inliers']):
                # 뷰 포즈 → 실제 카메라(cam0) 포즈로 환원
                ri['T_wc'] = ri['T_wc'] @ np.linalg.inv(rect.T_cam_view)
                r = ri
        if r is None:
            results.append((e['t'], np.inf, np.inf, 0))
            records.append((e['t'], *T_gt[:3, 3], np.nan, np.nan, np.nan, 0))
            continue
        dp = np.linalg.norm(r['T_wc'][:3, 3] - T_gt[:3, 3])
        dr = rot_err_deg(r['T_wc'][:3, :3], T_gt[:3, :3])
        ok = dp < 0.25 and dr < 5.0
        n_ok += ok
        results.append((e['t'], dp, dr, r['inliers']))
        records.append((e['t'], *T_gt[:3, 3], *r['T_wc'][:3, 3], int(ok)))
        print(f'[{i:4d}] t={e["t"]:.2f} pos={dp:6.3f}m rot={dr:5.2f}° '
              f'inl={r["inliers"]:4d} {"OK" if ok else "FAIL"}')

    np.savez(Path(cfg['db_dir']) / 'eval_results.npz',
             records=np.array(records, float))   # 열: t gx gy gz ex ey ez ok
    n = len(results)
    dps = np.array([r[1] for r in results])
    print(f'\n성공률: {n_ok}/{n} ({100 * n_ok / max(n, 1):.1f}%)  '
          f'중앙값 위치오차: {np.median(dps[np.isfinite(dps)]):.3f}m')


if __name__ == '__main__':
    args = sys.argv[1:]

    def opt(name, default=None):
        return args[args.index(name) + 1] if name in args else default

    cfg = args[0] if args and not args[0].startswith('--') else 'config.yaml'
    main(cfg, int(opt('--limit', 0)),
         query_bag=opt('--query-bag'), query_gt=opt('--query-gt'))
