# 매칭 갤러리 — 최근 평가의 쿼리들을 성공/실패/None별로 샘플링해서
# 각각 [쿼리 | 최우수 DB 후보] 매칭 이미지를 db/gallery/에 생성한다.
# 사용: python tools/match_gallery.py [config.yaml] [--n 6] [--query-bag <폴더>]
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, '.')
from src.bag2 import messages
from src.features import XFeat
from src.pnp import solve_pnp
from src.rectify import Rectifier, load_kalibr
from src.retrieval import Retrieval, top_k


def render(q_img, q_feats, f, xf, K, label_extra=''):
    """쿼리 vs DB 프레임 1개 → (매칭그림, 인라이어수)"""
    iq, idb = xf.match(q_feats, {'descriptors': f['descriptors']}) \
        if len(f['descriptors']) else (np.array([], int), np.array([], int))
    r = solve_pnp(q_feats['keypoints'][iq], f['pts3d_w'][idb], K, None) \
        if len(iq) else None
    inl_set = set(r['inlier_idx']) if r else set()
    db_img = cv2.imread(str(f['img'])) if 'img' in f else np.zeros_like(q_img)
    kq = [cv2.KeyPoint(*q_feats['keypoints'][i], 1) for i in iq]
    kd = [cv2.KeyPoint(*f['keypoints'][j], 1) for j in idb]
    dm = [cv2.DMatch(i, i, 0) for i in range(len(iq))]
    mask = [1 if i in inl_set else 0 for i in range(len(iq))]
    vis = cv2.drawMatches(q_img, kq, db_img, kd, dm, None,
                          matchColor=(0, 220, 0), matchesMask=mask,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    inl = r['inliers'] if r else 0
    cam = str(f['cam']) if 'cam' in f else '?'
    yaw = f' y{int(f["yaw"]):+d}' if 'yaw' in f else ''
    cv2.putText(vis, f'{label_extra} DB[{cam}{yaw}] 매칭 {len(iq)} 인라이어 {inl}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return vis, inl


def main(cfg_path='config.yaml', n=6, query_bag=None):
    cfg = yaml.safe_load(open(cfg_path))
    db = Path(cfg['db_dir'])
    out_dir = db / 'gallery'
    out_dir.mkdir(exist_ok=True)
    calib = load_kalibr(cfg['calib']['kalibr_yaml'], cfg['calib']['cam'])
    rect = Rectifier(calib['K'], calib['D'], calib['size'],
                     out_size=tuple(cfg['calib']['rect_size']),
                     fov_scale=cfg['calib']['fov_scale'])
    xf = XFeat()
    ret = Retrieval(cfg['retrieval_model'])
    idx = np.load(db / 'index.npz')
    K = idx['K']

    # 평가 records → 케이스 분류
    rec = np.load(db / 'eval_results.npz')['records']
    t_all, g, e = rec[:, 0], rec[:, 1:4], rec[:, 4:7]
    fin = np.isfinite(e[:, 0])
    err = np.linalg.norm(e - g, axis=1)
    cases = {'ok': np.where(fin & (err < 0.25))[0],
             'off': np.where(fin & (err >= 0.25))[0],   # 추정했지만 오차 큼
             'none': np.where(~fin)[0]}

    # 쿼리 t → 이미지 파일 (bag 인덱스 복원)
    bag = query_bag or cfg['bag_path']
    q_dir = db / ('query_crossrun' if query_bag else 'query_images')
    tmap = {round(t, 6): i
            for i, (t, _) in enumerate(messages(bag, cfg['image_topic']))}

    for kind, rows in cases.items():
        picks = rows[np.linspace(0, len(rows) - 1, min(n, len(rows)),
                                 dtype=int)] if len(rows) else []
        for r_i in picks:
            bi = tmap.get(round(float(t_all[r_i]), 6))
            qp = q_dir / f'{bi:06d}.png'
            if bi is None or not qp.exists():
                continue
            q_img = rect.rectify(cv2.imread(str(qp)))
            q_feats = xf.extract(q_img)
            cands = top_k(ret.embed(q_img), idx['vecs'], cfg['top_k'])
            # 후보 중 인라이어 최다 프레임으로 렌더
            best_vis, best_inl = None, -1
            for c in cands:
                f = np.load(db / f'frame_{c:05d}.npz')
                vis, inl = render(q_img, q_feats, f, xf, K,
                                  f'[{kind}] q{bi:06d} #{c}')
                if inl > best_inl:
                    best_vis, best_inl = vis, inl
            err_s = f'{err[r_i]:.2f}m' if np.isfinite(err[r_i]) else 'None'
            out = out_dir / f'{kind}_{bi:06d}_err{err_s}.png'
            cv2.imwrite(str(out), best_vis)
            print(f'{out.name}  (인라이어 {best_inl})')
    print(f'\n갤러리: {out_dir}/  →  open {out_dir}')


if __name__ == '__main__':
    a = sys.argv[1:]

    def opt(name, default=None):
        return a[a.index(name) + 1] if name in a else default

    cfg = a[0] if a and not a[0].startswith('--') else 'config.yaml'
    main(cfg, int(opt('--n', 6)), opt('--query-bag'))
