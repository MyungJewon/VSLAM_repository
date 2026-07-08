# 쿼리 이미지가 어떤 DB 프레임과 매칭되는지 시각 확인.
# 쿼리 1장 → MegaLoc top-k 후보 각각에 대해 [쿼리|DB이미지] 매칭 라인 그림 저장.
# 사용: python tools/show_matches.py <쿼리이미지(raw)> [config.yaml] [출력접두사]
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


def bag_time_index(bag_path, topic):
    """bag의 (타임스탬프 → 메시지 인덱스) 매핑 — DB 이미지 파일명 복원용."""
    return {round(t, 6): i for i, (t, _) in enumerate(messages(bag_path, topic))}


def main(query_path, cfg_path='config.yaml', out_prefix='db/match'):
    cfg = yaml.safe_load(open(cfg_path))
    db = Path(cfg['db_dir'])
    calib = load_kalibr(cfg['calib']['kalibr_yaml'], cfg['calib']['cam'])
    rect = Rectifier(calib['K'], calib['D'], calib['size'],
                     out_size=tuple(cfg['calib']['rect_size']),
                     fov_scale=cfg['calib']['fov_scale'])
    xf = XFeat()
    ret = Retrieval(cfg['retrieval_model'])
    idx = np.load(db / 'index.npz')
    K = idx['K']

    # 구형 DB(frame npz에 img 경로 없음)에서만 필요 — 지연 생성
    tmaps = {}

    def tmap(cam):
        if cam not in tmaps:
            topic = next(c['topic'] for c in cfg.get('db_cameras', [])
                         if c['cam'] == cam)
            tmaps[cam] = bag_time_index(cfg['bag_path'], topic)
        return tmaps[cam]

    q_img = rect.rectify(cv2.imread(query_path))
    q_feats = xf.extract(q_img)
    cands = top_k(ret.embed(q_img), idx['vecs'], cfg['top_k'])
    print(f'쿼리: {query_path}\ntop-{len(cands)} 후보: {list(cands)}')

    for rank, c in enumerate(cands):
        f = np.load(db / f'frame_{c:05d}.npz')
        cam = str(f['cam']) if 'cam' in f else cfg['calib']['cam']
        if 'yaw' in f:
            cam += f' y{int(f["yaw"]):+d}'
        if 'img' in f:                                # 신형 DB: 경로 저장됨
            db_img = cv2.imread(str(f['img']))
        else:                                         # 구형 DB: 타임스탬프 복원
            bag_i = tmap(str(f['cam'])).get(round(float(f['t']), 6))
            img_p = db / 'images' / cam / f'{bag_i:06d}.png'
            db_img = cv2.imread(str(img_p)) if bag_i is not None \
                and img_p.exists() else None
        if db_img is None:
            db_img = np.zeros_like(q_img)
        iq, idb = xf.match(q_feats, {'descriptors': f['descriptors']})
        r = solve_pnp(q_feats['keypoints'][iq], f['pts3d_w'][idb], K, None) \
            if len(iq) else None
        inl = r['inliers'] if r else 0
        inl_set = set(r['inlier_idx']) if r else set()

        kq = [cv2.KeyPoint(*q_feats['keypoints'][i], 1) for i in iq]
        kd = [cv2.KeyPoint(*f['keypoints'][j], 1) for j in idb]
        dm = [cv2.DMatch(i, i, 0) for i in range(len(iq))]
        mask = [1 if i in inl_set else 0 for i in range(len(iq))]
        vis = cv2.drawMatches(q_img, kq, db_img, kd, dm, None,
                              matchColor=(0, 220, 0), matchesMask=mask,
                              singlePointColor=(80, 80, 80),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        label = (f'#{rank} frame_{c:05d} [{cam}] 매칭 {len(iq)} / '
                 f'인라이어 {inl}' + (' → PnP 성공' if r else ' → PnP 실패'))
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        out = f'{out_prefix}_{rank}.png'
        cv2.imwrite(out, vis)
        print(f'  {label}  → {out}')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[1] if len(a) > 1 else 'config.yaml',
         a[2] if len(a) > 2 else 'db/match')
