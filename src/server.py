# 리로컬라이징 HTTP 서버 — 이미지 POST → 6DOF JSON.
#
# 실행: .venv/bin/uvicorn src.server:app --host 0.0.0.0 --port 8000
#
# 호출 (데이터셋 어안 이미지 — 서버 config 캘리브 사용):
#   curl -F "image=@photo.jpg" http://localhost:8000/localize
# 호출 (임의 핀홀 카메라 — 클라이언트 캘리브 동봉):
#   curl -F "image=@photo.jpg" \
#        -F "intrinsics=615.0,615.0,320.0,240.0"   # fx,fy,cx,cy \
#        -F "dist=0.1,-0.2,0,0,0"                  # 왜곡계수(선택) \
#        http://localhost:8000/localize
import os
import sys

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile

sys.path.insert(0, '.')
from src.reloc import Relocalizer

app = FastAPI(title='CPU Visual Relocalization')
_reloc = None


def get_reloc() -> Relocalizer:
    global _reloc
    if _reloc is None:      # 첫 요청(또는 startup)에서 1회 로드
        _reloc = Relocalizer(os.environ.get('RELOC_CONFIG', 'config.yaml'))
    return _reloc


@app.on_event('startup')
def _warmup():
    get_reloc()


@app.get('/health')
def health():
    return {'status': 'ok', 'views': [y for y, _ in get_reloc().rects],
            'db_frames': len(get_reloc().loc.vecs)}


@app.post('/localize')
async def localize(image: UploadFile = File(...),
                   intrinsics: str = Form(None),   # "fx,fy,cx,cy"
                   dist: str = Form(None)):        # "k1,k2,p1,p2[,k3]"
    buf = np.frombuffer(await image.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return {'ok': False, 'reason': 'decode_failed'}
    K = None
    if intrinsics:
        try:
            fx, fy, cx, cy = [float(v) for v in intrinsics.split(',')]
        except ValueError:
            return {'ok': False, 'reason': 'bad_intrinsics'}
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    d = [float(v) for v in dist.split(',')] if dist else None
    return get_reloc().localize(img, K=K, dist=d)
