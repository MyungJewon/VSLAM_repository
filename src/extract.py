# bag의 이미지 토픽을 PNG로 저장하고 [{'t','path'}] 목록 반환 (rosbags — ROS 설치 불필요)
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader


def _decode(msg) -> np.ndarray:
    """raw Image(mono8/bgr8/rgb8) + CompressedImage 지원."""
    if hasattr(msg, 'format'):  # CompressedImage
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    buf = np.frombuffer(msg.data, np.uint8)
    if msg.encoding == 'mono8':
        return buf.reshape(msg.height, msg.width)
    img = buf.reshape(msg.height, msg.width, -1)
    if msg.encoding == 'rgb8':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def extract_images(bag_path: str, topic: str, out_dir: str, stride: int = 1,
                   limit: int = 0):
    """bag(ROS1 .bag 또는 ROS2 폴더)에서 이미지 추출. 반환: [{'t','path'}, ...]"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = Path(bag_path)
    if p.is_dir() and (p / 'metadata.yaml').exists():   # ROS2 — bag2 직접 리더
        from src.bag2 import messages
        entries = []
        for idx, (t, msg) in enumerate(messages(bag_path, topic)):
            if limit and len(entries) >= limit:
                break
            if idx % stride:
                continue
            path = out / f'{idx:06d}.png'
            cv2.imwrite(str(path), _decode(msg))
            entries.append({'t': t, 'path': str(path)})
        return entries
    entries, idx = [], 0
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise ValueError(f'토픽 없음: {topic}')
        for conn, _timestamp, raw in reader.messages(connections=conns):
            if idx % stride == 0:
                msg = reader.deserialize(raw, conn.msgtype)
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                path = out / f'{idx:06d}.png'
                cv2.imwrite(str(path), _decode(msg))
                entries.append({'t': t, 'path': str(path)})
            idx += 1
    return entries
