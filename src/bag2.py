# ROS2 bag(.db3) 최소 리더 — rosbags Reader의 타입해시 검증을 우회한다.
# 배경: 일부 신형(Jazzy) bag은 기록된 타입 해시가 rosbags 내장값과 달라 open이
# AssertionError로 실패한다. 스키마 검증 없이 sqlite를 직접 읽고, 메시지 해석만
# rosbags 타입스토어(deserialize_cdr)에 맡기면 필요한 토픽만 안전하게 읽을 수 있다.
import sqlite3
from pathlib import Path

from rosbags.typesys import Stores, get_typestore

_TS = get_typestore(Stores.ROS2_HUMBLE)  # Imu/Image/CompressedImage는 배포판 무관 동일


def _db_path(bag_dir: str) -> Path:
    d = Path(bag_dir)
    dbs = sorted(d.glob('*.db3'))
    if not dbs:
        raise FileNotFoundError(f'.db3 없음: {bag_dir}')
    return dbs[0]


def topics(bag_dir: str):
    """[(name, type, count)] 반환."""
    with sqlite3.connect(_db_path(bag_dir)) as con:
        rows = con.execute(
            'SELECT t.name, t.type, COUNT(m.id) FROM topics t '
            'LEFT JOIN messages m ON m.topic_id = t.id GROUP BY t.id').fetchall()
    return rows


def messages(bag_dir: str, topic: str):
    """해당 토픽 메시지를 (t_sec: float, msg) 로 순서대로 yield.

    t_sec은 메시지 헤더 stamp(있으면) 기준 — bag 기록시각이 아니라 센서 시각.
    """
    db = _db_path(bag_dir)
    with sqlite3.connect(db) as con:
        row = con.execute('SELECT id, type FROM topics WHERE name=?',
                          (topic,)).fetchone()
        if row is None:
            raise ValueError(f'토픽 없음: {topic}')
        tid, msgtype = row
        cur = con.execute(
            'SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp',
            (tid,))
        for bag_ns, raw in cur:
            msg = _TS.deserialize_cdr(raw, msgtype)
            if hasattr(msg, 'header'):
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            else:
                t = bag_ns * 1e-9
            yield t, msg
