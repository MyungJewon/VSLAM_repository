# bag의 토픽/타입/메시지수 출력 — Task 2에서 config.yaml 채우기용
# 사용: python tools/inspect_bag.py <bag파일(.bag) 또는 ROS2 폴더>
import sys
from pathlib import Path

sys.path.insert(0, '.')


def main(bag_path: str) -> None:
    p = Path(bag_path)
    if p.is_dir() and (p / 'metadata.yaml').exists():   # ROS2
        from src.bag2 import topics
        for name, typ, cnt in sorted(topics(bag_path)):
            print(f'{name:45s} {typ:45s} {cnt:7d} msgs')
    else:                                                # ROS1
        from rosbags.highlevel import AnyReader
        with AnyReader([p]) as reader:
            print(f'duration: {(reader.end_time - reader.start_time) / 1e9:.1f}s')
            for c in sorted(reader.connections, key=lambda c: c.topic):
                print(f'{c.topic:45s} {c.msgtype:45s} {c.msgcount:7d} msgs')


if __name__ == '__main__':
    main(sys.argv[1])
