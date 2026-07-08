import numpy as np
from pathlib import Path
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

from src.extract import extract_images


def _make_bag(path: Path, n: int = 3) -> None:
    ts = get_typestore(Stores.ROS1_NOETIC)
    Image = ts.types['sensor_msgs/msg/Image']
    Header = ts.types['std_msgs/msg/Header']
    Time = ts.types['builtin_interfaces/msg/Time']
    with Writer(path) as w:
        conn = w.add_connection('/cam0/image_raw', Image.__msgtype__, typestore=ts)
        for i in range(n):
            img = np.full((4, 5), i, dtype=np.uint8)
            msg = Image(
                header=Header(seq=i, stamp=Time(sec=100 + i, nanosec=0), frame_id='cam'),
                height=4, width=5, encoding='mono8', is_bigendian=0, step=5,
                data=img.reshape(-1))
            w.write(conn, (100 + i) * 10**9, ts.serialize_ros1(msg, Image.__msgtype__))


def test_extract_images(tmp_path):
    bag = tmp_path / 'tiny.bag'
    _make_bag(bag)
    out = tmp_path / 'imgs'
    entries = extract_images(str(bag), '/cam0/image_raw', str(out))
    assert len(entries) == 3
    assert abs(entries[0]['t'] - 100.0) < 1e-6
    assert Path(entries[0]['path']).exists()


def test_missing_topic_raises(tmp_path):
    bag = tmp_path / 'tiny.bag'
    _make_bag(bag)
    import pytest
    with pytest.raises(ValueError):
        extract_images(str(bag), '/no/such/topic', str(tmp_path / 'x'))
