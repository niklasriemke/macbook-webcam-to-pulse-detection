import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pulse import region_colors  # noqa: E402


def test_skin_mask_excludes_eyebrows_and_hair():
    skin_bgr = (110, 140, 200)
    frame = np.zeros((100, 300, 3), np.uint8)
    frame[:] = skin_bgr
    frame[0:20, 0:100] = (30, 35, 40)  # dunkle Augenbraue/Haare in Region 1
    frame[50:60, 200:300] = (255, 255, 255)  # Glanzlicht in Region 3
    rois = [(0, 0, 100, 100), (100, 0, 100, 100), (200, 0, 100, 100)]
    c = region_colors(frame, rois)
    assert c.shape == (4, 3)
    expected_rgb = np.array(skin_bgr[::-1], float)
    assert np.allclose(c, expected_rgb, atol=0.5)
