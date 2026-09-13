import numpy as np
import pytest

from flywirehead.engine import decode_frame, FRAME_HEIGHT, FRAME_WIDTH


def test_screen_pixels_are_flipped_from_webgl_and_alpha_is_not_a_sensory_channel():
    pixels = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 4), np.uint8)
    pixels[0, 0] = [10, 20, 30, 255]
    pixels[-1, -1] = [70, 80, 90, 0]
    frame = decode_frame(pixels.tobytes())
    assert frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
    assert frame.flags.c_contiguous
    assert frame[-1, 0].tolist() == [10, 20, 30]
    assert frame[0, -1].tolist() == [70, 80, 90]
    with pytest.raises(ValueError):
        decode_frame(pixels.tobytes()[:-1])

