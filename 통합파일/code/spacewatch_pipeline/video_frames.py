"""Read identical requested source frames without repeatedly seeking a video.

The capture must be freshly opened at frame zero. Backward/repeated requests
explicitly seek, so callers are not required to supply monotonic indices.
"""
from numbers import Integral


class FrameReader:
    def __init__(self, capture):
        self.capture = capture
        self.next_index = 0

    def read(self, index):
        if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
            raise ValueError('Source frame index must be a nonnegative integer')
        index = int(index)
        if self.next_index is None or index < self.next_index:
            # OpenCV's CAP_PROP_POS_FRAMES is 1; keep cv2 optional in CPU workers.
            if not self.capture.set(1, index):
                self.next_index = None
                raise IOError('Cannot seek requested source frame: ' + str(index))
            self.next_index = index
        while self.next_index < index:
            if not self.capture.grab():
                self.next_index = None
                raise IOError('Cannot advance to requested source frame: ' + str(index))
            self.next_index += 1
        ok, image = self.capture.read()
        self.next_index = index + 1 if ok else None
        if not ok or image is None:
            raise IOError('Cannot decode requested source frame: ' + str(index))
        return image
