"""The Atari-style observation shared by the pixel games: the last K grayscale frames, as DQN and NEC see Atari."""
from collections import deque

import numpy as np

PIXELS = 84  # side of every frame, as in DQN's preprocessing


def scale(frames):
    """uint8 gray frames -> the flat float32 observation, values k/255."""
    return (np.stack(frames).astype(np.float32) / np.float32(255)).ravel()


class FrameStack:
    """The last `k` frames. An episode's stack starts as k copies of its first frame (as in DQN)."""

    def __init__(self, k):
        self.k = k
        self._frames = deque(maxlen=k)

    def reset(self):
        """Forget the frames; the next push starts a new episode's stack."""
        self._frames.clear()

    def push(self, frame):
        """Add the newest frame and return the observation (oldest frame first)."""
        if not self._frames:
            self._frames.extend([frame] * self.k)
        else:
            self._frames.append(frame)
        return scale(list(self._frames))

    def static(self, frame):
        """The observation of a state as if it had always been so; the stack itself is not touched."""
        return scale([frame] * self.k)
