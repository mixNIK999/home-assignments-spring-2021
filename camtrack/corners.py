#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import itertools

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    # TODO
    get_index = itertools.count()
    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.1,
                          minDistance=7,
                          blockSize=7)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    block_size = feature_params["blockSize"]
    image_0 = frame_sequence[0]
    old_points = cv2.goodFeaturesToTrack(image_0, **feature_params)

    corners_xy = old_points.reshape([-1, 2])
    ind = np.array([next(get_index) for _ in range(corners_xy.shape[0])])
    box = np.ones(corners_xy.shape[0]) * block_size

    corners = FrameCorners(
        ind.copy(),
        corners_xy.copy(),
        box.copy()
    )

    builder.set_corners_at_frame(0, corners)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        img_0_uint8 = (image_0 * 255).astype(np.uint8)
        img_1_uint8 = (image_1 * 255).astype(np.uint8)
        updated_points, st, _ = cv2.calcOpticalFlowPyrLK(img_0_uint8, img_1_uint8, old_points, None, **lk_params)

        updated_points = updated_points.reshape([-1, 2])
        st = st.reshape(-1)

        updated_points = updated_points[st == 1]
        ind = ind[st == 1]

        corners_xy = updated_points
        box = np.ones(corners_xy.shape[0]) * block_size

        corners = FrameCorners(
            ind.copy(),
            corners_xy.copy(),
            box.copy()
        )

        builder.set_corners_at_frame(frame, corners)

        image_0 = image_1
        old_points = updated_points


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
