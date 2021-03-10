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


def _create_mask(mask, points, r):
    for p in points.astype(np.int32):
        mask = cv2.circle(mask, (p[0], p[1]), r, 255, -1)
    return 255 - mask


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    get_index = itertools.count()
    mask_radius = 10
    feature_params = dict(maxCorners=2000,
                          qualityLevel=0.04,
                          minDistance=7,
                          blockSize=7)

    lk_params = dict(winSize=(50, 50),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                     minEigThreshold=0.001)

    block_size = feature_params["blockSize"]
    image_0 = frame_sequence[0]
    old_points = cv2.goodFeaturesToTrack(image_0, **feature_params).reshape([-1, 2])

    corners_xy = old_points
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

        if frame % 20 == 0:
            print(f"corners progress {int(100 * frame / len(frame_sequence))}%")
            print(f"old_points size = {old_points.shape[0]}")

        updated_points = updated_points.reshape([-1, 2])
        st = st.reshape(-1)

        updated_points = updated_points[st == 1]
        ind = ind[st == 1]

        mask = np.zeros_like(img_0_uint8)
        mask = _create_mask(mask, updated_points, mask_radius)

        corners_xy = updated_points
        final_ind = ind

        curr_settings = feature_params.copy()
        curr_settings["maxCorners"] = max(feature_params["maxCorners"] - corners_xy.shape[0], 1)
        new_points = cv2.goodFeaturesToTrack(image_1, mask=mask, **curr_settings)
        if new_points is not None:
            new_points = new_points.reshape([-1, 2])
            new_ind = np.array([next(get_index) for _ in range(new_points.shape[0])])

            corners_xy = np.vstack([corners_xy, new_points])
            final_ind = np.append(final_ind, new_ind)

        box = np.ones(corners_xy.shape[0]) * block_size
        corners = FrameCorners(
            final_ind.copy(),
            corners_xy.copy(),
            box.copy()
        )

        builder.set_corners_at_frame(frame, corners)

        image_0 = image_1
        old_points = corners_xy
        ind = final_ind


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
