#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import sortednp as snp
import cv2

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose, build_correspondences, triangulate_correspondences, calc_inlier_indices,
    rodrigues_and_translation_to_view_mat3x4, TriangulationParameters, check_baseline
)

RUNSAC_STEPS = 107
MAX_REPROJ_ERROR = 0.1
TRIANGULATION_PARAMS = TriangulationParameters(2, 0.1, 0.1)
INIT_TRIANGULATION_PARAMS = TriangulationParameters(10, 0.01, 0.01)
BASELINE = 20


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    # TODO: implement
    # init
    id1, pose1 = known_view_1
    id2, pose2 = known_view_2
    print(f"init tracking with {id1} & {id2} frames")
    correspondence = build_correspondences(corner_storage[id1], corner_storage[id2])
    points3d, correspondence_ids, median_cos = triangulate_correspondences(correspondence,
                                                                           pose_to_view_mat3x4(pose1),
                                                                           pose_to_view_mat3x4(pose2),
                                                                           intrinsic_mat,
                                                                           INIT_TRIANGULATION_PARAMS
                                                                           )

    point_cloud_builder = PointCloudBuilder(correspondence_ids, points3d)
    frame_count = len(corner_storage)
    unknown_frames = set(range(frame_count))
    unknown_frames.remove(id1)
    unknown_frames.remove(id2)
    known_frames = {id1, id2}
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count
    view_mats[id1] = pose_to_view_mat3x4(pose1)
    view_mats[id2] = pose_to_view_mat3x4(pose2)

    while len(unknown_frames) > 0:
        # get new frame
        new_frame = next(iter(unknown_frames))
        unknown_frames.remove(new_frame)
        # pnp + ransac
        # 1) 2d-3d
        frame_2d = corner_storage[new_frame]
        _, (point_3d_ids, point_2d_ids) = snp.intersect(point_cloud_builder.ids, frame_2d.ids)
        good_points_3d = point_cloud_builder.points[point_3d_ids]
        good_points_2d = frame_2d.points[point_2d_ids]
        # 2) RANSAC
        _, hypothesis_rvec, hypothesis_tvec, inliers = cv2.solvePnPRansac(good_points_3d,
                                                                          good_points_2d,
                                                                          intrinsic_mat, None,
                                                                          iterationsCount=RUNSAC_STEPS)
        # 3) optimise
        _, rvec, tvec = cv2.solvePnP(good_points_3d[inliers],
                                     good_points_2d[inliers],
                                     intrinsic_mat, None,
                                     tvec=hypothesis_tvec, rvec=hypothesis_rvec, useExtrinsicGuess=True)

        new_view_mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
        view_mats[new_frame] = new_view_mat
        # retriangulate
        for known_frame in known_frames:

            if check_baseline(new_view_mat, view_mats[known_frame], BASELINE):
                continue
            correspondence = build_correspondences(frame_2d, corner_storage[known_frame],
                                                   ids_to_remove=np.setdiff1d(frame_2d.ids, inliers))
            new_points3d, new_correspondence_ids, new_median_cos = triangulate_correspondences(correspondence,
                                                                                               new_view_mat,
                                                                                               view_mats[known_frame],
                                                                                               intrinsic_mat,
                                                                                               TRIANGULATION_PARAMS
                                                                                               )
            point_cloud_builder.add_points(new_correspondence_ids, new_points3d)
        known_frames.add(new_frame)

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
