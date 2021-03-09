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
    view_mat3x4_to_pose, build_correspondences, triangulate_correspondences, calc_inlier_indices
)

RUNSAC_STEPS = 107
MAX_REPROJ_ERROR = 0.1


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
    correspondence = build_correspondences(corner_storage[id1], corner_storage[id2])
    points3d, correspondence_ids, median_cos = triangulate_correspondences(correspondence,
                                                                           pose_to_view_mat3x4(pose1),
                                                                           pose_to_view_mat3x4(pose2),
                                                                           intrinsic_mat,
                                                                           (1, 0.1, 0.1)
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
        # get new frame and old frame
        new_frame = next(iter(unknown_frames))
        # pnp + ransac
        # 1) 2d-3d
        frame_2d = corner_storage[new_frame]
        _, (point_3d_ids, point_2d_ids) = snp.intersect(point_cloud_builder.ids, frame_2d.ids)
        # 2) RANSAC
        best_hypothesis = None
        best_inliers_count = -1
        for _ in range(RUNSAC_STEPS):
            # sample
            good_sample = np.random.choice(np.arange(len(point_3d_ids)), 4, raplace=False)
            # hypothesis
            retval, rvec, tvec = cv2.solvePnP(point_cloud_builder.points[point_3d_ids[good_sample]],
                                              frame_2d.points[point_2d_ids[good_sample]],
                                              intrinsic_mat, None, flags=cv2.SOLVEPNP_EPNP)
            r_mat, _ = cv2.Rodrigues(rvec)
            hypothesis_pose = Pose(r_mat, tvec)
            hypothesis_matrix = pose_to_view_mat3x4(hypothesis_pose)
            # check
            inlier_indexes = calc_inlier_indices(point_cloud_builder.points[point_3d_ids],
                                              frame_2d.points[point_2d_ids],
                                              hypothesis_matrix, MAX_REPROJ_ERROR)
            if len(inlier_indexes) > best_inliers_count:
                best_inliers_count = len(inlier_indexes)
                best_hypothesis = hypothesis_pose
        # 3) optimise
        cv2.solvePnP()
        cv2.solvePnPRansac()

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
