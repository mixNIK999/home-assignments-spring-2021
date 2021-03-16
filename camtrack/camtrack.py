#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

import random
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

RUNSAC_STEPS = 1007
TRIANGULATION_PARAMS = TriangulationParameters(2, 10, 0.1)
INLIER_MAX_ERROR = 4
INIT_TRIANGULATION_PARAMS = TriangulationParameters(10, 0.01, 0.01)
MIN_2D_3D = 10
MIN_INLIER = 10
BASELINE = 10

INIT_FRAME_WINDOW = 20
MIN_E_INLIERS = 50
E_THRESHOLD = 0.01


def init_frame_view(corner_storage: CornerStorage,
                    intrinsic_mat: np.ndarray) -> Tuple[Tuple[int, Pose], Tuple[int, Pose]]:
    def test_frames(f_1: int, f_2: int) -> Tuple[Pose, int]:
        print(f"testing {(f_1, f_2)}")
        correspondence = build_correspondences(corner_storage[f_1], corner_storage[f_2])
        E, E_mask = cv2.findEssentialMat(correspondence.points_1, correspondence.points_2, intrinsic_mat,
                                         method=cv2.RANSAC, threshold=E_THRESHOLD)
        E_mask = E_mask.flatten()
        if np.sum(E_mask) > MIN_E_INLIERS:
            pass
        print(f"#inliers of E = {np.sum(E_mask)}")
        R_1, R_2, t_base = cv2.decomposeEssentialMat(E)
        new_correspondence = build_correspondences(corner_storage[f_1], corner_storage[f_2],
                                                   ids_to_remove=np.argwhere(1 - E_mask))

        def test_pose(p_2: Pose):
            p_1 = Pose(np.eye(3), np.zeros(3))

            _, ids, _ = triangulate_correspondences(new_correspondence,
                                                    pose_to_view_mat3x4(p_1),
                                                    pose_to_view_mat3x4(p_2),
                                                    intrinsic_mat,
                                                    TriangulationParameters(2, 0.1, 0.1)
                                                    )
            return len(ids)

        possible_poses = [Pose(R_1, t_base), Pose(R_1, -t_base), Pose(R_2, t_base), Pose(R_2, -t_base)]
        confidence = [test_pose(pose) for pose in possible_poses]
        best_ans = np.argmax(confidence)
        print(f"win pose {best_ans} with #inliers = {confidence[best_ans]}")
        return possible_poses[best_ans], confidence[best_ans]

    all_interesting_pairs = list(zip(range(len(corner_storage)), range(INIT_FRAME_WINDOW, len(corner_storage))))
    # all_interesting_pairs = [(i, j) for i in range(len(corner_storage))
    #                          for j in range(i + INIT_FRAME_WINDOW, len(corner_storage))]
    res = [test_frames(f_1, f_2) for f_1, f_2 in all_interesting_pairs]

    ans = max(range(len(res)), key=lambda i: res[i][1])
    best_f_1, best_f_2 = all_interesting_pairs[ans]
    return (best_f_1, Pose(np.eye(3), np.zeros(3))), (best_f_2, res[ans][0])


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    if known_view_1 is None or known_view_2 is None:
        # raise NotImplementedError()
        known_view_1, known_view_2 = init_frame_view(corner_storage, intrinsic_mat)

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
    unknown_frames = set(range(int(frame_count)))
    unknown_frames.remove(id1)
    unknown_frames.remove(id2)
    known_frames = {id1, id2}
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count
    view_mats[id1] = pose_to_view_mat3x4(pose1)
    view_mats[id2] = pose_to_view_mat3x4(pose2)

    while len(unknown_frames) > 0:
        print(f"progress = {100 * len(known_frames) / len(corner_storage):.1f}%")
        print(f"#points in cloud = {len(point_cloud_builder.ids)}")
        # get new frame
        # new_frame = random.sample(unknown_frames, 1)[0]
        new_frame = next(iter(unknown_frames))
        print(f"choosed new frame = {new_frame}")
        # pnp + ransac
        # 1) 2d-3d
        frame_2d = corner_storage[new_frame]
        _, (point_3d_ids, point_2d_ids) = snp.intersect(point_cloud_builder.ids.flatten(), frame_2d.ids.flatten(),
                                                        indices=True)
        print(f"#2d-3d: {len(point_3d_ids)}")
        if len(point_3d_ids) < MIN_2D_3D:
            print("Skip\n")
            continue
        # 2) RANSAC
        good_points_3d = point_cloud_builder.points[point_3d_ids]
        good_points_2d = frame_2d.points[point_2d_ids]
        _, hypothesis_rvec, hypothesis_tvec, inliers = cv2.solvePnPRansac(good_points_3d,
                                                                          good_points_2d,
                                                                          intrinsic_mat, None,
                                                                          reprojectionError=INLIER_MAX_ERROR,
                                                                          iterationsCount=RUNSAC_STEPS)
        if inliers is None or len(inliers) < MIN_INLIER:
            print("Skip\n")
            continue
        print(f"#inliers = {len(inliers)}")
        # 3) optimise
        _, rvec, tvec = cv2.solvePnP(good_points_3d[inliers],
                                     good_points_2d[inliers],
                                     intrinsic_mat, None,
                                     tvec=hypothesis_tvec, rvec=hypothesis_rvec, useExtrinsicGuess=True)

        new_view_mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
        view_mats[new_frame] = new_view_mat
        # triangulate
        print(f"#triangulated points:")
        for known_frame in known_frames:

            if check_baseline(new_view_mat, view_mats[known_frame], BASELINE):
                print(f"#triangulated points:")
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
            print(f"({known_frame}, {len(new_correspondence_ids)})", end=" ")
        unknown_frames.remove(new_frame)
        known_frames.add(new_frame)
        print("\n")

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
