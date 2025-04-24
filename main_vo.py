#!/usr/bin/env -S python3 -O
"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import cv2
import os
import math
import time 
import platform 

from config import Config

from visual_odometry import VisualOdometryEducational
from visual_odometry_rgbd import VisualOdometryRgbd, VisualOdometryRgbdTensor
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset_factory import dataset_factory
from dataset_types import DatasetType, SensorType

from mplot_thread import Mplot2d, Mplot3d
from qplot_thread import Qplot2d

from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_tracker_configs import FeatureTrackerConfigs

from utils_sys import Printer
from rerun_interface import Rerun

from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

bf = cv2.BFMatcher(cv2.NORM_L2)

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kResultsFolder = kRootFolder + '/results'

global_plotting = True

kUseRerun = True
# check rerun does not have issues 
if global_plotting:
    if kUseRerun and not Rerun.is_ok():
        kUseRerun = False
    
"""
use or not pangolin (if you want to use it then you need to install it by using the script install_thirdparty.sh)
"""
if global_plotting:
    kUsePangolin = True  
    if platform.system() == 'Darwin':
        kUsePangolin = True # Under mac force pangolin to be used since Mplot3d() has some reliability issues                
    if kUsePangolin:
        from viewer3D import Viewer3D

    kUseQplot2d = False
    if platform.system() == 'Darwin':
        kUseQplot2d = True # Under mac force the usage of Qtplot2d: It is smoother 

    def factory_plot2d(*args,**kwargs):
        if kUseRerun:
            return None
        if kUseQplot2d:
            return Qplot2d(*args,**kwargs)
        else:
            return Mplot2d(*args,**kwargs)


def factory_plot2d(*args,**kwargs):
    if kUseRerun:
        return None
    if kUseQplot2d:
        return Qplot2d(*args,**kwargs)
    else:
        return Mplot2d(*args,**kwargs)
    

def run_exp(name, feature, max_images=10, baseline=False):
    config_loc = os.environ.get('PYSLAM_CONFIG')
    config_name = config_loc.split('.')[0]
    config = Config()
    
    dataset = dataset_factory(config)

    groundtruth = groundtruth_factory(config.dataset_settings)

    cam = PinholeCamera(config)

    num_features=2000  # how many features do you want to detect and track?
    if config.num_features_to_extract > 0:  # override the number of features to extract if we set something in the settings file
        num_features = config.num_features_to_extract
        
    # select your tracker configuration (see the file feature_tracker_configs.py) 
    # LK_SHI_TOMASI, LK_FAST
    # SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, LIGHTGLUE, XFEAT, XFEAT_XFEAT, LOFTR
    tracker_config = feature
    tracker_config['num_features'] = num_features
   
    feature_tracker = feature_tracker_factory(**tracker_config)

    # create visual odometry object 
    if dataset.sensor_type == SensorType.RGBD:
        vo = VisualOdometryRgbdTensor(cam, groundtruth)  # only for RGBD
        Printer.green('Using VisualOdometryRgbdTensor')
    else:
        vo = VisualOdometryEducational(cam, groundtruth, feature_tracker)
        Printer.green('Using VisualOdometryEducational')
    time.sleep(1) # time to read the message
    if global_plotting:
        is_draw_traj_img = True
        traj_img_size = 800
        traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
        half_traj_img_size = int(0.5*traj_img_size)
        draw_scale = 1

        plt3d = None
        
        viewer3D = None 
        
        is_draw_3d = True
        is_draw_with_rerun = kUseRerun
        if is_draw_with_rerun:
            Rerun.init_vo()
        else: 
            if kUsePangolin:
                viewer3D = Viewer3D(scale=dataset.scale_viewer_3d*10)
            else:
                plt3d = Mplot3d(title='3D trajectory')

        is_draw_err = True 
        err_plt = factory_plot2d(xlabel='img id', ylabel='m',title='error')
        
        is_draw_matched_points = True 
        matched_points_plt = factory_plot2d(xlabel='img id', ylabel='# matches',title='# matches')

    
    matched_kps = []
    num_inliers = []
    px_shifts = []
    rs = []
    ts = []
    kps = []
    des = []
    
    img_id = 0
    images = []
    masks = []
    while True:
        if max_images is not None and img_id >= max_images:
            break
        if max_images is None and img_id >= dataset.num_frames-1:
            break

        img = None

        if dataset.isOk():
            timestamp = dataset.getTimestamp()          # get current timestamp 
            img, mask = dataset.getImageColor(img_id)
            depth = dataset.getDepth(img_id)
            if baseline:
                mask = None

        if img is not None:
            images.append(img)
            matched_kp, num_inlier, px_shift, kp_cur, des_cur,rot,trans = vo.track(img, depth, img_id, timestamp, mask=mask)  # main VO function 
            if matched_kp is not None:
                matched_kps.append(matched_kp)
                num_inliers.append(num_inlier)
                px_shifts.append(px_shift)
                kps.append(kp_cur)
                des.append(des_cur)
                rs.append(rot)
                ts.append(trans)

            if(len(vo.traj3d_est)>1):	       # start drawing from the third image (when everything is initialized and flows in a normal way)

                x, y, z = vo.traj3d_est[-1]
                gt_x, gt_y, gt_z = vo.traj3d_gt[-1]
                if global_plotting:
                    if is_draw_traj_img:      # draw 2D trajectory (on the plane xz)
                        draw_x, draw_y = int(draw_scale*x) + half_traj_img_size, half_traj_img_size - int(draw_scale*z)
                        draw_gt_x, draw_gt_y = int(draw_scale*gt_x) + half_traj_img_size, half_traj_img_size - int(draw_scale*gt_z)
                        cv2.circle(traj_img, (draw_x, draw_y), 1,(img_id*255/4540, 255-img_id*255/4540, 0), 1)   # estimated from green to blue
                        cv2.circle(traj_img, (draw_gt_x, draw_gt_y), 1,(0, 0, 255), 1)  # groundtruth in red
                        # write text on traj_img
                        cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
                        text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
                        cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                        # show 		

                        if is_draw_with_rerun:
                            Rerun.log_img_seq('trajectory_img/2d', img_id, traj_img)
                        else:
                            cv2.imshow('Trajectory', traj_img)

                if global_plotting:
                    if is_draw_with_rerun:                                        
                        Rerun.log_2d_seq_scalar('trajectory_error/err_x', img_id, math.fabs(gt_x-x))
                        Rerun.log_2d_seq_scalar('trajectory_error/err_y', img_id, math.fabs(gt_y-y))
                        Rerun.log_2d_seq_scalar('trajectory_error/err_z', img_id, math.fabs(gt_z-z))
                        
                        Rerun.log_2d_seq_scalar('trajectory_stats/num_matches', img_id, vo.num_matched_kps)
                        Rerun.log_2d_seq_scalar('trajectory_stats/num_inliers', img_id, vo.num_inliers)
                        
                        Rerun.log_3d_camera_img_seq(img_id, vo.draw_img, None, cam, vo.poses[-1])
                        Rerun.log_3d_trajectory(img_id, vo.traj3d_est, 'estimated', color=[0,0,255])
                        Rerun.log_3d_trajectory(img_id, vo.traj3d_gt, 'ground_truth', color=[255,0,0])     
                    else:
                        if is_draw_3d:           # draw 3d trajectory 
                            if kUsePangolin:
                                viewer3D.draw_vo(vo)   
                            else:
                                plt3d.draw(vo.traj3d_gt,'ground truth',color='r',marker='.')
                                plt3d.draw(vo.traj3d_est,'estimated',color='g',marker='.')

                        if is_draw_err:         # draw error signals 
                            errx = [img_id, math.fabs(gt_x-x)]
                            erry = [img_id, math.fabs(gt_y-y)]
                            errz = [img_id, math.fabs(gt_z-z)] 
                            err_plt.draw(errx,'err_x',color='g')
                            err_plt.draw(erry,'err_y',color='b')
                            err_plt.draw(errz,'err_z',color='r')

                        if is_draw_matched_points:
                            matched_kps_signal = [img_id, vo.num_matched_kps]
                            inliers_signal = [img_id, vo.num_inliers]                    
                            matched_points_plt.draw(matched_kps_signal,'# matches',color='b')
                            matched_points_plt.draw(inliers_signal,'# inliers',color='g')                                                     
            if global_plotting:
                # draw camera image 
                if not is_draw_with_rerun:
                    cv2.imshow('Camera', vo.draw_img)				

        else: 
            time.sleep(0.1) 
                
        # get keys 
        if global_plotting:
            key = matched_points_plt.get_key() if matched_points_plt is not None else None
            if key == '' or key is None:
                key = err_plt.get_key() if err_plt is not None else None
            if key == '' or key is None:
                key = plt3d.get_key() if plt3d is not None else None
                
            # press 'q' to exit!
            key_cv = cv2.waitKey(1) & 0xFF
            if key == 'q' or (key_cv == ord('q')):            
                break
            if viewer3D and viewer3D.is_closed():
                break
        img_id += 1

    #print('press a key in order to exit...')
    #cv2.waitKey(0)
    if global_plotting:
        try:
            if is_draw_traj_img:
                if not os.path.exists(kResultsFolder):
                    os.makedirs(kResultsFolder, exist_ok=True)
                print(f'saving {kResultsFolder}/map.png')
                cv2.imwrite(f'{kResultsFolder}/map.png', traj_img)
            if is_draw_3d:
                if not kUsePangolin:
                    plt3d.quit()
                else: 
                    viewer3D.quit()
            if is_draw_err:
                err_plt.quit()
            if is_draw_matched_points is not None:
                matched_points_plt.quit()
                        
            cv2.destroyAllWindows()
        except Exception as e:
            print(f'Error in closing windows: {e}')
            pass


    idxs = range(len(matched_kps))
    idxs = [i*dataset.skip for i in idxs]
    fig, ax = plt.subplots(2,1)
    ax[0].plot(idxs, matched_kps, label='matched_kps')
    ax[0].plot(idxs, num_inliers, label='num_inliers')
    ax[0].legend()
    ax[1].plot(idxs, px_shifts, label='px_shifts')
    ax[1].legend()

    # Initialize an empty list to store matches between consecutive frames
    matches_list = []

    # Loop over consecutive pairs of frames
    for i in range(len(des) - 1):
        des1 = des[i]
        des2 = des[i + 1]

        # If either descriptor is None, we simply append an empty match list
        if des1 is None or des2 is None:
            matches = []
        else:
            # Match features using BFMatcher and sort the matches based on distance
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
        
        # Append the matches corresponding to frame i (matched with frame i+1)
        matches_list.append(matches)

    tracks = {}
    next_track_id = 0

    for i in range(len(des)):
        if des[i] is not None and des[i].dtype != np.float32:
            des[i] = des[i].astype(np.float32)

    # Process each frame as a starting point
    for start_frame in tqdm(range(len(images) - 1)):
        # Initialize new tracks for features in this frame
        current_features = {i: next_track_id + i for i in range(len(kps[start_frame]))}
        
        # Add new features to tracks
        for i in range(len(kps[start_frame])):
            tracks[next_track_id + i] = [(start_frame, i)]
        
        next_track_id += len(kps[start_frame])

        # Track these features in subsequent frames
        prev_descriptors = des[start_frame]
        prev_features = current_features

        if prev_descriptors is None:
            continue  # Skip frames without descriptors

        for frame_idx in range(start_frame + 1, len(images)):
            current_descriptors = des[frame_idx]

            if current_descriptors is None:
                continue  # Skip frames without descriptors

            # Ensure descriptors are of the same type
            if prev_descriptors.dtype != np.float32:
                prev_descriptors = prev_descriptors.astype(np.float32)
            if current_descriptors.dtype != np.float32:
                current_descriptors = current_descriptors.astype(np.float32)

            matches = bf.knnMatch(prev_descriptors, current_descriptors, k=2)

            # Apply Lowe’s ratio test
            good_matches = []
            for match in matches:
                if len(match) >= 2:
                    m, n = match[:2]
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            # Update tracks with good matches
            new_features = {}
            for match in good_matches:
                query_idx = match.queryIdx  # Previous frame feature
                train_idx = match.trainIdx  # Current frame feature
                
                if query_idx in prev_features:
                    track_id = prev_features[query_idx]
                    tracks[track_id].append((frame_idx, train_idx))
                    new_features[train_idx] = track_id  # Carry forward to next frame

            # Prepare for the next frame
            prev_descriptors = current_descriptors
            prev_features = new_features

    plt.figure(figsize=(24, 12))
    for track_id, track in tqdm(tracks.items()):
        frames = [f for f, _ in track]
        plt.plot([track_id] * len(frames), frames, marker='o', linestyle='-', lw=0.05)


    plt.xlabel('Feature Track ID ')
    plt.ylabel('Frame ID')
    plt.title(f'Feature Tracks Over Multiple Frames - {name}')
    plt.gca().invert_yaxis() 
    plt.savefig(f'nh_data/nighthawk_{config_name}_{name}_tracks.png')
    plt.close()

    with open(f'nh_data/nighthawk_{config_name}_{name}_tracks.pkl', 'wb') as f:
        pickle.dump(tracks, f)

    # write csv
    with open(f'nh_data/nighthawk_{config_name}_{name}.csv', 'w') as f:
        f.write('frame_id,matched_kps,num_inliers,px_shifts\n')
        for i in range(len(matched_kps)):
            f.write(f'{i*dataset.skip},{matched_kps[i]},{num_inliers[i]},{px_shifts[i]}\n')


if __name__ == "__main__":
    # set PYSLAM_CONFIG environment variable to the path of the settings file
    feature_dict = {
        # "LK_SHI_TOMASI": FeatureTrackerConfigs.LK_SHI_TOMASI,
        # "LK_FAST": FeatureTrackerConfigs.LK_FAST,
        # "ORB": FeatureTrackerConfigs.ORB,
        # "BRISK": FeatureTrackerConfigs.BRISK,
        # "AKAZE": FeatureTrackerConfigs.AKAZE,
        "SIFT": FeatureTrackerConfigs.SIFT,
        # "SUPERPOINT": FeatureTrackerConfigs.SUPERPOINT,
        # "R2D2": FeatureTrackerConfigs.R2D2
    }
    for key,val in feature_dict.items():
        # try:
        run_exp(key,val,70)
        # except Exception as e:
        #     print(f'Error in {key}: {e}')
        #     pass

    
