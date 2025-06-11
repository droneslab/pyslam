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

import traceback

bf = cv2.BFMatcher(cv2.NORM_L2)

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kResultsFolder = kRootFolder + '/results'

global_plotting = False

kUseRerun = True
# check rerun does not have issues 
if global_plotting:
    if kUseRerun and not Rerun.is_ok():
        kUseRerun = False
    
"""
use or not pangolin (if you want to use it then you need to install it by using the script install_thirdparty.sh)
"""
if global_plotting:
    kUsePangolin = False  
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
    
def process_data(results: dict, images=None,masks=None, draw_tracks=False, plot_traj=True, traj_skips=1):

    exp_name = results['exp_name']
    name = results['feature_name']
    matched_kps = results['matched_kps']
    num_inliers = results['num_inliers']
    px_shifts = results['px_shifts']
    dataset_skip = results['dataset_skip']
    rs = results['rs']
    ts = results['ts']
    kps = results['kps']
    des = results['des']        
    original_kps = results['original_kps']
    masked_kps = results['masked_kps']
    xs = results['xs']
    ys = results['ys']
    zs = results['zs']
    gtxs = results['gtxs']
    gtys = results['gtys']
    gtzs = results['gtzs']
    est_times = results['est_times']

    print_str = f'''
        matched_kps: {np.mean(matched_kps)}
        num_inliers: {np.mean(num_inliers)}
        px_shifts: {np.mean(px_shifts)}
        rs: {len(rs)}
        ts: {len(ts)}
        xs: {len(xs)}
        ys: {len(ys)}
        zs: {len(zs)}
        gtxs: {len(gtxs)}
        gtys: {len(gtys)}
        gtzs: {len(gtzs)}
        kps: {len(kps)}
        des: {len(des)}
        images: {len(images)}
        masks: {len(masks)},
        original_kps: {np.sum([len(kp) for kp in original_kps])/len(original_kps)}
        masked_kps: {np.sum([len(kp) for kp in masked_kps])/len(masked_kps)}
        est_times: {np.sum(est_times)/len(est_times)}
            '''
    print(print_str)
    with open(f'{kResultsFolder}/{exp_name}_stats.txt', 'w') as f:
        f.write(print_str)

    if draw_tracks:
        idxs = range(len(matched_kps))
        idxs = [i*dataset_skip for i in idxs]
        fig, ax = plt.subplots(2,1)
        ax[0].plot(idxs, matched_kps, label='matched_kps')
        ax[0].plot(idxs, num_inliers, label='num_inliers')
        ax[0].legend()
        ax[1].plot(idxs, px_shifts, label='px_shifts')
        ax[1].legend()
        plt.savefig(f'{kResultsFolder}/{exp_name}_kp_inliers.png')
        plt.close()

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
        plt.savefig(f'{kResultsFolder}/{exp_name}_tracks.png')
        plt.close()

        with open(f'{kResultsFolder}/{exp_name}_tracks.pkl', 'wb') as f:
            pickle.dump(tracks, f)

        

    if plot_traj:
        xs_p = xs[::traj_skips]
        zs_p = zs[::traj_skips]
        gtxs_p = gtxs[::traj_skips]
        gtzs_p = gtzs[::traj_skips]
        plt.figure(figsize=(12, 12))
        plt.plot(xs_p, zs_p, c='tab:red', label='estimated')
        plt.plot(gtxs_p, gtzs_p, c='tab:green', label='ground truth')
        plt.scatter(xs_p, zs_p, c='tab:red', s=2)
        plt.scatter(gtxs_p, gtzs_p, c='tab:green', s=2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(f'2D Trajectory - {name}')
        plt.legend()
        plt.savefig(f"{kResultsFolder}/{exp_name}_2d.png")
        plt.close()

        with open(f'{kResultsFolder}/{exp_name}_trajectory.csv', 'w') as f:
            print(f"saving {kResultsFolder}/{exp_name}_2d.csv with {len(xs)} points")
            f.write(f'i,x,y,z,gtx,gty,gtz\n')
            for i in range(len(xs)):
                # f.write(f'{xs[i]},{ys[i]},{zs[i]},{gtxs[i]},{gtys[i]},{gtzs[i]}\n')
                f.write(f'{i},{xs[i]},{ys[i]},{zs[i]},{gtxs[i]},{gtys[i]},{gtzs[i]}\n')

    return
    

def run_exp(name, feature, feature_num=2000, max_images=10, top_k = 85, mask_name='mc_trials_50', baseline=False, save_intermediate=False, plot_tracks=False, plot_traj=True):
    config_loc = os.environ.get('PYSLAM_CONFIG')
    config_name = config_loc.split('.')[0]
    config = Config()
    setattr(config, 'top_k', top_k)
    setattr(config, 'mask_name', mask_name)

    # exp_name = f'{config_name}_^{name}^_@{feature_num}@_${"baseline" if baseline else "masked"}$'
    exp_name = f'{config_name}_^{name}^_@{feature_num}@_${"baseline" if baseline else "masked"}$_#{top_k}#_*{mask_name}*'
    
    dataset = dataset_factory(config)
    setattr(dataset, 'skip', 1)

    groundtruth = groundtruth_factory(config.dataset_settings)

    cam = PinholeCamera(config)

    num_features=feature_num  # how many features do you want to detect and track?
    # if config.num_features_to_extract > 0:  # override the number of features to extract if we set something in the settings file
    #     num_features = config.num_features_to_extract
        
    # select your tracker configuration (see the file feature_tracker_configs.py) 
    # LK_SHI_TOMASI, LK_FAST
    # SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, LIGHTGLUE, XFEAT, XFEAT_XFEAT, LOFTR
    tracker_config = feature
    tracker_config['num_features'] = num_features
   
    feature_tracker = feature_tracker_factory(**tracker_config)

    if save_intermediate:
        save_loc = kResultsFolder
    else:
        save_loc = None

    # create visual odometry object 
    if dataset.sensor_type == SensorType.RGBD:
        vo = VisualOdometryRgbdTensor(cam, groundtruth)  # only for RGBD
        Printer.green('Using VisualOdometryRgbdTensor')
    else:
        vo = VisualOdometryEducational(cam, groundtruth, feature_tracker,save_loc=save_loc)
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
    xs = []
    ys = []
    zs = []
    gtxs = []
    gtys = []
    gtzs = []
    img_id = 0
    images = []
    masks = []
    original_kps = []
    masked_kps = []
    est_times = []
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
            matched_kp, num_inlier, px_shift, kp_cur, des_cur,rot,trans, kp_before, kp_after, est_time = vo.track(img, depth, img_id, timestamp, mask=mask)  # main VO function 
            if matched_kp is not None:
                images.append(img)
                matched_kps.append(matched_kp)
                num_inliers.append(num_inlier)
                px_shifts.append(px_shift)
                kps.append(kp_cur)
                des.append(des_cur)
                rs.append(rot)
                ts.append(trans)
                original_kps.append(kp_before)
                masked_kps.append(kp_after)
                est_times.append(est_time)

            if(len(vo.traj3d_est)>=1):	       # start drawing from the third image (when everything is initialized and flows in a normal way)

                x, y, z = vo.traj3d_est[-1]
                gt_x, gt_y, gt_z = vo.traj3d_gt[-1]
                xs.append(x)
                ys.append(y)
                zs.append(z)
                gtxs.append(gt_x)
                gtys.append(gt_y)
                gtzs.append(gt_z)
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
    
    results = {
        'exp_name': exp_name,
        'feature_name': name,
        'dataset_skip': dataset.skip,
        'matched_kps': matched_kps,
        'num_inliers': num_inliers,
        'px_shifts': px_shifts,
        'rs': rs,
        'ts': ts,
        'kps': kps,
        'des': des,
        'xs': xs,
        'ys': ys,
        'zs': zs,
        'gtxs': gtxs,
        'gtys': gtys,
        'gtzs': gtzs,
        'original_kps': original_kps,
        'masked_kps': masked_kps,
        'est_times': est_times,
    }

    with open(f'{kResultsFolder}/{exp_name}.pkl', 'wb') as f:
        pickle.dump(results, f)

    process_data(results, images=images, masks=images, draw_tracks=plot_tracks, plot_traj=plot_traj, traj_skips=20)

    return


if __name__ == "__main__":
    # set PYSLAM_CONFIG environment variable to the path of the settings file
    # feature_dict = {
    #     # "LK_SHI_TOMASI": FeatureTrackerConfigs.LK_SHI_TOMASI,
    #     # "LK_FAST": FeatureTrackerConfigs.LK_FAST,
    #     # "ORB": FeatureTrackerConfigs.ORB,
    #     # "BRISK": FeatureTrackerConfigs.BRISK,
    #     # "AKAZE": FeatureTrackerConfigs.AKAZE,
    #     "SIFT": FeatureTrackerConfigs.SIFT,
    #     # "SUPERPOINT": FeatureTrackerConfigs.SUPERPOINT,
    #     # "R2D2": FeatureTrackerConfigs.R2D2
    # }
    # for key,val in feature_dict.items():
    #     # try:
    #     run_exp(key,val,70)
    #     # except Exception as e:
    #     #     print(f'Error in {key}: {e}')
    #     #     pass

    features = [
        ['LK_SHI_TOMASI', FeatureTrackerConfigs.LK_SHI_TOMASI],
        ['LK_FAST', FeatureTrackerConfigs.LK_FAST],
        ['ORB', FeatureTrackerConfigs.ORB],
        ['BRISK', FeatureTrackerConfigs.BRISK],
        ['AKAZE', FeatureTrackerConfigs.AKAZE],
        ['SIFT', FeatureTrackerConfigs.SIFT],
        # ['SUPERPOINT', FeatureTrackerConfigs.SUPERPOINT],
        # ['R2D2', FeatureTrackerConfigs.R2D2],
        # ['LIGHTGLUE', FeatureTrackerConfigs.LIGHTGLUE],
        # ['XFEAT', FeatureTrackerConfigs.XFEAT],
        # ['XFEAT_XFEAT', FeatureTrackerConfigs.XFEAT_XFEAT],
        # ['LOFTR', FeatureTrackerConfigs.LOFTR]
    ]

    feature_nums = [
        3000,
        2000,
        1500,
        1000,
        500,
        400,
        100
    ]

    baselines = [
        True,
        False,
    ]

    # top_ks = np.arange(100, -1, -20, dtype=int).tolist()
    top_ks = [0,25,33,50,66]

    mask_loc = [
        # 'mc_trials_50',
        'moped_uh_25000_mse_mc100_iter25000'
    ]

    max_images = 1000

    for mask_l in mask_loc:
        for f in features:
            for num in feature_nums:
                for baseline in baselines:
                    for k in top_ks:
                        try:
                            if not baseline and k >= 0:
                                run_exp(f[0], f[1], num, max_images, k, mask_l, baseline, save_intermediate=False, plot_tracks=False)
                            elif baseline and k==0:
                                run_exp(f[0], f[1], num, max_images, k, mask_l, baseline, save_intermediate=False, plot_tracks=False)
                            else:
                                print(f'Skipping {f[0]} with num {num} and top_k {k} for baseline {baseline}')
                        except Exception as e:
                            config= os.environ.get('PYSLAM_CONFIG')
                            config_name = config.split('.')[0]
                            with open(f'{kResultsFolder}/{config_name}_{f[0]}_{num}_{"baseline" if baseline else "masked"}_{k}_{mask_l}.txt', 'w') as wf:
                                wf.write(traceback.format_exc())
