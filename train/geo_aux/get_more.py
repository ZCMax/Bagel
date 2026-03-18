import os
import cv2
import json
import numpy as np
import pickle
from PIL import Image
import torch
from typing import Dict, List, Sequence, Tuple
import torchvision
data_dir = '/mnt/petrelfs/linjingli/UMM_Spatial/vis/img_root'
embodiedscan_info_dir = '/mnt/petrelfs/linjingli/mmscan_db/mmscan_data/embodiedscan_info'
def read_image_cv2_local(path: str, rgb: bool = True) -> np.ndarray:
    """
    Reads an image from disk using OpenCV, returning it as an RGB image array (H, W, 3).

    Args:
        path (str):
            File path to the image.
        rgb (bool):
            If True, convert the image to RGB.
            If False, leave the image in BGR/grayscale.

    Returns:
        np.ndarray or None:
            A numpy array of shape (H, W, 3) if successful,
            or None if the file does not exist or could not be read.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"File does not exist or is empty: {path}")
        return None

    img = cv2.imread(path)
    if img is None:
        print(f"Could not load image={path}. Retrying...")
        img = cv2.imread(path)
        if img is None:
            print("Retry failed.")
            return None

    if rgb:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
def get_dl3dv(image_name):
    def convert_intrinsics(meta_data):
        store_h, store_w = meta_data["h"], meta_data["w"]
        fx, fy, cx, cy = (
            meta_data["fl_x"],
            meta_data["fl_y"],
            meta_data["cx"],
            meta_data["cy"],
        )
        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = float(fx) / 4.0 # downsample by 4
        intrinsics[1, 1] = float(fy) / 4.0
        intrinsics[0, 2] = float(cx) / 4.0
        intrinsics[1, 2] = float(cy) / 4.0
        return intrinsics
    
    raw_dl3dv_root = '/mnt/inspurfs/efm_t/huwenbo/hoss_datasets/dl3dv/DL3DV-10K'

    video_id = image_name.split('/')[1]+'/'+image_name.split('/')[2]

    base_path = os.path.join(raw_dl3dv_root, video_id)
    json_path = os.path.join(base_path, "transforms.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    item_intrinsic = np.array(convert_intrinsics(data).tolist()).astype(np.float32)
    item_pose = np.load(data_dir+'/'+image_name.split('.')[0]+'.npy').astype(np.float32)

    depth_path = os.path.join(base_path, "dense", "depth", image_name.split('/')[-1][:-4] + ".npy")
    sky_mask_path = os.path.join(base_path, "dense", "sky_mask", image_name.split('/')[-1][:-4] + ".png")
    outlier_mask_path = os.path.join(base_path, "dense", "outlier_mask", image_name.split('/')[-1][:-4] + ".png")
    rgb_image = read_image_cv2_local(data_dir+'/'+image_name).astype(np.float32)
    depthmap = np.load(depth_path).astype(np.float32)
    depthmap[~np.isfinite(depthmap)] = 0  # invalid
    with Image.open(sky_mask_path) as img:
        sky_mask = np.array(img).astype(np.float32)
    sky_mask = sky_mask >= 127

    with Image.open(outlier_mask_path) as img:
        outlier_mask = np.array(img).astype(np.float32)
    depthmap[sky_mask] = -1.0
    depthmap[outlier_mask >= 127] = 0.0
    depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
    threshold = (
        np.percentile(depthmap[depthmap > 0], 98)
        if depthmap[depthmap > 0].size > 0
        else 0
    )
    depthmap[depthmap > threshold] = 0.0
    H, W = rgb_image.shape[:2]
    item_rgb = rgb_image.astype(np.float32)
    item_depthmap = cv2.resize(depthmap, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return item_intrinsic,item_pose,item_rgb,item_depthmap


def get_scannet(image_name):
    raw_scannet_dir = '/mnt/inspurfs/mozi_t/linjingli/transfer/ScanNet_v2/posed_images'

    scan_id = image_name.split('/')[1]
    info_file_path = os.path.join(embodiedscan_info_dir,scan_id+'.pkl')
    with open(info_file_path, "rb") as f:
        data = np.load(f, allow_pickle=True)
    item_intrinsic = np.array(data["data_list"][0]["cam2img"]).astype(np.float32)
    item_pose = np.load(data_dir+'/'+image_name.split('.')[0]+'.npy').astype(np.float32)


    image_path =image_name.replace('scannet',raw_scannet_dir)
    depth_path = image_path.replace('jpg','png')
    rgb_image = read_image_cv2_local(image_path, rgb=True)
    if rgb_image is None:
        raise FileNotFoundError(f"Failed to load RGB image: {image_path}")
    rgb_image = rgb_image.astype(np.float32)
    depth_raw = cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)
        
    depthmap = depth_raw.astype(np.float32) / 1000.0
    depthmap[~np.isfinite(depthmap)] = 0

    # Resize depth map to match image dimensions
    if depthmap.shape[0] != rgb_image.shape[0] or depthmap.shape[1] != rgb_image.shape[1]:
        depthmap = cv2.resize(
            depthmap, 
            (rgb_image.shape[1], rgb_image.shape[0]),  # (width, height) for OpenCV
            interpolation=cv2.INTER_NEAREST  # Better for depth maps to preserve sharpness
        )

    depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
    threshold = (
        np.percentile(depthmap[depthmap > 0], 98)
        if depthmap[depthmap > 0].size > 0
        else 0
    )
    depthmap[depthmap > threshold] = 0.0

    item_rgb = rgb_image
    item_depthmap = depthmap
    return item_intrinsic,item_pose,item_rgb,item_depthmap

def get_matterport3d(image_name):
    
    raw_matterport3_dir = "/mnt/inspurfs/mozi_t/linjingli/transfer/matterport3d/scans"
    NEW_PKL_DIR = "/mnt/inspurfs/mozi_t/linjingli/tmp_data_mp3d"
    mp3d_dict = json.load(open('/mnt/petrelfs/linjingli/MMScan_code/data_preparation/meta-data/mp3d_mapping.json'))
    mp3d_dict = {mp3d_dict[key_]:key_ for key_ in mp3d_dict}

    video_id = image_name.split('/')[1]

    pkl_data = pickle.load(open(f'{NEW_PKL_DIR}/{video_id}.pkl','rb'))

    scan_name = mp3d_dict[video_id.split('_')[0]+'_'+video_id.split('_')[1]]

    image_paths = pkl_data['image_paths']
    depth_paths = pkl_data['depth_image_paths']
    

    index_ = image_paths.index(image_name.replace(video_id,scan_name+'/matterport_color_images'))
    image_path = image_paths[index_].replace('matterport3d',raw_matterport3_dir)
    depth_path = depth_paths[index_].replace('matterport3d',raw_matterport3_dir)
    item_rgb = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    depth_raw = cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)
    item_depthmap = depth_raw.astype(np.float32) / 4000.0

    item_intrinsic = pkl_data['intrinsics'][index_]
    item_pose = np.load(data_dir+'/'+image_name.split('.')[0]+'.npy').astype(np.float32)
    return item_intrinsic,item_pose,item_rgb,item_depthmap

def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    # assert camera_intrinsics[0, 1] == 0.0
    # assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    # Invalid any depth > 80m
    valid_mask = valid_mask
    return X_cam, valid_mask
def _get_by_image_name(image_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = image_name.split('/')[0].lower()
    if dataset == 'dl3dv':
        return get_dl3dv(image_name)
    if dataset == 'scannet':
        return get_scannet(image_name)
    if dataset == 'matterport3d':
        return get_matterport3d(image_name)
    raise ValueError(f"Unsupported dataset prefix in image_name: {image_name}")
def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, z_far=0, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)
    if z_far > 0:
        valid_mask = valid_mask & (depthmap < z_far)

    X_world = X_cam # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask
def save_ply_visualization_pi3_cpu_batch(pred_dict, cur_step, save_name, init_conf_threshold=20.0, save_depth=True, save_point=False, save_gt=True, save_images=True, debug=False, all_batch=False,gt_only=False):
    seq_len = len(pred_dict["images"])  
    import open3d as o3d

    if save_depth:
       
        gt_pts = np.array(pred_dict['world_points'])#.float().cpu().numpy()     # (N, H, W, 3)
        
        valid_mask= pred_dict['point_masks']#.cpu().numpy()   # (N, H, W)
        valid_mask = np.array(valid_mask)
        # print('valid_mask', valid_mask.shape)
        images = pred_dict["images"]  # (S, 3, H, W) # #(10, 224, 224, 3)
        images = torch.from_numpy(np.stack(images).astype(np.float32)).contiguous().permute(0, 3, 1, 2).to(torch.float32).div(255)
        colors = images.permute(0, 2, 3, 1)[valid_mask].numpy().reshape(-1, 3)
        # colors = images[valid_mask].reshape(-1, 3)
        gt_pts = gt_pts[valid_mask].reshape(-1, 3)

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
        pcd_gt.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud('results/'+save_name+'.ply', pcd_gt)

    if save_images:
        # Save images as a grid
        images = np.array(pred_dict["images"])
        images = torch.from_numpy(np.stack(images).astype(np.float32)).contiguous().permute(0, 3, 1, 2).to(torch.float32).div(255)

        images_grid = torchvision.utils.make_grid(
            images, nrow=8, normalize=False, scale_each=False
        )
        torchvision.utils.save_image(
            images_grid, 'results/'+save_name+'.png', normalize=False, scale_each=False
        )

# image_list = [
#     'scannet/scene0000_00/00000.jpg',
#     "scannet/scene0000_00/00020.jpg",
#     "scannet/scene0000_00/00040.jpg",
#     "scannet/scene0000_00/00060.jpg",
#     "scannet/scene0000_00/00180.jpg", 
#     "scannet/scene0000_00/00010.jpg",
# ]

# images = []
# depths = []
# extrinsics = []
# intrinsics = []
# for image_name in image_list:
#     item_intrinsic,item_pose,item_rgb,item_depthmap = get_scannet(image_name)

#     images.append(item_rgb)
#     depths.append(item_depthmap)
#     extrinsics.append(item_pose)
#     intrinsics.append(item_intrinsic)

# world_points = []
# point_masks = []
# for v,(img,depthmap,camera_pose, camera_intrinsics) in enumerate(zip(images,depths,extrinsics,intrinsics)):

#     z_far = 0
#     pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, z_far=z_far)

#     valid_mask = valid_mask & np.isfinite(pts3d).all(axis=-1)
#     depthmap[~valid_mask] = 0.0

#     world_points.append(pts3d)
#     point_masks.append(valid_mask)

# predictions = {}
# predictions['world_points'] = world_points
# predictions['point_masks'] = point_masks
# predictions['images'] = images
# predictions['view_infos'] = ''


# save_ply_visualization_pi3_cpu_batch(predictions, 0, 'test_it', gt_only=True)
