import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
import os.path as osp

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

class CameraParams:
    def __init__(self, near, far, pose_scale, pose_scale2, move_all_cam_vec):
        self.near = near
        self.far = far
        self.pose_scale = pose_scale
        self.pose_scale2 = pose_scale2
        self.move_all_cam_vec = move_all_cam_vec
def load_blender_data_Cam(datadir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    base_dir, scene = osp.split(datadir)

    world_setup_fn = osp.join(base_dir, scene) + '/world_setup.json'

    # read json file
    with open(world_setup_fn, 'r') as myfile:
        data = myfile.read()

    # parse json file
    obj = json.loads(data)
    near = obj['near']
    far = obj['far']
    pose_scale = obj['pose_scale']
    pose_scale2 = obj['pose_scale2']
    move_all_cam_vec = obj['move_all_cam_vec']

    camera_params = CameraParams(near, far, pose_scale, pose_scale2, move_all_cam_vec)


    all_imgs = []
    all_poses = []
    counts = [0]

    for s in splits:
        root_dir = os.path.join(datadir,s)
        rgb_dir = root_dir + '/rgb/'
        pose_dir = root_dir + '/poses/'
        if s=='train' or testskip==0:
            skip = 4
        else:
            skip = testskip


        rgb_files = os.listdir(rgb_dir)
        rgb_files = [rgb_dir + f for f in rgb_files]
        rgb_files.sort()

        pose_files = os.listdir(pose_dir)
        pose_files = [pose_dir + f for f in pose_files]
        pose_files.sort()

        if scene == 'ShopFacade' and s == 'train' :
            del rgb_files[42]
            del rgb_files[35]
            del pose_files[42]
            del pose_files[35]
        if len(rgb_files) != len(pose_files):
            raise Exception('RGB file count does not match pose file count!')

        # trainskip and testskip
        frame_idx = np.arange(len(rgb_files))
        if s == 'train' and skip > 1 :
            frame_idx_tmp = frame_idx[::skip]
            frame_idx = frame_idx_tmp
        elif s != 'train' and testskip > 1:
            frame_idx_tmp = frame_idx[::testskip]
            frame_idx = frame_idx_tmp
        gt_idx = frame_idx

        rgb_files = [rgb_files[i] for i in frame_idx]
        pose_files = [pose_files[i] for i in frame_idx]

        if len(rgb_files) != len(pose_files):
            raise Exception('RGB file count does not match pose file count!')
        imgs = []
        # read poses
        poses = []
        for i in range(len(pose_files)):
            pose = np.loadtxt(pose_files[i])
            poses.append(pose)
            image = imageio.imread(rgb_files[i])
            if image.shape[-1] == 3:
                alpha_channel = np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype) * 255
                image = np.concatenate((image, alpha_channel), axis=-1)
            imgs.append(image)

        poses = np.array(poses).astype(np.float32)  # [N, 4, 4]
        all_poses.append(poses)
        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)


    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    [H, W, focal] = [480, 854, 744.]
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split, camera_params


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        print(imgs.shape)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split
if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    datadir="./data/nerf_synthetic/lego"
    datadir2 = "./data/Cambridge/GreatCourt"
    imgs, poses, render_poses, [H, W, focal], i_split = load_blender_data(datadir, half_res=False, testskip=1)
    #imgs, poses, render_poses, [H, W, focal], i_split = load_blender_data_Cam(datadir2, half_res=False, testskip=1)
    #print(i_split[0])
    #print(poses.shape)
    #print(render_poses.shape)