import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import glob
import imageio
import numpy as np
import cv2
from utils.util import get_image_to_tensor_balanced, coordinate_transformation
from utils.data_augmentation import get_transformation


class SIASUNDataModule:
    def __init__(self, cfg):
        self.batch_size = cfg["batch_size"]
        self.shuffle = cfg["shuffle"]
        self.num_workers = cfg["num_workers"]

        self.dataset_cfg = cfg["dataset"]
        self.data_augmentation = cfg["data_augmentation"]

    def load_dataset(self, mode, use_data_augmentation=False, scene_list=None):
        self.mode = mode
        self.dataset_cfg["mode"] = mode
        self.dataset_cfg["scene_list"] = scene_list

        if use_data_augmentation:
            self.dataset_cfg["transformation"] = self.data_augmentation
        else:
            self.dataset_cfg["transformation"] = None

        return SIASUNDataset.init_from_cfg(self.dataset_cfg)

    def get_dataloader(self, dataset):
        batch_size = self.batch_size
        shuffle = self.shuffle
        num_workers = self.num_workers

        if self.mode == "test":
            batch_size = 1
            shuffle = False
            num_workers = 0

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        return dataloader


class SIASUNDataset(Dataset):
    def __init__(
        self,
        mode,
        data_rootdir,
        max_imgs,
        image_size,
        z_near,
        z_far,
        trans_cfg,
        dataset_format,
        scene_list,
    ):
        """
        Inits DTU dataset instance

        Args:
        mode: either train, val or test
        data_rootdir: root directory of dataset
        max_imgs: maximal images for the object
        image_size: [H, W] pixels
        z_near: minimal distance of the object
        z_far: maximal distance of the object
        trans_cfg: configurations for data augmentations(transformation)
        dataset_formate: the coordinate system the original dataset uses
        """

        super().__init__()
        self.max_imgs = max_imgs
        self.image_size = image_size
        self.z_near = z_near
        self.z_far = z_far
        self.dataset_format = dataset_format
        self.rootdir=data_rootdir

        self.transformations = []
        if trans_cfg is not None:
            self.transformations = get_transformation(trans_cfg)

        assert os.path.exists(data_rootdir)
        file_list = os.path.join(data_rootdir, f"{mode}.lst")
        assert os.path.exists(file_list)
        base_dir = os.path.dirname(file_list)
        if scene_list is None:
            with open(file_list, "r") as f:
                self.scene_list = [x.strip() for x in f.readlines()]
        else:
            self.scene_list = [f"scan{x}" for x in scene_list]

        self.objs_path = [os.path.join(base_dir, scene) for scene in self.scene_list]

        self.image_to_tensor = get_image_to_tensor_balanced()

    def __len__(self):
        return len(self.objs_path)

    def __getitem__(self, index):
        scan_name = self.scene_list[index]
        root_dir = self.objs_path[index]
        rgb_paths = [
            x
            for x in glob.glob(os.path.join(root_dir, "images", "*"))
            if (x.endswith(".jpg") or x.endswith(".png"))
        ]

        rgb_paths = sorted(rgb_paths)
        #print(len(rgb_paths))
        #print(self.max_imgs)
        if len(rgb_paths) <= self.max_imgs:
            sel_indices = np.arange(len(rgb_paths))
        else:
            sel_indices = np.random.choice(len(rgb_paths), self.max_imgs, replace=False)
            rgb_paths = [rgb_paths[i] for i in sel_indices]

        transforms=[]

        poses_folder=os.path.join(self.rootdir, scan_name, "poses")

        Fx=396
        Fy=396
        Cx=240
        Cy=135
        for file_name in os.listdir(poses_folder):
            if file_name.endswith('.txt'):
                file_path = os.path.join(poses_folder, file_name)
                # 读取txt文件中的位姿数据
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    pose_matrix = np.zeros((4, 4))
                    for i, line in enumerate(lines):
                        values = [float(val) for val in line.strip().split()]
                        pose_matrix[i, :] = values

                # 将位姿数据矩阵添加到列表中
                transforms.append(pose_matrix)

        #cam_path = os.path.join(root_dir, "cameras.npz")
        #all_cam = np.load(cam_path)
        all_imgs = []
        all_poses = []
        focal = None
        fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0

        for idx, rgb_path in enumerate(rgb_paths):
            i = sel_indices[idx]
            img = imageio.imread(rgb_path)[..., :3]

            # decompose projection matrix
            P = transforms[i]
            fx += Fx
            fy += Fy
            cx += Cx
            cy += Cy

            pose = np.eye(4, dtype=np.float32)
            pose = P


            #scale_mtx = all_cam.get("scale_mat_" + str(i))
            #if scale_mtx is not None:
                #norm_trans = scale_mtx[:3, 3:]
                #norm_scale = np.diagonal(scale_mtx[:3, :3])[..., None]
                #pose[:3, 3:] -= norm_trans
                #pose[:3, 3:] /= norm_scale

            # camera poses in world coordinate
            pose = coordinate_transformation(pose, format=self.dataset_format)
            img_tensor = self.image_to_tensor(img)
            all_imgs.append(img_tensor)
            all_poses.append(pose)#pose is c2w

        # get average intrinsics for one object
        fx /= len(rgb_paths)
        fy /= len(rgb_paths)
        cx /= len(rgb_paths)
        cy /= len(rgb_paths)
        focal = torch.tensor((fx, fy), dtype=torch.float32)
        c = torch.tensor((cx, cy), dtype=torch.float32)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        print("all_img_shape")
        print(all_imgs.shape)
        # resize images if given image size is not euqal to original size
        if np.any(np.array(all_imgs.shape[-2:]) != self.image_size):
            scale_h = self.image_size[0] / all_imgs.shape[-2]
            scale_w = self.image_size[1] / all_imgs.shape[-1]
            print(self.image_size[0])
            print(all_imgs.shape[-2])
            print(self.image_size[1])
            print(all_imgs.shape[-1])
            focal[0] *= scale_w
            focal[1] *= scale_h
            c[0] *= scale_w
            c[1] *= scale_h
            print("all_img_shape")
            print(all_imgs.shape)
            print(type(self.image_size))
            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            print(all_imgs.shape)
        # aplly data augmentations
        for transformer in self.transformations:
            all_imgs = transformer(all_imgs)
        data_instance = {
            "scan_name": scan_name,
            "path": root_dir,
            "img_id": index,
            "focal": focal,
            "c": c,
            "images": all_imgs,
            "poses": all_poses,
        }
        return data_instance

    @classmethod
    def init_from_cfg(cls, cfg):
        return cls(
            mode=cfg["mode"],
            data_rootdir=cfg["data_rootdir"],
            max_imgs=cfg["max_imgs"],
            image_size=cfg["image_size"],
            z_near=cfg["z_near"],
            z_far=cfg["z_far"],
            trans_cfg=cfg["transformation"],
            dataset_format=cfg["format"],
            scene_list=cfg["scene_list"],
        )
