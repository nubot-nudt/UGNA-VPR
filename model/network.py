from model.encoder import Encoder
from model.mlp import MLPFeature, MLPOut
import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
from utils import util
import numpy as np
from dotmap import DotMap
from networks.MIxVPR import TeacherNet


def pose_tensor_to_pose_representations(pose_tensor):
    # 获取张量的形状
    num_poses, _, _ = pose_tensor.shape
    pose_tensor = pose_tensor.cpu()
    # 初始化结果数组
    pose_representations_euler = np.zeros((num_poses, 6))

    for i in range(num_poses):
        # 提取位置信息
        position = pose_tensor[i, :3, 3]
        # 提取旋转信息
        rotation_matrix = pose_tensor[i, :3, :3]
        # 欧拉角表示
        euler_angles = np.array([0, 0, 0])
        euler_angles = np.degrees(np.around(np.array([
            np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]),
            np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2)),
            np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        ]), decimals=6))
        pose_representations_euler[i] = np.concatenate((position, euler_angles))
    pose_representations_euler = torch.tensor(pose_representations_euler)
    # print(pose_tensor.device)
    return pose_representations_euler


def relative_poses(pose1, poses_ref):
    # pose1   (ON,N-RN,4, 4)  c2w
    # poses_ref (ON,N-RN,4, 4)  w2c
    # Invert pose1 to get pose from world to camera
    pose1_inv = torch.inverse(pose1)

    # Compute relative poses
    relative_poses = torch.matmul(pose1_inv.unsqueeze(2), poses_ref.unsqueeze(1))

    return relative_poses


def transform_images_to_reference_image_coordinates(pose1, poses_ref, focal, c, focal_ref, c_ref):
    """
    Transform points from the coordinate system of image1 to the coordinate system of reference images.

    Args:
        image1: Tensor of shape (ON, X-RN, H, W) representing the first image.
        image_ref: Tensor of shape (ON, RN, H, W) representing the reference images.
        pose1: Tensor of shape (ON, X-RN, 4, 4) representing the c2w pose of the first camera.
        poses_ref: Tensor of shape (ON, RN, 4, 4) representing the w2c poses of reference cameras.
        focal: Tensor of shape (ON, 2) representing the focal lengths of the first cameras.
        c: Tensor of shape (ON, 2) representing the principal points of  the first cameras.
        focal_ref: Tensor of shape (ON, 2) representing the focal lengths of reference cameras.
        c_ref: Tensor of shape (ON, 2) representing the principal points of reference cameras.

    Returns:
        uv_feats: Transformed points in the coordinate system of reference images, compatible with F.grid_sample.
    """
    ON, X_RN, _, _ = pose1.shape
    _, RN, _, _ = poses_ref.shape
    # print(RN)
    H = 16  # 需要根据网络修改
    W = 16
    focal = focal.unsqueeze(1).expand(ON, X_RN, 2)  # (ON, X_RN, 2)
    c = c.unsqueeze(1).expand(ON, X_RN, 2)
    focal_ref = focal_ref.unsqueeze(1).expand(ON, RN, 2)  # (ON, RN, 2)
    c_ref = c_ref.unsqueeze(1).expand(ON, RN, 2)
    # Generate pixel grid for image1
    pixel_grid1 = generate_pixel_grid(H, W, device=pose1.device)  # (H, W, 2)
    pixel_grid1 = pixel_grid1.to(focal.device)
    # Convert pixel grid to camera coordinates for image1
    camera_coords1 = pixel_to_camera(pixel_grid1, focal, c)  # (ON, X-RN, H, W, 3)

    # Convert camera coordinates to world coordinates for image1
    world_coords1 = camera_to_world(camera_coords1, pose1)  # (ON, N-RN, H*W, 3)

    # Convert world coordinates to pixel coordinates for reference images
    pixels_ref = world_to_pixel(world_coords1, poses_ref, focal_ref, c_ref)
    # print(pixels_ref.shape)
    # Normalize pixel coordinates to [-1, 1] range
    uv_feats = normalize_coordinates(pixels_ref, W, H)

    return uv_feats


def generate_pixel_grid(H, W, device):
    """
    Generate a pixel grid.

    Args:
        H: Height of the image.
        W: Width of the image.
        device: Device to place the tensor on.

    Returns:
        pixel_grid: Tensor of shape (H, W, 2) representing the pixel grid.
    """
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    pixel_grid = torch.stack((x, y), dim=-1)
    return pixel_grid


def pixel_to_camera(pixel_grid, focal, c):
    """
    Convert pixel coordinates to camera coordinates.

    Args:
        pixel_grid: Tensor of shape (H, W, 2) representing the pixel grid.
        pose1: Tensor of shape (ON, X-RN, 4, 4) representing the c2w pose of the first camera.
        focal: Tensor of shape (ON, X-RN,2) representing the focal lengths of the first cameras.
        c: Tensor of shape (ON, X-RN, 2) representing the principal points of  the first cameras.

    Returns:
        camera_coords: Tensor of shape (ON, X-RN, H, W, 3) representing the camera coordinates.
    """
    camera_coords = ((pixel_grid[None, None] - c[:, :, None, None]) / focal[:, :, None, None])
    camera_coords = torch.cat((camera_coords, torch.ones_like(camera_coords[..., :1])), dim=-1)

    return camera_coords


def camera_to_world(camera_coords, pose1):
    """
    Convert camera coordinates to world coordinates.

    Args:
        camera_coords: Tensor of shape (ON, N-RN, H, W, 3) representing the camera coordinates.
        pose1: Tensor of shape (ON, N-RN, 4, 4) representing the c2w pose of the first camera.

    Returns:
        world_coords: Tensor of shape (ON, N-RN, H, W, 3) representing the world coordinates.
    """

    # camera_coords = camera_coords.unsqueeze(-1)
    camera_coords = camera_coords.reshape(camera_coords.size(0), camera_coords.size(1), -1, camera_coords.size(-1))
    camera_coords = camera_coords.to(pose1.device)

    homogeneous_coords = torch.cat((camera_coords, torch.ones_like(camera_coords[..., :1])), dim=-1)  # [2, 175, 49, 4]
    # print(pose1.dtype)
    # print((homogeneous_coords.transpose(-1, -2)).dtype)
    world_coords = torch.matmul(pose1, homogeneous_coords.transpose(-1, -2))  # ([2, 175, 4, 4])
    world_coords = world_coords.transpose(-1, -2)
    world_coords = world_coords.reshape(world_coords.size(0), world_coords.size(1), 16, 16,
                                        world_coords.size(-1))  # 根据网络修改
    # print(world_coords.shape)
    # world_coords = world_coords[..., :3]  # [2, 175, 49, 3]

    return world_coords


def world_to_pixel(world_coords, poses_ref, focal_ref, c_ref):
    """
    Convert world coordinates to pixel coordinates for reference images.

    Args:
        world_coords: Tensor of shape (ON, N-RN, H*W, 3) representing the world coordinates.
        poses_ref: Tensor of shape (ON, RN, 4, 4) representing the w2c poses of reference cameras.
        focal_ref: Tensor of shape (ON, RN,2) representing the focal lengths of reference cameras.
        c_ref: Tensor of shape (ON, RN, 2) representing the principal points of reference cameras.

    Returns:
        pixels_ref: Tensor of shape (ON, N-RN, RN, H, W, 2) representing the pixel coordinates for reference images.
    """
    ON, X_RN, H, W, _ = world_coords.shape
    _, RN, _, _ = poses_ref.shape
    # print(RN)
    pixels_ref = torch.zeros(ON, X_RN, RN, H, W, 2, device=world_coords.device)
    for i in range(ON):
        for j in range(X_RN):
            for k in range(RN):
                K = torch.tensor([[focal_ref[i, k, 0], 0, c_ref[i, k, 0]],
                                  [0, focal_ref[i, k, 1], c_ref[i, k, 1]],
                                  [0, 0, 1]], device=world_coords.device)
                # 使用参考相机的姿势将世界坐标转换为相机坐标
                camera_coords = torch.matmul(poses_ref[i, k], world_coords[i, j].transpose(-1, -2))
                # print(camera_coords.shape)
                camera_coords = camera_coords.transpose(-1, -2)  # (7,7,4)
                # print(camera_coords.shape)
                camera_coords = camera_coords[..., :3]
                # print(camera_coords.shape)
                # 使用相机内参将相机坐标转换为像素坐标
                pixels = torch.matmul(K, camera_coords.transpose(-1, -2))
                # print(pixels.shape)
                pixels = pixels.transpose(-1, -2)
                pixels = pixels[..., :2]
                # print(pixels.shape)
                pixels_ref[i, j, k] = pixels
    return pixels_ref


def normalize_coordinates(pixels_ref, W, H):
    """
    Normalize pixel coordinates to [-1, 1] range.

    Args:
        pixels_ref: Tensor of shape (ON, N-RN, RN, H, W, 2) representing the pixel coordinates for reference images.
        W: Width of the image.
        H: Height of the image.

    Returns:
        uv_feats: Tensor of shape (ON, N-RN, RN, H, W, 1, 2) representing the normalized pixel coordinates.
    """
    uv_feats = 2 * pixels_ref / torch.tensor([W, H], device=pixels_ref.device).reshape(1, 1, 1, 1, 1, 2) - 1.0
    return uv_feats


class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.make_encoder(cfg["encoder"])
        self.make_mlp(cfg["mlp"])
        self.linear = nn.Sequential(
            nn.Flatten(),  # 展平操作
            nn.Dropout(p=0.2),  # Dropout 层
            nn.Linear(in_features=768 * 16 * 16, out_features=256)  # 全连接层 根据网络修改
        )

    def make_encoder(self, cfg):
        self.encoder = TeacherNet()
        pretrained_weights_path = 'logs/MixVPR/Cambridge/ckpt_best.pth.tar'
        pretrained_state_dict = torch.load(pretrained_weights_path)
        self.encoder.load_state_dict(pretrained_state_dict["state_dict"])
        self.stop_encoder_grad = False

    def make_mlp(self, cfg):
        self.mlp_feature = MLPFeature.init_from_cfg(cfg["mlp_feature"])
        self.mlp_out = MLPOut.init_from_cfg(cfg["mlp_output"])

    def encode(self, images, poses, focal, c):
        """
        Encode feature map from reference images. Must be called before forward method.
        """
        with profiler.record_function("encode_reference_images"):
            ON, RN, _, _ = poses.shape
            self.num_objs = images.size(0)
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(1)  # Be consistent with RN
            self.num_ref_views = images.size(1)
            # print(poses.shape)
            images = images.reshape(-1, *images.shape[2:])  # (ON*RN, 3, H, W)
            poses = poses.reshape(-1, 4, 4)
            # print(poses.shape)
            # generate projection matrix, w2c
            rot = poses[:, :3, :3].transpose(1, 2)  # (ON*RN, 3, 3)
            trans = -torch.bmm(rot, poses[:, :3, 3:])  # (ON*RN, 3, 1)
            poses = torch.cat((rot, trans), dim=-1)  # (ON*RN, 3, 4)
            # # (ON*RN, 3, 4)->(ON*RN, 4, 4)
            extra_row = torch.tensor([0, 0, 0, 1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            # 使用 torch.cat 将 extra_row 广播为形状 (ON*RN, 1, 4)
            extra_rows = extra_row.expand(poses.size(0), 1, 4)
            extra_rows = extra_rows.to(poses.device)
            # 在第二维度上将 extra_rows 与 pose 拼接，形成形状为 (ON*RN, 4, 4) 的张量
            poses = torch.cat([poses, extra_rows], dim=1)
            poses = poses.reshape(ON, RN, 4, 4)
            _, latent = self.encoder(images)  # (ON*RN, d_latent, H, W)
            if self.stop_encoder_grad:  # 可以选择是否让来自PR的特征提取网络参与更新
                latent = latent.detach()

            self.focal = focal
            self.c = c

            self.ref_image = images
            self.ref_pose = poses
            self.ref_latent = latent

            self.image_shape = torch.empty(2, dtype=torch.float32)
            self.latent_scaling = torch.empty(2, dtype=torch.float32)

            self.image_shape[0] = images.shape[-1]
            self.image_shape[1] = images.shape[-2]
            self.latent_scaling[0] = latent.shape[-1]
            self.latent_scaling[1] = latent.shape[-2]
            self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1)

    def get_features(self, pose):
        """
        Get encoded features from reference images

        Args:
            rgb: (ON, IN-RN, 3, H, W)
            poses: (ON, IN-RN, 4, 4)

        Returns:
            latent: extracted reference image featues (ON*RN*RB, d_latent)
            p_feature: extracted point pose features in reference coordinates (ON*RN*RB, 6)
        """

        with profiler.record_function("extract_features"):
            RN = self.num_ref_views
            device = pose.device
            N_RN = pose.size(1)
            uv_feats = transform_images_to_reference_image_coordinates(pose, self.ref_pose, self.focal, self.c,
                                                                       self.focal,
                                                                       self.c)  # (ON, N-RN, RN, H, W, 1, 2)
            # (ON, N-RN, RN, H, W, 2)->(ON*(N-RN), RN, H, W, 2)
            uv_feats_reshaped = uv_feats.view(-1, uv_feats.size(2), uv_feats.size(3), uv_feats.size(4),
                                              uv_feats.size(5))
            # (ON * (N - RN), RN, H, W, 2)->(ON*(N-RN)* RN, H, W, 2)
            uv_feats_final = uv_feats_reshaped.view(-1, uv_feats_reshaped.size(2), uv_feats_reshaped.size(3),
                                                    uv_feats_reshaped.size(4))
            # uv_feats_final = uv_feats_reshaped.squeeze(-2)
            uv_feat = uv_feats_final  # (ON*(N-RN)* RN, H, W, 2)

            ref_latent = self.ref_latent.repeat(N_RN, 1, 1, 1)  # (ON*RN*(N-RN), d_latent, H, W)
            ref_poses = relative_poses(pose, self.ref_pose)  # (ON,N-RN,RN,4,4)
            ref_poses_reshaped = ref_poses.view(-1, ref_poses.size(2), ref_poses.size(3), ref_poses.size(4))
            ref_poses = ref_poses_reshaped.view(-1, ref_poses_reshaped.size(2), ref_poses_reshaped.size(3))
            p_feature = pose_tensor_to_pose_representations(ref_poses)
            p_feature = p_feature.to(ref_latent.device)
            latent = F.grid_sample(
                ref_latent,  # (ON*RN*(N-RN), d_latent, H, W)
                uv_feat,  # (ON*(N-RN)*RN, H, W, 2)
                align_corners=True,
                mode="bilinear",
                padding_mode="border",
            )  # (ON*RN*(N-RN), d_latent, H, W)
            # print(latent.shape)
            latent = self.linear(latent)  # ((ON*(N-RN)*RN, outfeatures))

            # if self.stop_encoder_grad:
            # latent = latent.detach()

        return (
            latent,
            p_feature,
        )  # (ON*RN*(n-RN), outfeatures), (ON*RN*(n-RN),6)

    def forward(self, pose):
        """
        Get model final prediction given surface point position and view directions.

        Args:
            #rgb: (ON, N-RN, 3, H, W)
            pose: (ON, N-RN, 4, 4)

        Returns:
            output: (ON, RB, d_out)
        """

        with profiler.record_function("model_inference"):
            ON, N_RN, _, _ = pose.shape
            latent, p_feature = self.get_features(pose)  # (ON*RN*(n-RN), outfeatures), (ON*RN*(N-RN),6)

            feature, weight = self.mlp_feature(
                latent,
                p_feature,
            )  # (ON*RN*(N-RN), d_feature) (ON*RN*(N-RN), d_feature)
            # print("feature_weight")
            # print(feature.shape)
            # print(weight.shape)
            feature = util.weighted_pooling(  # (ON,RB,2*d_feature)
                feature, inner_dims=(self.num_ref_views, N_RN), weight=weight
            ).reshape(
                ON * N_RN, -1
            )  # (ON*N_RN, 2*d_feature) mean+var
            # print("feature")
            # print(feature.shape)
            final_output = self.mlp_out(feature).reshape(ON, N_RN, -1)  # (ON, N_RN, d_out)
            # print(final_output.shape)
            logit_mean = final_output[:, :, :512]  # (ON, RB, 512)
            logit_log_var = final_output[:, :, 512:]  # (ON, RB , 512)
            # print(logit_mean)
            # print(logit_log_var)
            # print(logit_mean.shape)
            # print(logit_log_var.shape)
            des_dict = {}
            # EPSILON = 1e-6
            logit_log_var = torch.clamp(logit_log_var, min=-5 + 1.0e-6, max=5.0 - 1.0e-6)
            des_dict["logit_mean"] = logit_mean
            des_dict["logit_log_var"] = logit_log_var

            with torch.no_grad():
                sampled_predictions = util.get_samples(
                    logit_mean, torch.sqrt(torch.exp(logit_log_var)), 100
                )
                des_mean = torch.mean(sampled_predictions, axis=0)
                des_std = torch.std(sampled_predictions, dim=0)
                des_dict["des"] = des_mean
                # print(des_dict["des"])
                des_dict["uncertainty"] = des_std
                des_dict["all_uncertainty"] = torch.mean(des_std, dim=-1)
                # print(des_dict["all_uncertainty"])

        return des_dict  # des logit mean and log variance


if __name__ == '__main__':
    """
    uv_feat=torch.rand((1,25,25,2))
    inputs = torch.rand((1, 256, 16, 16))
    outputs = F.grid_sample(
                inputs,  # (ON*RN, d_latent, H, W)
                uv_feat,  # (ON*RN, RB, 1, 2)
                align_corners=True,
                mode='bilinear',
                padding_mode='zeros',
            )  # (ON*RN, d_latent, RB, 1)
    print(outputs.shape)

    logit_mean=torch.rand((1,60,512))
    logit_log_var=torch.rand((1,60,512))
    sampled_predictions = util.get_samples(
        logit_mean, torch.sqrt(torch.exp(logit_log_var)), 100
    )
    rgb_mean = torch.mean(sampled_predictions, axis=0)
    rgb_std = torch.std(sampled_predictions, axis=0)
    render_dict={}
    render_dict["rgb"] = rgb_mean
    render_dict["uncertainty"] = torch.mean(rgb_std, dim=-1)
    print(DotMap(render_dict).toDict())
    """
    inputs = torch.rand((1, 25, 25, 2))
    encoder = TeacherNet()
    pretrained_weights_path = 'logs/ckpt_best.pth.tar'
    pretrained_state_dict = torch.load(pretrained_weights_path)
    encoder.load_state_dict(pretrained_state_dict["state_dict"])
