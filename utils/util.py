import cv2
import numpy as np
import torch
from torchvision import transforms
from torch import nn
import functools
import math
import warnings
import torchmetrics
import torch.nn.functional as F


def image_float_to_uint8_norm(img, min, max):
    """
    Convert a float image (min-max) to uint8 (0-255)
    """
    img = (img - min) / (max - min)
    img *= 255.0
    img = np.clip(img, a_min=0.0, a_max=255.0)
    return img.astype(np.uint8)


def image_float_to_uint8(img):
    """
    Convert a float image (0.0-1.0) to uint8 (0-255)
    """
    img *= 255.0
    img = np.clip(img, a_min=0.0, a_max=255.0)
    return img.astype(np.uint8)


def cmap(img, color_map=cv2.COLORMAP_BONE):
    return cv2.applyColorMap(image_float_to_uint8(img), color_map)


def unc_cmap(img, color_map=cv2.COLORMAP_PLASMA):
    img = cv2.applyColorMap(image_float_to_uint8(img/ 0.02), color_map)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def error_cmap(img, color_map=cv2.COLORMAP_PLASMA):
    img = cv2.applyColorMap(image_float_to_uint8(img), color_map)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def des_cmap(img, color_map=cv2.COLORMAP_SUMMER):
    img = cv2.applyColorMap(image_float_to_uint8(img/0.25), color_map)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def depth_cmap(img, min, max, color_map=cv2.COLORMAP_BONE):
    return cv2.applyColorMap(image_float_to_uint8_norm(img, min, max), color_map)


def batched_index_select_nd(t, inds):
    """
    Index select on dim 1 of a n-dimensional batched tensor.
    :param t (batch, n, ...)
    :param inds (batch, k)
    :return (batch, k, ...)
    """
    return t.gather(
        1, inds[(...,) + (None,) * (len(t.shape) - 2)].expand(-1, -1, *t.shape[2:])
    )


def batched_index_select_nd_last(t, inds):
    """
    Index select on dim -1 of a >=2D multi-batched tensor. inds assumed
    to have all batch dimensions except one data dimension 'n'
    :param t (batch..., n, m)
    :param inds (batch..., k)
    :return (batch..., n, k)
    """
    dummy = inds.unsqueeze(-2).expand(*inds.shape[:-1], t.size(-2), inds.size(-1))
    out = t.gather(-1, dummy)
    return out


def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """

    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


def get_image_to_tensor_balanced(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(ops)


def gen_grid(*args, ij_indexing=False):
    """
    Generete len(args)-dimensional grid.
    Each arg should be (lo, hi, sz) so that in that dimension points
    are taken at linspace(lo, hi, sz).
    Example: gen_grid((0,1,10), (-1,1,20))
    :return (prod_i args_i[2], len(args)), len(args)-dimensional grid points
    """
    return torch.from_numpy(
        np.vstack(
            np.meshgrid(
                *(np.linspace(lo, hi, sz, dtype=np.float32) for lo, hi, sz in args),
                indexing="ij" if ij_indexing else "xy",
            )
        )
        .reshape(len(args), -1)
        .T
    )


def quat_to_rot(q):
    """
    Quaternion to rotation matrix
    """
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3, 3), device=q.device)
    qr = q[:, 0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj * qi - qk * qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1 - 2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2 * (qj * qk - qi * qr)
    R[:, 2, 0] = 2 * (qk * qi - qj * qr)
    R[:, 2, 1] = 2 * (qj * qk + qi * qr)
    R[:, 2, 2] = 1 - 2 * (qi**2 + qj**2)
    return R


def rot_to_quat(R):
    """
    Rotation matrix to quaternion
    """
    batch_size, _, _ = R.shape
    q = torch.ones((batch_size, 4), device=R.device)

    R00 = R[:, 0, 0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:, 0] = torch.sqrt(1.0 + R00 + R11 + R22) / 2
    q[:, 1] = (R21 - R12) / (4 * q[:, 0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q


def unproj_map(width, height, f, c=None, device="cpu"):
    """
    Get camera unprojection map for given image size.
    [y,x] of output tensor will contain unit vector of camera ray of that pixel.
    :param width image width
    :param height image height
    :param f focal length, either a number or tensor [fx, fy]
    :param c principal point, optional, either None or tensor [fx, fy]
    if not specified uses center of image
    :return unproj map (height, width, 3)
    """
    if c is None:
        c = [width * 0.5, height * 0.5]
    else:
        c = c.squeeze()
    if isinstance(f, float):
        f = [f, f]
    elif len(f.shape) == 0:
        f = f[None].expand(2)
    elif len(f.shape) == 1:
        f = f.expand(2)
    Y, X = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) - float(c[1]),
        torch.arange(width, dtype=torch.float32) - float(c[0]),
    )
    X = X.to(device=device) / float(f[0])
    Y = Y.to(device=device) / float(f[1])
    Z = torch.ones_like(X)
    unproj = torch.stack((X, Y, Z), dim=-1)
    unproj /= torch.norm(unproj, dim=-1).unsqueeze(-1)
    return unproj


def get_cuda(gpu_id):
    """
    Get a torch.device for GPU gpu_id. If GPU not available,
    returns CPU device.
    """
    return (
        torch.device("cuda:%d" % gpu_id)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def gen_rays(poses, width, height, focal, z_near, z_far, c=None, ndc=False):
    """
    Generate camera rays
    :return (B, H, W, 8)
    """
    num_images = poses.shape[0]
    device = poses.device
    cam_unproj_map = (
        unproj_map(width, height, focal.squeeze(), c=c, device=device)
        .unsqueeze(0)
        .repeat(num_images, 1, 1, 1)
    )
    cam_centers = poses[:, None, None, :3, 3].expand(-1, height, width, -1)
    cam_raydir = torch.matmul(
        poses[:, None, None, :3, :3], cam_unproj_map.unsqueeze(-1)
    )[:, :, :, :, 0]
    if ndc:
        if not (z_near == 0 and z_far == 1):
            warnings.warn(
                "dataset z near and z_far not compatible with NDC, setting them to 0, 1 NOW"
            )
        z_near, z_far = 0.0, 1.0
        cam_centers, cam_raydir = ndc_rays(
            width, height, focal, 1.0, cam_centers, cam_raydir
        )

    cam_nears = (
        torch.tensor(z_near, device=device)
        .view(1, 1, 1, 1)
        .expand(num_images, height, width, -1)
    )
    cam_fars = (
        torch.tensor(z_far, device=device)
        .view(1, 1, 1, 1)
        .expand(num_images, height, width, -1)
    )
    return torch.cat(
        (cam_centers, cam_raydir, cam_nears, cam_fars), dim=-1
    )  # (B, H, W, 8)


def coordinate_transformation(pose, format):
    """
    transform camera coordinate to opencv format
    """

    _coord_trans_normal_2_opencv = torch.tensor(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32
    )

    if format == "opencv":
        return torch.tensor(pose, dtype=torch.float32)
    elif format == "normal":
        return torch.tensor(pose, dtype=torch.float32) @ _coord_trans_normal_2_opencv


def generate_mask(uv):
    mask1 = uv < 1.0
    mask2 = uv > -1.0
    mask = mask1 * mask2
    mask = torch.all(mask, dim=-1)
    mask = ~torch.all(~mask, dim=-2)
    return mask


def get_samples_rp(mean, std, sample_num):
    """
    sample from gaussain distribution using reparameterization trick
    """

    device = mean.device
    sampled_predictions = torch.zeros((sample_num, *mean.size()), device=device)
    for i in range(sample_num):
        noise_mean = torch.zeros(mean.size(), device=device)
        noise_std = torch.ones(std.size(), device=device)
        epsilon = torch.distributions.normal.Normal(noise_mean, noise_std).sample()
        sample = mean + torch.mul(std, epsilon)
        sampled_predictions[i] = sample

    sampled_predictions = torch.sigmoid(sampled_predictions)  # (num, ON, RB, 3)
    return sampled_predictions


def get_samples(mean, std, sample_num):
    """
    sample from gaussain distribution
    """

    device = mean.device
    sampled_predictions = torch.zeros((sample_num, *mean.size()), device=device)
    for i in range(sample_num):
        sample = torch.distributions.normal.Normal(mean, std).sample()
        sampled_predictions[i] = sample

    sampled_predictions = torch.sigmoid(sampled_predictions)  # (num, ON, RB, 3)
    return sampled_predictions


def get_norm_layer(norm_type="instance", group_norm_groups=32):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def make_conv_2d(
    dim_in,
    dim_out,
    padding_type="reflect",
    norm_layer=None,
    activation=None,
    kernel_size=3,
    use_bias=False,
    stride=1,
    no_pad=False,
    zero_init=False,
):
    conv_block = []
    amt = kernel_size // 2
    if stride > 1 and not no_pad:
        raise NotImplementedError(
            "Padding with stride > 1 not supported, use same_pad_conv2d"
        )

    if amt > 0 and not no_pad:
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(amt)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(amt)]
        elif padding_type == "zero":
            conv_block += [nn.ZeroPad2d(amt)]
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

    conv_block.append(
        nn.Conv2d(
            dim_in, dim_out, kernel_size=kernel_size, bias=use_bias, stride=stride
        )
    )
    if zero_init:
        nn.init.zeros_(conv_block[-1].weight)
    #  else:
    #  nn.init.kaiming_normal_(conv_block[-1].weight)
    if norm_layer is not None:
        conv_block.append(norm_layer(dim_out))

    if activation is not None:
        conv_block.append(activation)
    return nn.Sequential(*conv_block)


def feature_attention(feature, inner_dims, weight):
    feature_shape = feature.shape

    feature = feature.reshape(-1, *inner_dims, feature_shape[-1])
    weight = weight.reshape(-1, *inner_dims, feature_shape[-1])  # (ON ,RN, RB, L)

    softmax_weight = nn.Softmax(weight, dim=-1)
    feature = softmax_weight * feature
    feature = torch.sum(feature, dim=1)  # (ON, RB, L)
    return feature


def cal_statistics(t, inner_dims):
    t = t.reshape(-1, *inner_dims, t.shape[-1])  # (ON, RN, RB, L)
    t_mean = torch.mean(t, dim=1)
    t_var = torch.var(t, dim=1)
    return t_mean, t_var


# def weighted_pooling(t, inner_dims, weight):
#     weight = weight.reshape(-1, *inner_dims, 1)
#     weight = weight / (torch.sum(weight, dim=1, keepdim=True) + 1e-8)  # (ON, RN, RB, 1)

#     t = t.reshape(-1, *inner_dims, t.shape[-1])  # (ON, RN, RB, L)

#     t_mean = torch.sum(t * weight, dim=1, keepdim=True)  # (ON, 1, RB, L)
#     t_var = torch.sum(weight * (t - t_mean) ** 2, dim=1, keepdim=True)  # (ON, 1, RB, L)
#     t_fuse = torch.cat((t_mean, t_var), dim=-1).squeeze(1)  # (ON, RB, 2L)
#     return t_fuse


def weighted_pooling(t, inner_dims, weight):
    t_weighted = t * weight
    t_shape = t.shape
    t = t_weighted.reshape(-1, *inner_dims, *t_shape[1:])
    #print(t.shape)
    t_mean = torch.mean(t, dim=1)
    #print(t_mean.shape)
    t_var = torch.var(t, dim=1)
    t = torch.cat((t_mean, t_var), dim=-1)
    #print(t.shape)
    return t


def merge_feature(t, inner_dims):
    t_shape = t.shape
    t = t.reshape(-1, *inner_dims, *t_shape[1:])

    t_mean = torch.mean(t, dim=1)
    t_var = torch.var(t, dim=1)
    t = torch.cat((t_mean, t_var), dim=-1)
    return t


def calc_psnr(pred, target):
    """
    Compute PSNR of two tensors in decibels.
    pred/target should be of same size or broadcastable
    """
    mse = ((pred - target) ** 2).mean()
    psnr = -10 * math.log10(mse)
    return psnr


# https://github.com/abdo-eldesokey/pncnn/blob/c6122e9c442eabeb0145b241121aeba0039eb5e7/utils/sparsification_plot.py#L10
def cal_ause(err_vec, uncert_vec):
    # import matplotlib

    # matplotlib.use("TkAgg")
    # import matplotlib.pyplot as plt
    #print(err_vec.shape)
    #print(uncert_vec.shape)
    ratio_removed = np.linspace(0, 1, 100, endpoint=False)
    err_vec_sorted, _ = torch.sort(err_vec)

    # Calculate the error when removing a fraction pixels with error
    n_valid_pixels = len(err_vec)
    rmse_err = []
    for i, r in enumerate(ratio_removed):
        mse_err_slice = err_vec_sorted[0 : int((1 - r) * n_valid_pixels)]
        rmse_err.append(torch.sqrt(mse_err_slice).mean().cpu().numpy())

    # Normalize RMSE
    rmse_err = rmse_err / rmse_err[0]

    ###########################################

    # Sort by variance
    # print('Sorting Variance ...')
    uncert_vec = torch.sqrt(uncert_vec)
    _, uncert_vec_sorted_idxs = torch.sort(uncert_vec, descending=True)

    # Sort error by variance
    err_vec_sorted_by_uncert = err_vec[uncert_vec_sorted_idxs]

    rmse_err_by_var = []
    for i, r in enumerate(ratio_removed):
        mse_err_slice = err_vec_sorted_by_uncert[0 : int((1 - r) * n_valid_pixels)]
        rmse_err_by_var.append(torch.sqrt(mse_err_slice).mean().cpu().numpy())

    # Normalize RMSE
    rmse_err_by_var = rmse_err_by_var / max(rmse_err_by_var)

    # plt.plot(ratio_removed, rmse_err, "--")
    # plt.plot(ratio_removed, rmse_err_by_var, "-r")
    # plt.show()
    ause = np.trapz(np.array(rmse_err_by_var) - np.array(rmse_err), ratio_removed)
    return ause


def calc_metrics(predict, gt):
    #H, W, _ = gt.shape
    gt=gt.cpu()
    des = predict.des[0].cpu()
    std = predict.uncertainty[0].cpu()
    variance = std**2
    mean_variance = torch.mean(variance)
    log_mean_variance = torch.log10(mean_variance)


    error = des - gt

    mse=torch.square(error)
    mean_mse=torch.mean(torch.square(error), dim=-1)
    mean_mae = torch.mean(torch.abs(error), dim=-1)

    ause = cal_ause(mse.reshape(-1), variance.reshape(-1))
    #mean_mse = torch.mean(mse)  # mean over whole image
    #mean_mae = torch.mean(mae)
    EPSILON = 1e-6  # 很小的正数值
    std = torch.clamp(std, min=EPSILON)
    normal_distribution = torch.distributions.normal.Normal(
        des, std
    )
    ll = normal_distribution.log_prob(gt)
    nll = -torch.mean(ll)

    metrics_dict = {
        "mean_variance": mean_variance,
        "log_mean_variance": log_mean_variance,
        "psnr": 0,
        "ssim": 0,
        "mean_mse": mean_mse,
        "mean_mae": mean_mae,
        "ause": ause,
        "nll": nll,
    }

    return metrics_dict


def get_module(net):
    """
    Shorthand for either net.module (if net is instance of DataParallel) or net
    """
    if isinstance(net, torch.nn.DataParallel):
        return net.module
    else:
        return net


def cal_cosine_simularity(v1, v2):
    cos = nn.CosineSimilarity(dim=-1)
    return cos(v1, v2)

def visualize_descriptors(descriptors, img_shape):
    _, N = np.shape(descriptors)
    img = np.zeros(img_shape)

    num_cols = min(N, img_shape[1])
    for col in range(num_cols):
        img[:, col] = descriptors[:, col]

    return img
def tb_visualizer(predict, gt, writer, H, W, i):
    if len(predict.des) > 0 and predict.des is not None:
        des_np = predict.des[0].cpu()
        des_np=des_np.numpy()
        #print(des_np.shape)
        rgb_np = visualize_descriptors(des_np, (100, 512))

        writer.add_image(
            "predict/rgb",
            (rgb_np * 255).astype(np.uint8),
            global_step=i,
            dataformats="HW",
        )
        gt=gt.cpu().numpy()
        gt_np = visualize_descriptors(gt, (100, 512))
        error_np = np.abs(rgb_np - gt_np)
        error = error_cmap(error_np)
        writer.add_image(
            "predict/error",
            error*255,
            global_step=i,
            dataformats="HWC",
        )

    if len(predict.uncertainty) > 0 and predict.uncertainty is not None:
        #print(predict.uncertainty.shape)
        uncertainty_np = predict.uncertainty[0].cpu()
        uncertainty_np = uncertainty_np.numpy()
        #print(uncertainty_np.shape)
        uncertainty_np = visualize_descriptors(uncertainty_np, (100, 512))
        uncertainty = unc_cmap(uncertainty_np)
        writer.add_image(
            "predict/uncertainty",
            uncertainty,
            global_step=i,
            dataformats="HWC",
        )
if __name__ == '__main__':
    des_np=torch.rand((1,512)).cpu()
    des_np=des_np.numpy()
    gt=torch.rand((1,512)).cpu()
    gt = gt.numpy()
    rgb_np = visualize_descriptors(des_np, (200, 300))
    gt_np = visualize_descriptors(gt, (200, 300))
    error_np = np.abs(rgb_np - gt_np)
    error = error_cmap(error_np)

    print(error.shape)