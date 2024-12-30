import sys

sys.path.append('../')
import torch
from torch import nn, optim
from torchvision.utils import save_image
import os, pdb
from torchsummary import summary
from dataset_loaders.load_7Scenes import load_7Scenes_dataloader
from dataset_loaders.load_Cambridge import load_Cambridge_dataloader
import os.path as osp
import numpy as np
from utils.utils import plot_features, save_image_saliancy, save_image_saliancy_single
from utils.utils import freeze_bn_layer, freeze_bn_layer_train
from models.nerfw import create_nerf
from tqdm import tqdm
from dm.callbacks import EarlyStopping
from feature.dfnet import DFNet, DFNet_s
# from feature.efficientnet import EfficientNetB3 as DFNet
# from feature.efficientnet import EfficientNetB0 as DFNet
from feature.misc import *
from feature.options_nerf import config_parser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(0)
torch.manual_seed(0)
import random

random.seed(0)


def tmp_plot(target_in, rgb_in, features_target, features_rgb):
    '''
    print 1 pair of salient feature map
    '''
    print("for debug only...")
    pdb.set_trace()
    ### plot featues with pixel-wise addition
    save_image(target_in[1], './tmp/target_in.png')
    save_image(rgb_in[1], './tmp/rgb_in.png')
    save_image_saliancy(features_target[1], './tmp/target', True)
    save_image_saliancy(features_rgb[1], './tmp/rgb', True)
    ### plot featues seperately
    save_image(target_in[1], './tmp/target_in.png')
    save_image(rgb_in[1], './tmp/rgb_in.png')
    plot_features(features_target[:, 1:2, ...], './tmp/target', False)
    plot_features(features_rgb[:, 1:2, ...], './tmp/rgb', False)
    sys.exit()


def tmp_plot2(target_in, rgb_in, features_target, features_rgb, i=0):
    '''
    print 1 pair of batch of salient feature map
    :param: target_in [B, 3, H, W]
    :param: rgb_in [B, 3, H, W]
    :param: features_target [B, C, H, W]
    :param: features_rgb [B, C, H, W]
    :param: frame index i of batch
    '''
    print("for debug only...")
    save_image(target_in[i], './tmp/target_in.png')
    save_image(rgb_in[i], './tmp/rgb_in.png')
    pdb.set_trace()
    features_t = features_target[i].clone()[:, None, :, :]
    features_r = features_rgb[i].clone()[:, None, :, :]
    save_image_saliancy(features_t, './tmp/target', True)
    save_image_saliancy(features_r, './tmp/rgb', True)


def tmp_plot3(target_in, rgb_in, features_target, features_rgb, i=0):
    '''
    print 1 pair of 1 sample of salient feature map
    :param: target_in [B, 3, H, W]
    :param: rgb_in [B, 3, H, W]
    :param: features_target [B, C, H, W]
    :param: features_rgb [B, C, H, W]
    :param: frame index i of batch
    '''
    print("for debug only...")
    save_image(target_in[i], './tmp/target_in.png')
    save_image(rgb_in[i], './tmp/rgb_in.png')
    features_t = features_target[i].clone()[:, None, :, :]
    features_r = features_rgb[i].clone()[:, None, :, :]
    save_image_saliancy(features_t[0], './tmp/target', True)
    save_image_saliancy(features_r[0], './tmp/rgb', True)


def lognuniform(low=-2, high=0, size=1, base=10):
    ''' sample from log uniform distribution between 0.01~1 '''
    return np.power(base, np.random.uniform(low, high, size))


def getrelpose(pose1, pose2):
    ''' get relative pose from abs pose pose1 to abs pose pose2
    R^{v}_{gt} = R_v * R_gt.T
    :param: pose1 [B, 3, 4]
    :param: pose2 [B, 3, 4]
    return rel_pose [B, 3, 4]
    '''
    assert (pose1.shape == pose2.shape)
    rel_pose = pose1 - pose2  # compute translation term difference
    rel_pose[:, :3, :3] = pose2[:, :3, :3] @ torch.transpose(pose1[:, :3, :3], 1, 2)  # compute rotation term difference
    return rel_pose


parser = config_parser()
args = parser.parse_args()


def train_on_batch(args, targets, rgbs, poses, feat_model, dset_size, FeatureLoss, optimizer, hwf):
    ''' core training loop for featurenet'''
    feat_model.train()
    H, W, focal = hwf
    H, W = int(H), int(W)
    if args.freezeBN:
        feat_model = freeze_bn_layer_train(feat_model)

    train_loss_epoch = []
    select_inds = np.random.choice(dset_size, size=[dset_size], replace=False)  # (N_rand,)

    batch_size = args.featurenet_batch_size  # manual setting, use smaller batch size like featurenet_batch_size = 4 if OOM
    if dset_size % batch_size == 0:
        N_iters = dset_size // batch_size
    else:
        N_iters = dset_size // batch_size + 1
    i_batch = 0

    for i in range(0, N_iters):
        if i_batch + batch_size > dset_size:
            i_batch = 0
            break
        i_inds = select_inds[i_batch:i_batch + batch_size]
        i_batch = i_batch + batch_size

        # convert input shape to [B, 3, H, W]
        target_in = targets[i_inds].clone().permute(0, 3, 1, 2).to(device)
        rgb_in = rgbs[i_inds].clone().permute(0, 3, 1, 2).to(device)
        pose = poses[i_inds].clone().reshape(batch_size, 12).to(device)
        pose = torch.cat([pose, pose])  # double gt pose tensor

        features, predict_pose = feat_model(torch.cat([target_in, rgb_in]), True, upsampleH=H,
                                            upsampleW=W)  # features: (1, [2, B, C, H, W])

        # get features_target and features_rgb
        if args.DFNet:
            features_target = features[0]  # [3, B, C, H, W]
            features_rgb = features[1]
        else:
            features_target = features[0][0]
            features_rgb = features[0][1]

        # svd, seems not very benificial here, therefore removed

        if args.poselossonly:
            loss_pose = PoseLoss(args, predict_pose, pose, device)  # target
            loss = loss_pose
        elif args.featurelossonly:  # Not good. To be removed later
            loss_f = FeatureLoss(features_rgb, features_target)
            loss = loss_f
        else:
            loss_pose = PoseLoss(args, predict_pose, pose, device)  # target
            if args.tripletloss:
                loss_f = triplet_loss_hard_negative_mining_plus(features_rgb, features_target,
                                                                margin=args.triplet_margin)
            else:
                loss_f = FeatureLoss(features_rgb, features_target)
            loss = loss_pose + loss_f

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_epoch.append(loss.item())
    train_loss = np.mean(train_loss_epoch)
    return train_loss


def train_on_batch_with_random_view_synthesis(args, targets, rgbs, poses, virtue_view, poses_perturb, feat_model,
                                              dset_size, FeatureLoss, optimizer, hwf, img_idxs, render_kwargs_test):
    ''' we implement random view synthesis for generating more views to help training posenet '''
    feat_model.train()

    H, W, focal = hwf
    H, W = int(H), int(W)

    if args.freezeBN:
        feat_model = freeze_bn_layer_train(feat_model)

    train_loss_epoch = []

    # random generate batch_size of idx
    select_inds = np.random.choice(dset_size, size=[dset_size], replace=False)  # (N_rand,)

    batch_size = args.featurenet_batch_size  # manual setting, use smaller batch size like featurenet_batch_size = 4 if OOM
    if dset_size % batch_size == 0:
        N_iters = dset_size // batch_size
    else:
        N_iters = dset_size // batch_size + 1

    i_batch = 0
    for i in range(0, N_iters):
        if i_batch + batch_size > dset_size:
            i_batch = 0
            break
        i_inds = select_inds[i_batch:i_batch + batch_size]
        i_batch = i_batch + batch_size

        # convert input shape to [B, 3, H, W]
        target_in = targets[i_inds].clone().permute(0, 3, 1, 2).to(device)
        rgb_in = rgbs[i_inds].clone().permute(0, 3, 1, 2).to(device)
        pose = poses[i_inds].clone().reshape(batch_size, 12).to(device)
        rgb_perturb = virtue_view[i_inds].clone().permute(0, 3, 1, 2).to(device)
        pose_perturb = poses_perturb[i_inds].clone().reshape(batch_size, 12).to(device)

        # inference feature model for GT and nerf image
        pose = torch.cat([pose, pose])  # double gt pose tensor
        features, predict_pose = feat_model(torch.cat([target_in, rgb_in]), return_feature=True, upsampleH=H,
                                            upsampleW=W)  # features: (1, [2, B, C, H, W])

        # get features_target and features_rgb
        if args.DFNet:
            features_target = features[0]  # [3, B, C, H, W]
            features_rgb = features[1]

        loss_pose = PoseLoss(args, predict_pose, pose, device)  # target

        if args.tripletloss:
            loss_f = triplet_loss_hard_negative_mining_plus(features_rgb, features_target, margin=args.triplet_margin)
        else:
            loss_f = FeatureLoss(features_rgb, features_target)  # feature Maybe change to s2d-ce loss

        # inference model for RVS image
        _, virtue_pose = feat_model(rgb_perturb.to(device), False)

        # add relative pose loss here. TODO: This FeatureLoss is nn.MSE. Should be fixed later
        loss_pose_perturb = PoseLoss(args, virtue_pose, pose_perturb, device)
        loss = args.combine_loss_w[0] * loss_pose + args.combine_loss_w[1] * loss_f + args.combine_loss_w[
            2] * loss_pose_perturb

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_epoch.append(loss.item())
    train_loss = np.mean(train_loss_epoch)
    return train_loss


def train_feature(args, train_dl, val_dl, test_dl, hwf, i_split, near, far):
    i_train, i_val, i_test = i_split
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # load NeRF
    _, render_kwargs_test, start, _, _ = create_nerf(args)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    # render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    if args.reduce_embedding == 2:
        render_kwargs_test['i_epoch'] = start

    N_epoch = args.epochs + 1  # epoch
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    world_setup_dict = {
        'pose_scale': train_dl.dataset.pose_scale,
        'pose_scale2': train_dl.dataset.pose_scale2,
        'move_all_cam_vec': train_dl.dataset.move_all_cam_vec,
    }
    pose_list = []
    img_idx_list = []
    n=3
    #targets, rgbs, poses, img_idxs = render_nerfw_imgs(args, train_dl, hwf, device, render_kwargs_test,
    #                                                  world_setup_dict)
    for batch_idx, (target, pose, img_idx) in enumerate(train_dl):
        pose = pose.reshape(3, 4)  # reshape to 3x4 rot matrix
        pose_list.append(pose.cpu())
        img_idx = img_idx.to(device)
        img_idx_list.append(img_idx.cpu())

    poses = torch.stack(pose_list).detach()
    #print(poses.shape)
    img_idxs = torch.stack(img_idx_list).detach()
    #print(img_idxs.dim())
    #print("img_idxs")
    #print(img_idxs)
    # 对每个元素进行复制
    print(img_idxs)
    #img_idxs_repeated = img_idxs.repeat(n, 1, 1)
    img_idxs_repeated = torch.repeat_interleave(img_idxs, repeats=3, dim=0)
    print(img_idxs_repeated)
    #print("img_idxs_repeated")
    #print(img_idxs_repeated.shape)
    #img_idxs = torch.cat((img_idxs, img_idxs_repeated), dim=0)
    img_idxs = img_idxs_repeated
    dset_size = len(train_dl.dataset)
    #print("dset_size")
    #print(dset_size)
    # clean GPU memory before testing, try to avoid OOM
    torch.cuda.empty_cache()



    if args.random_view_synthesis:

    # random sample virtual camera locations, todo:
        rand_trans = args.rvs_trans
        rand_rot = args.rvs_rotation

        # determine bounding box
        b_min = [poses[:, 0, 3].min() - args.d_max, poses[:, 1, 3].min() - args.d_max,
                         poses[:, 2, 3].min() - args.d_max]
        b_max = [poses[:, 0, 3].max() + args.d_max, poses[:, 1, 3].max() + args.d_max,
                         poses[:, 2, 3].max() + args.d_max]
        poses_perturb = poses.clone().numpy()
        #print("pose_perturb")
        #print(poses_perturb.shape)
        #print(poses_perturb[1])
        #poses_pertur = perturb_single_render_pose(poses_perturb[1], rand_trans, rand_rot)
        #print(poses_pertur[1])
        """
        for i in range(poses_pertur.shape[0]):
            for j in range(3):
                if poses_pertur[i, j, 3] < b_min[j]:
                    poses_pertur[i, j, 3] = b_min[j]
                elif poses_pertur[i, j, 3] > b_max[j]:
                    poses_pertur[i, j, 3] = b_max[j]
        poses_pertur = torch.Tensor(poses_pertur).to(device)
        virtue_view = render_virtual_imgs(args, poses_pertur, img_idxs, hwf, device, render_kwargs_test,
                                  world_setup_dict)
         """
        poses_perturb0= None
        for i in range(dset_size):
            poses_perturb_new = perturb_single_render_pose(poses_perturb[i], rand_trans, rand_rot)
            if poses_perturb0 is None:
                poses_perturb0 = poses_perturb_new
            else:
                poses_perturb0 = np.append(poses_perturb0, poses_perturb_new, axis=0)#原位姿加随机生成的位姿来生成图像
            #poses_perturb=poses_perturb_new
            #print(poses_perturb.shape)
            #print(poses_perturb[i,1,3])
        #print(poses_perturb0.shape)
        for i in range(poses_perturb0.shape[0]):
            for j in range(3):
                if poses_perturb0[i, j, 3] < b_min[j]:
                    poses_perturb0[i, j, 3] = b_min[j]
                elif poses_perturb0[i, j, 3] > b_max[j]:
                    poses_perturb0[i, j, 3] = b_max[j]
        #print(poses_perturb0.shape)
        #print(poses_perturb)
        poses_perturb0 = torch.Tensor(poses_perturb0).to(device)  # [B, 3, 4]
        #print(poses_perturb0.shape)
        #tqdm.write("renders RVS...")
        virtue_view = render_virtual_imgs(args, poses_perturb0, img_idxs, hwf, device, render_kwargs_test,
                                              world_setup_dict)


        return


def train():
    #print(parser.format_values())

    # Load data
    if args.dataset_type == '7Scenes':

        train_dl, val_dl, test_dl, hwf, i_split, near, far = load_7Scenes_dataloader(args)
        near = near
        far = far
        print('NEAR FAR', near, far)
        train_feature(args, train_dl, val_dl, test_dl, hwf, i_split, near, far)
        return

    elif args.dataset_type == 'Cambridge':

        train_dl, val_dl, test_dl, hwf, i_split, near, far = load_Cambridge_dataloader(args)
        near = near
        far = far
        print("hwf")
        print(hwf)
        print('NEAR FAR', near, far)
        train_feature(args, train_dl, val_dl, test_dl, hwf, i_split, near, far)
        return

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return


if __name__ == "__main__":
    train()