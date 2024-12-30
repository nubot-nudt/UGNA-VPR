from collections import namedtuple
from os.path import join
from PIL import Image
import faiss
import numpy as np
from pynvml import *
from scipy import stats
from scipy.io import loadmat
from scipy.optimize import least_squares
from skimage import io
import yaml
import torch
from utils import parser, util
from evaluation.pretrained_model import PretrainedModel
from dotmap import DotMap
from networks.mobilenet import TeacherNet
import imageio
from script.dm.direct_pose_model import fix_coord_supp
from script.models.rendering import render, render_path
from utils.util import get_image_to_tensor_balanced, coordinate_transformation
from copy import deepcopy
import torch.nn.functional as F
from nerf_init import render_path_init, render_init

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
# x rotation
rot_phi = lambda phi: np.array([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]], dtype=float)

# y rotation
rot_theta = lambda th: np.array([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]], dtype=float)

# z rotation
rot_psi = lambda psi: np.array([
    [np.cos(psi), -np.sin(psi), 0, 0],
    [np.sin(psi), np.cos(psi), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]], dtype=float)


def linear_fit(x, y, w, report_error=False):
    def cost(p, x, y, w):
        k = p[0]
        b = p[1]
        error = y - (k * x + b)
        error *= w
        return error

    p_init = np.array([-1, 1])
    ret = least_squares(cost, p_init, args=(x, y, w), verbose=0)
    # print(ret['x'][0], ret['x'][1], )
    y_fitted = ret['x'][0] * x + ret['x'][1]
    error = ret['cost']
    if report_error:
        return y_fitted, error
    else:
        return y_fitted


def reduce_sigma(sigma, std_or_sq, log_or_linear, hmean_or_mean):
    '''
    input sigma: sigma^2, ([1, D])
    output sigma: sigma, (1)
    '''
    if log_or_linear == 'log':
        print('log')
        sigma = np.log(sigma)
    elif log_or_linear == 'linear':
        pass
    else:
        raise NameError('undefined')

    if std_or_sq == 'std':
        sigma = np.sqrt(sigma)
    elif std_or_sq == 'sq':
        pass
    else:
        raise NameError('undefined')

    if hmean_or_mean == 'hmean':
        sigma = stats.hmean(sigma, axis=1)  # ([numQ,])
    elif hmean_or_mean == 'mean':
        sigma = np.mean(sigma, axis=1)  # ([numQ,])
    else:
        raise NameError('undefined')

    return sigma


def schedule_device():
    ''' output id of the graphic card with most free memory
    '''
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    frees = []
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        # print("GPU", i, ":", nvmlDeviceGetName(handle))
        info = nvmlDeviceGetMemoryInfo(handle)
        frees.append(info.free / 1e9)
    nvmlShutdown()
    # print(frees)
    id = frees.index(max(frees))
    # print(id)
    return id


def light_log(path, args):
    with open(join(path, 'screen.log'), 'a') as f:
        for arg in args:
            f.write(arg)
            f.flush()
            print(arg, end='')


def cal_recall_from_embeddings(gt, qFeat, dbFeat):
    n_values = [1, 5, 10]

    # ---------------------------------------------------- sklearn --------------------------------------------------- #
    # knn = NearestNeighbors(n_jobs=-1)
    # knn.fit(dbFeat)
    # dists, predictions = knn.kneighbors(qFeat, len(dbFeat))

    # --------------------------------- use faiss to do NN search -------------------------------- #
    faiss_index = faiss.IndexFlatL2(qFeat.shape[1])
    faiss_index.add(dbFeat)
    dists, predictions = faiss_index.search(qFeat, max(n_values))  # the results is sorted

    recall_at_n = cal_recall(predictions, gt, n_values)
    return recall_at_n


def cal_recall(ranks, pidx, ks):
    #  ranks: preds, pidx:gt, ks: n_values
    recall_at_k = np.zeros(len(ks))
    q_id = []
    # print(ranks.shape[0])
    for qidx in range(ranks.shape[0]):
        found = False
        for i, k in enumerate(ks):
            if np.sum(np.in1d(ranks[qidx, :k], pidx[qidx])) > 0:
                recall_at_k[i:] += 1
                if k == 1:
                    found = True
                break
        if not found:
            q_id.append(qidx)
    recall_at_k /= ranks.shape[0]
    # print(q_id) # recall@1 没找到的q的id
    # print(len(q_id))
    return recall_at_k * 100.0, q_id


def cal_apk(pidx, rank, k):
    if len(rank) > k:
        rank = rank[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(rank):
        if p in pidx and p not in rank[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(pidx), k) * 100.0


def cal_mapk(ranks, pidxs, k):
    return np.mean([cal_apk(a, p, k) for a, p in zip(pidxs, ranks)])


def get_zoomed_bins(sigma, num_of_bins):
    s_min = np.min(sigma)
    s_max = np.max(sigma)
    print(s_min, s_max)
    bins_parent = np.linspace(s_min, s_max, num=num_of_bins)
    k = 0
    while True:
        indices = []
        bins_child = np.linspace(bins_parent[0], bins_parent[-1 - k], num=num_of_bins)
        for index in range(num_of_bins - 1):
            target_q_ind_l = np.where(sigma >= bins_child[index])
            if index != num_of_bins - 2:
                target_q_ind_r = np.where(sigma < bins_child[index + 1])
            else:
                target_q_ind_r = np.where(sigma <= bins_child[index + 1])
            target_q_ind = np.intersect1d(target_q_ind_l[0], target_q_ind_r[0])
            indices.append(target_q_ind)
        # if len(indices[-1]) > int(sigma.shape[0] * 0.0005):
        if len(indices[-1]) > int(sigma.shape[0] * 0.001) or k == num_of_bins - 2:
            break
        else:
            k = k + 1
    # print('{:.3f}'.format(sum([len(x) for x in indices]) / sigma.shape[0]), [len(x) for x in indices])
    # print('k=', k)
    return indices, bins_child, k


def bin_pr(preds, dists, gt, vis=False):
    # dists_m = np.around(dists[:, 0], 2)          # (4620,)
    # dists_u = np.array(list(set(dists_m)))
    # dists_u = np.sort(dists_u)                   # small > large

    dists_u = np.linspace(np.min(dists[:, 0]), np.max(dists[:, 0]), num=100)

    recalls = []
    precisions = []
    for th in dists_u:
        TPCount = 0
        FPCount = 0
        FNCount = 0
        TNCount = 0
        for index_q in range(dists.shape[0]):
            # Positive
            if dists[index_q, 0] < th:
                # True
                if np.any(np.in1d(preds[index_q, 0], gt[index_q])):
                    TPCount += 1
                else:
                    FPCount += 1
            else:
                if np.any(np.in1d(preds[index_q, 0], gt[index_q])):
                    FNCount += 1
                else:
                    TNCount += 1
        assert TPCount + FPCount + FNCount + TNCount == dists.shape[0], 'Count Error!'
        if TPCount + FNCount == 0 or TPCount + FPCount == 0:
            # print('zero')
            continue
        recall = TPCount / (TPCount + FNCount)
        precision = TPCount / (TPCount + FPCount)
        recalls.append(recall)
        precisions.append(precision)
    if vis:
        from matplotlib import pyplot as plt
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.plot(recalls, precisions)
        ax.set_title('Precision-Recall')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.savefig('pr.png', dpi=200)
    return recalls, precisions


def parse_dbStruct_pitts(path):
    dbStruct = namedtuple('dbStruct',
                          ['whichSet', 'dataset', 'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ', 'posDistThr',
                           'posDistSqThr', 'nonTrivPosDistSqThr'])

    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    dataset = 'pitts'

    whichSet = matStruct[0].item()

    # .mat file is generated by python, I replace the use of cell (in Matlab) with char (in Python)
    dbImage = [f[0].item() for f in matStruct[1]]
    # dbImage = matStruct[1]
    utmDb = matStruct[2].T
    # utmDb = matStruct[2]

    # .mat file is generated by python, I replace the use of cell (in Matlab) with char (in Python)
    qImage = [f[0].item() for f in matStruct[3]]
    # qImage = matStruct[3]
    utmQ = matStruct[4].T
    # utmQ = matStruct[4]

    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    posDistThr = matStruct[7].item()
    posDistSqThr = matStruct[8].item()
    nonTrivPosDistSqThr = matStruct[9].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, utmQ, numDb, numQ, posDistThr, posDistSqThr,
                    nonTrivPosDistSqThr)


def cal_hs(img_path):
    img = io.imread(img_path, as_gray=True).reshape(-1, 1)
    counts, bins = np.histogram((img * 255).astype(np.int16), np.arange(0, 256, 1))
    counts = counts / np.sum(counts)
    cumulative = np.cumsum(counts)
    in_min = np.min((img * 255).astype(np.int16))
    in_max = np.max((img * 255).astype(np.int16))
    per_75 = np.argwhere(cumulative < 0.75)[-1]
    per_25 = np.argwhere(cumulative < 0.25)[-1]
    hs = (per_75 - per_25) / 255
    return hs


def gen_pose(q_id, preds, dir):
    # q_id需要进行nerf增加数据的idx
    dir_q = os.path.join(dir, "poses4")
    poseq, scene = read_pose(dir_q)
    poseq = poseq[q_id]
    selected_scenes = [scene[idx] for idx in q_id]
    return poseq, selected_scenes


def find_image(dir):
    dir_d = os.path.join(dir, "rgb")
    image_d = read_image(dir_d)

    return image_d


from torchvision.datasets.folder import default_loader


def load_image(filename, loader=default_loader):
    try:
        img = loader(filename)
    except IOError as e:
        print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
        return None
    except:
        print('Could not load image {:s}, unexpected error'.format(filename))
        return None
    return img


def read_pose(path):
    folder_path = path

    # 获取文件夹中的所有 txt 文件
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    last_row = np.array([0, 0, 0, 1])
    # 初始化一个空列表，用于存储所有数据
    data_list = []
    prefixes = []
    # 逐个读取每个 txt 文件的数据
    for file in txt_files:
        file_path = os.path.join(folder_path, file)
        prefix = file.split("_")[0]
        prefixes.append(prefix)
        # 读取 txt 文件数据
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # 将数据转换为 numpy 数组，并添加到列表中
            data = np.array([[float(val) for val in line.strip().split()] for line in lines])
            # print(data.shape[0])
            if data.shape[0] == 3:
                data = np.vstack([data, last_row])
            data_list.append(data)

    # 将列表转换为 numpy 数组
    all_data = np.array(data_list)
    # print(all_data)
    all_data = torch.tensor(all_data)
    if (all_data.shape[-2] == 3):
        extra_row = torch.tensor([0, 0, 0, 1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        extra_rows = extra_row.expand(all_data.size(0), 1, 4)
        all_data = torch.cat([all_data, extra_rows], dim=1)
    all_data = all_data.float()
    # print(poses)
    # print(poses.shape)
    return all_data, prefixes


def read_image(path):
    dpath = path
    dfiles = os.listdir(dpath)
    dImage = []

    # 遍历文件夹中的所有文件
    for file in dfiles:
        # 检查文件是否为PNG图片
        if file.endswith(".png"):
            # 将图片名字添加到列表中
            dImage.append(file)
    dImages = [join(dpath, dbIm) for dbIm in dImage]

    return dImages


def NBP_Cam(model_name, images_ref, poses_ref, poses_target):
    log_path = os.path.join("logs", model_name)
    ON, _, _, _ = poses_ref.shape

    assert os.path.exists(log_path), "experiment does not exist"
    with open(f"{log_path}/training_setup.yaml", "r") as config_file:
        cfg = yaml.safe_load(config_file)

    checkpoint_path = os.path.join(log_path, "checkpoints", "best.ckpt")
    assert os.path.exists(checkpoint_path), "checkpoint does not exist"
    ckpt_file = torch.load(checkpoint_path)
    gpu_id = "0"
    gpu_id = list(map(int, gpu_id.split()))
    device = util.get_cuda(gpu_id[0])

    images = images_ref.to(device)
    poses = poses_ref.to(device)

    novel_pose = poses_target.to(device)
    # novel_pose = novel_pose.unsqueeze(0)

    # adjust f and c
    fx = 745
    fy = 745
    cx = 427
    cy = 240
    focal = torch.tensor((fx, fy), dtype=torch.float32)
    c = torch.tensor((cx, cy), dtype=torch.float32)

    scale_h = 224 / 480
    scale_w = 224 / 854
    focal[0] *= scale_w
    focal[1] *= scale_h
    c[0] *= scale_w
    c[1] *= scale_h

    focal = focal.to(device)
    c = c.to(device)
    focal = focal.unsqueeze(0).expand(ON, -1)
    c = c.unsqueeze(0).expand(ON, -1)

    # print(focal.shape)
    # print(c.shape)
    model = PretrainedModel(cfg["model"], ckpt_file, device, gpu_id)
    # encoder = encoder #encoder in this epoch

    # encoder has been trained
    # encoder = TeacherNet()
    # pretrained_weights_path = 'logs/ckpt_best.pth.tar'
    # pretrained_state_dict = torch.load(pretrained_weights_path)
    # encoder.load_state_dict(pretrained_state_dict["state_dict"])
    # encoder = encoder.to(device)
    with torch.no_grad():
        model.network.encode(
            images,
            poses,
            focal,
            c,
        )

    predict = DotMap(model.network(novel_pose))

    uncertainty = predict.all_uncertainty
    # print(uncertainty.shape)
    # 计算按照从小到大排列的索引
    sorted_indices = torch.argsort(uncertainty)
    # 反转顺序，得到按照从高到低排列的索引
    descending_indices = sorted_indices.flip(dims=[1])

    # 对比实验
    # data = [[0, 1, 2, 3, 4, 5]]
    # descending_indices = torch.tensor(data, device=device)
    return descending_indices[:, :3]


def NBP_NEU(model_name, images_ref, poses_ref, poses_target, device):
    log_path = os.path.join("logs", model_name)
    ON, _, _, _ = poses_ref.shape

    assert os.path.exists(log_path), "experiment does not exist"
    with open(f"{log_path}/training_setup.yaml", "r") as config_file:
        cfg = yaml.safe_load(config_file)

    checkpoint_path = os.path.join(log_path, "checkpoints", "best.ckpt")
    assert os.path.exists(checkpoint_path), "checkpoint does not exist"
    ckpt_file = torch.load(checkpoint_path)
    gpu_id = "0"

    images = images_ref.to(device)
    poses = poses_ref.to(device)

    novel_pose = poses_target.to(device)
    # novel_pose = novel_pose.unsqueeze(0)

    # adjust f and c
    fx = 211
    fy = 211
    cx = 240
    cy = 135
    focal = torch.tensor((fx, fy), dtype=torch.float32)
    c = torch.tensor((cx, cy), dtype=torch.float32)

    scale_h = 224 / 270
    scale_w = 224 / 480
    focal[0] *= scale_w
    focal[1] *= scale_h
    c[0] *= scale_w
    c[1] *= scale_h
    # images = F.interpolate(images, size=[224, 224], mode="area")

    focal = focal.to(device)
    c = c.to(device)
    focal = focal.unsqueeze(0).expand(ON, -1)
    c = c.unsqueeze(0).expand(ON, -1)

    # print(focal.shape)
    # print(c.shape)
    model = PretrainedModel(cfg["model"], ckpt_file, device, gpu_id)
    # encoder = encoder #encoder in this epoch

    # encoder has been trained
    # encoder = TeacherNet()
    # pretrained_weights_path = 'logs/ckpt_best.pth.tar'
    # pretrained_state_dict = torch.load(pretrained_weights_path)
    # encoder.load_state_dict(pretrained_state_dict["state_dict"])
    # encoder = encoder.to(device)
    with torch.no_grad():
        model.network.encode(
            images,
            poses,
            focal,
            c,
        )

    predict = DotMap(model.network(novel_pose))

    uncertainty = predict.all_uncertainty
    # print(uncertainty.shape)
    # 计算按照从小到大排列的索引
    sorted_indices = torch.argsort(uncertainty)
    # 反转顺序，得到按照从高到低排列的索引
    descending_indices = sorted_indices.flip(dims=[1])

    # 对比实验
    # data = [[0, 1, 2, 3, 4, 5]]
    # descending_indices = torch.tensor(data, device=device)
    return descending_indices[:, :3]


def NBP_SIASUN(model_name, images_ref, poses_ref, poses_target, device):
    log_path = os.path.join("logs", model_name)
    ON, _, _, _ = poses_ref.shape

    assert os.path.exists(log_path), "experiment does not exist"
    with open(f"{log_path}/training_setup.yaml", "r") as config_file:
        cfg = yaml.safe_load(config_file)

    checkpoint_path = os.path.join(log_path, "checkpoints", "best.ckpt")
    assert os.path.exists(checkpoint_path), "checkpoint does not exist"
    ckpt_file = torch.load(checkpoint_path)
    gpu_id = "0"

    images = images_ref.to(device)
    poses = poses_ref.to(device)

    novel_pose = poses_target.to(device)
    # novel_pose = novel_pose.unsqueeze(0)

    # adjust f and c
    fx = 396
    fy = 396
    cx = 240
    cy = 135
    focal = torch.tensor((fx, fy), dtype=torch.float32)
    c = torch.tensor((cx, cy), dtype=torch.float32)

    scale_h = 224 / 270
    scale_w = 224 / 480
    focal[0] *= scale_w
    focal[1] *= scale_h
    c[0] *= scale_w
    c[1] *= scale_h
    # images = F.interpolate(images, size=[224, 224], mode="area")

    focal = focal.to(device)
    c = c.to(device)
    focal = focal.unsqueeze(0).expand(ON, -1)
    c = c.unsqueeze(0).expand(ON, -1)

    # print(focal.shape)
    # print(c.shape)
    model = PretrainedModel(cfg["model"], ckpt_file, device, gpu_id)
    # encoder = encoder #encoder in this epoch

    # encoder has been trained
    # encoder = TeacherNet()
    # pretrained_weights_path = 'logs/ckpt_best.pth.tar'
    # pretrained_state_dict = torch.load(pretrained_weights_path)
    # encoder.load_state_dict(pretrained_state_dict["state_dict"])
    # encoder = encoder.to(device)
    with torch.no_grad():
        model.network.encode(
            images,
            poses,
            focal,
            c,
        )

    predict = DotMap(model.network(novel_pose))

    uncertainty = predict.all_uncertainty
    # print(uncertainty.shape)
    # 计算按照从小到大排列的索引
    sorted_indices = torch.argsort(uncertainty)
    # 反转顺序，得到按照从高到低排列的索引g
    descending_indices = sorted_indices.flip(dims=[1])

    # 对比实验
    # data = [[0, 1, 2, 3, 4, 5]]
    # descending_indices = torch.tensor(data, device=device)
    return descending_indices[:, :3]


def render_virtual_Cam_imgs(args, pose_perturb, img_idxs, hwf, device, render_kwargs_test, world_setup_dict, epoch,
                            scene):
    ''' render nerfw imgs, save unscaled pose and results'''
    H, W, focal = hwf
    rgb_list = []
    rgbs = []
    # 创建保存文件夹
    save_folder_rgb = "Cambridge/CambridgeNerf_train1_4/train/database/rgb"
    save_folder_poses = "Cambridge/CambridgeNerf_train1_4/train/database/poses"
    save_folder_poses4 = "Cambridge/CambridgeNerf_train1_4/train/database/poses4"
    os.makedirs(save_folder_rgb, exist_ok=True)
    os.makedirs(save_folder_poses, exist_ok=True)
    os.makedirs(save_folder_poses4, exist_ok=True)
    pose_perturb = torch.Tensor(pose_perturb)
    # inference nerfw and save rgb, target, pose
    for batch_idx in range(pose_perturb.shape[0]):
        # if batch_idx % 10 == 0:
        # print("renders RVS {}/total {}".format(batch_idx, pose_perturb.shape[0]))

        pose = pose_perturb[batch_idx]
        img_idx = img_idxs[batch_idx].to(device)
        pose = pose.reshape(3, 4)
        pose_nerf = pose.clone()
        # print(pose_nerf.shape)
        # print("fix_coord_supp")
        # rescale the predicted pose to nerf scales
        pose_nerf = fix_coord_supp(args, pose_nerf[None, ...].cpu(), world_setup_dict)
        # if(batch_idx==0):
        # print(pose_nerf)
        # generate nerf image
        with torch.no_grad():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            """
            if args.tinyimg:
                rgb, _, _, _ = render(int(H//args.tinyscale), int(W//args.tinyscale), focal/args.tinyscale, chunk=args.chunk, c2w=pose_nerf[0,:3,:4].to(device), retraw=False, img_idx=img_idx, **render_kwargs_test)
                # convert rgb to B,C,H,W
                rgb = rgb[None,...].permute(0,3,1,2)
                # upsample rgb to hwf size
                rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
                # convert rgb back to H,W,C format
                rgb = rgb[0].permute(1,2,0)

            else:
            """
            rgb, _, _, _ = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf[0, :3, :4].to(device), retraw=False,
                                  img_idx=img_idx, **render_kwargs_test)
            rgbs.append(rgb.cpu().numpy())
            torch.set_default_tensor_type('torch.FloatTensor')
        rgb_list.append(rgb.cpu())
        # 保存图像
        rgb8_f = to8b(rgbs[-1])  # save coarse+fine img
        filename = os.path.join(save_folder_rgb, 'zadd_{}_{:05d}_{}.png'.format(scene, batch_idx, epoch))
        imageio.imwrite(filename, rgb8_f)
        # 保存位姿
        poses_filename = os.path.join(save_folder_poses, "zadd_{}_{:05d}_{}.txt".format(scene, batch_idx, epoch))
        poses4_filename = os.path.join(save_folder_poses4, "zadd_{}_{:05d}_{}.txt".format(scene, batch_idx, epoch))
        # 将中心化的坐标系转回原来的坐标系
        # pose_avg_stats_file = osp.join(args.data_dir, scene) + '/pose_avg_stats.txt'
        pose = pose.cpu().numpy()
        pose_avg_stats_file = "data" + '/poses_avg_stats/' + scene + ".txt"
        pose_loc, pose_t = inv_concered(pose, pose_avg_stats_file, batch_idx, scene)
        np.savetxt(poses_filename, pose_loc)
        np.savetxt(poses4_filename, pose_t)
    rgbs = torch.stack(rgb_list).detach()
    return rgbs


def render_virtual_Cam_imgs_init(args, pose_perturb, img_idxs, hwf, device, render_kwargs_test, world_setup_dict, epoch,
                                 scene):
    ''' render nerfw imgs, save unscaled pose and results'''
    H, W, focal = hwf
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])
    rgb_list = []
    rgbs = []
    # 创建保存文件夹
    save_folder_rgb = "Cambridge/Cambridgenerf1_4_init/train/database/rgb"
    save_folder_poses = "Cambridge/Cambridgenerf1_4_init/train/database/poses"
    save_folder_poses4 = "Cambridge/Cambridgenerf1_4_init/train/database/poses4"
    os.makedirs(save_folder_rgb, exist_ok=True)
    os.makedirs(save_folder_poses, exist_ok=True)
    os.makedirs(save_folder_poses4, exist_ok=True)
    pose_perturb = torch.Tensor(pose_perturb)
    # inference nerfw and save rgb, target, pose
    for batch_idx in range(pose_perturb.shape[0]):
        # if batch_idx % 10 == 0:
        # print("renders RVS {}/total {}".format(batch_idx, pose_perturb.shape[0]))

        pose = pose_perturb[batch_idx]
        img_idx = img_idxs[batch_idx].to(device)
        pose = pose.reshape(3, 4)
        pose_nerf = pose.clone()
        # print(pose_nerf.shape)
        # print("fix_coord_supp")
        # rescale the predicted pose to nerf scales
        pose_nerf = fix_coord_supp(args, pose_nerf[None, ...].cpu(), world_setup_dict)
        # if(batch_idx==0):
        # print(pose_nerf)
        # generate nerf image
        with torch.no_grad():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            """
            if args.tinyimg:
                rgb, _, _, _ = render(int(H//args.tinyscale), int(W//args.tinyscale), focal/args.tinyscale, chunk=args.chunk, c2w=pose_nerf[0,:3,:4].to(device), retraw=False, img_idx=img_idx, **render_kwargs_test)
                # convert rgb to B,C,H,W
                rgb = rgb[None,...].permute(0,3,1,2)
                # upsample rgb to hwf size
                rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
                # convert rgb back to H,W,C format
                rgb = rgb[0].permute(1,2,0)

            else:
            """
            pose_nerf = pose_nerf.to(device)
            rgb, _, _, _ = render_init(H, W, K, chunk=args.chunk, c2w=pose_nerf[0, :3, :4], **render_kwargs_test)
            rgbs.append(rgb.cpu().numpy())
            torch.set_default_tensor_type('torch.FloatTensor')
        rgb_list.append(rgb.cpu())
        # 保存图像
        rgb8_f = to8b(rgbs[-1])  # save coarse+fine img
        filename = os.path.join(save_folder_rgb, 'zadd_{}_{:05d}_{}.png'.format(scene, batch_idx, epoch))
        imageio.imwrite(filename, rgb8_f)
        # 保存位姿
        poses_filename = os.path.join(save_folder_poses, "zadd_{}_{:05d}_{}.txt".format(scene, batch_idx, epoch))
        poses4_filename = os.path.join(save_folder_poses4, "zadd_{}_{:05d}_{}.txt".format(scene, batch_idx, epoch))
        # 将中心化的坐标系转回原来的坐标系
        # pose_avg_stats_file = osp.join(args.data_dir, scene) + '/pose_avg_stats.txt'
        pose = pose.cpu().numpy()
        pose_avg_stats_file = "data" + '/poses_avg_stats/' + scene + ".txt"
        pose_loc, pose_t = inv_concered(pose, pose_avg_stats_file, batch_idx, scene)
        np.savetxt(poses_filename, pose_loc)
        np.savetxt(poses4_filename, pose_t)
    rgbs = torch.stack(rgb_list).detach()
    return rgbs


def inv_concered(pose, pose_avg_stats_file, batch_idx, scene):
    E_pose = np.eye(4)
    E_pose[:3] = pose
    E_pose[:3, :3] = E_pose[:3, :3] @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    E_pose[:3, :3] = -E_pose[:3, :3]

    E_pose = np.linalg.inv(rot_phi(180 / 180. * np.pi)) @ E_pose

    # pose_un=E_pose[:3]

    pose_avg = np.loadtxt(pose_avg_stats_file)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg

    last_row = np.array([0, 0, 0, 1])
    # poses_homo = np.vstack([pose_un, last_row])
    poses_centered = E_pose

    pose_t = pose_avg_homo @ poses_centered
    # if (batch_idx == 4):
    # print(pose_t)
    pose_t = pose_t[:3, :]
    pose_t = np.vstack([pose_t, last_row])
    if (scene == "GreatCourt"):
        element_1 = pose_t[0, 3]
        element_2 = pose_t[1, 3]
    elif (scene == "KingsCollege"):
        element_1 = pose_t[0, 3] + 1000
        element_2 = pose_t[1, 3] + 1000
    elif (scene == "OldHospital"):
        element_1 = pose_t[0, 3] + 2000
        element_2 = pose_t[1, 3] + 2000
    elif (scene == "ShopFacade"):
        element_1 = pose_t[0, 3] + 3000
        element_2 = pose_t[1, 3] + 3000
    elif (scene == "StMarysChurch"):
        element_1 = pose_t[0, 3] + 4000
        element_2 = pose_t[1, 3] + 4000
    else:
        print("Error happen")
    # 将提取的元素组成一个新的矩阵
    pose_loc = np.array([[element_1, element_2]])
    return pose_loc, pose_t


def fix_coord(poses, scenes, pose_avg_stats_file='data/poses_avg_stats'):
    ''' fix coord for 7 Scenes to align with llff style dataset '''
    ON, N, H, W = poses.shape
    poses = poses.view(ON * N, H, W)
    # get all poses (train+val)
    all_poses = poses[:, :3, :4].reshape(poses.shape[0], 12)

    # all_poses = np.concatenate([train_poses, val_poses])

    # Center the poses for ndc
    all_poses = all_poses.reshape(all_poses.shape[0], 3, 4)
    print("all_poses[0]")
    print(all_poses.shape[0])
    Great = 0
    Kings = 0
    Old = 0
    Shop = 0
    StMarys = 0

    poses_name_Great = "GreatCourt.txt"
    pose_avg_stats_Great = os.path.join(pose_avg_stats_file, poses_name_Great)
    pose_avg_from_file_Great = np.loadtxt(pose_avg_stats_Great)
    poses_name_Kings = "KingsCollege.txt"
    pose_avg_stats_Kings = os.path.join(pose_avg_stats_file, poses_name_Kings)
    pose_avg_from_file_Kings = np.loadtxt(pose_avg_stats_Kings)
    poses_name_Old = "OldHospital.txt"
    pose_avg_stats_Old = os.path.join(pose_avg_stats_file, poses_name_Old)
    pose_avg_from_file_Old = np.loadtxt(pose_avg_stats_Old)
    poses_name_Shop = "ShopFacade.txt"
    pose_avg_stats_Shop = os.path.join(pose_avg_stats_file, poses_name_Shop)
    pose_avg_from_file_Shop = np.loadtxt(pose_avg_stats_Shop)
    poses_name_StMarys = "StMarysChurch.txt"
    pose_avg_stats_StMarys = os.path.join(pose_avg_stats_file, poses_name_StMarys)
    pose_avg_from_file_StMarys = np.loadtxt(pose_avg_stats_StMarys)

    for i in range(all_poses.shape[0]):
        if (scenes[i] == "GreatCourt"):
            Great = Great + 1
        elif (scenes[i] == "KingsCollege"):
            Kings = Kings + 1
        elif (scenes[i] == "OldHospital"):
            Old = Old + 1
        elif (scenes[i] == "ShopFacade"):
            Shop = Shop + 1
        elif (scenes[i] == "StMarysChurch"):
            StMarys = StMarys + 1
        else:
            print("ERROR!!!!")
            print(scenes[i])
    idx = [Great, Great + Kings, Great + Kings + Old, Great + Kings + Old + Shop, Great + Kings + Old + Shop + StMarys]
    all_poses_Great = all_poses[:Great]
    all_poses_Kings = all_poses[Great:Great + Kings]
    all_poses_Old = all_poses[Great + Kings:Great + Kings + Old]
    all_poses_Shop = all_poses[Great + Kings + Old:Great + Kings + Old + Shop]
    all_poses_StMarys = all_poses[Great + Kings + Old + Shop:Great + Kings + Old + Shop + StMarys]
    # Here we use either pre-stored pose average stats or calculate pose average stats on the flight to center the poses

    all_poses_Great, pose_avg_Great = center_poses(all_poses_Great, pose_avg_from_file_Great)
    all_poses_Kings, pose_avg_Kings = center_poses(all_poses_Kings, pose_avg_from_file_Kings)
    all_poses_Old, pose_avg_Old = center_poses(all_poses_Old, pose_avg_from_file_Old)
    all_poses_Shop, pose_avg_Shop = center_poses(all_poses_Shop, pose_avg_from_file_Shop)
    all_poses_StMarys, pose_avg_StMarys = center_poses(all_poses_StMarys, pose_avg_from_file_StMarys)

    all_poses_Great = torch.from_numpy(all_poses_Great)
    all_poses_Kings = torch.from_numpy(all_poses_Kings)
    all_poses_Old = torch.from_numpy(all_poses_Old)
    all_poses_Shop = torch.from_numpy(all_poses_Shop)
    all_poses_StMarys = torch.from_numpy(all_poses_StMarys)

    all_poses = (torch.cat((all_poses_Great, all_poses_Kings, all_poses_Old, all_poses_Shop, all_poses_StMarys), dim=0))
    all_poses = all_poses.numpy()
    # Correct axis to LLFF Style y,z -> -y,-z
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(all_poses), 1, 1))  # (N_images, 1, 4)
    all_poses = np.concatenate([all_poses, last_row], 1)

    # rotate tpose 90 degrees at x axis # only corrected translation position
    all_poses = rot_phi(180 / 180. * np.pi) @ all_poses

    # correct view direction except mirror with gt view
    all_poses[:, :3, :3] = -all_poses[:, :3, :3]

    # camera direction mirror at x axis mod1 R' = R @ mirror matrix
    # ref: https://gamedev.stackexchange.com/questions/149062/how-to-mirror-reflect-flip-a-4d-transformation-matrix
    all_poses[:, :3, :3] = all_poses[:, :3, :3] @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    all_poses = all_poses[:, :3, :4]
    # print("after centered")
    # print(all_poses[0])
    bounds = np.array([0, 10])  # manual tuned

    # Return all poses to dataset loaders
    all_poses = all_poses.reshape(all_poses.shape[0], 12)
    return all_poses, bounds, idx


def center_poses(poses, pose_avg_from_file=None):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)
        pose_avg_from_file: if not None, pose_avg is loaded from pose_avg_stats.txt

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    poses = poses.cpu()
    pose_avg = pose_avg_from_file

    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation (4,4)
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg  # np.linalg.inv(pose_avg_homo)


import torch
import numpy as np


def perturb_single_render_pose(poses, x, angle, n):
    """
    对给定的姿态张量进行随机扰动，产生新的姿态张量。

    Args:
    - poses (torch.Tensor): 姿态张量，形状为[58, 4, 4]
    - x (float): 平移的范围
    - angle (float): 角度扰动的范围
    - n (int): 生成新姿态的数量

    Returns:
    - perturbed_poses (torch.Tensor): 新姿态张量，形状为[58, n, 4, 4]
    """
    np.random.seed(42)  # for mobilenet
    device = torch.device("cuda")
    poses = poses.numpy()
    # 初始化一个空张量来存储扰动后的姿态
    new_c2w = np.zeros((poses.shape[0], n, 3, 4))
    new_poses = np.zeros((poses.shape[0], n, 4, 4))
    # print(new_c2w.shape)
    for i in range(poses.shape[0]):
        for j in range(n):
            new_c2w[i, j] = poses[i, :3, :]
            # print(new_c2w[i,j].shape)
            loc = deepcopy(new_c2w[i, j, :, 3])  # this is a must
            # 扰动旋转姿态
            rot_rand = np.random.uniform(-angle, angle, 3)  # in degrees
            theta, phi, psi = rot_rand

            new_c2w[i, j] = perturb_rotation(new_c2w[i, j], theta, phi, psi)

            trans_rand_xy = np.random.uniform(-x, x, 2)
            # 生成第三个元素的随机扰动，从-0.15到0.15之间
            trans_rand_z = np.random.uniform(-0.10, 0.10, 1)  # 0.15 Cambridge  0.1 NEU
            # 合并成一个长度为3的数组
            trans_rand = np.concatenate((trans_rand_xy, trans_rand_z))
            new_c2w[i, j, :, 3] = loc + trans_rand  # perturb pos between -1 to 1
            # 创建一个包含四个元素的列表
            vector = np.array([0, 0, 0, 1])
            new_poses[i, j] = np.vstack((new_c2w[i, j], vector))
    # 将矩阵和向量进行垂直堆叠
    # print(new_c2w.shape)
    # print(new_poses)
    new_poses = torch.Tensor(new_poses).to(device)
    return new_poses


def perturb_rotation(c2w, theta, phi, psi=0):
    last_row = np.array([[0, 0, 0, 1]])  # np.tile(np.array([0, 0, 0, 1]), (1, 1))  # (N_images, 1, 4)
    c2w = np.concatenate([c2w, last_row], 0)  # (N_images, 4, 4) homogeneous coordinate
    # print("c2w", c2w)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = rot_psi(psi / 180. * np.pi) @ c2w
    c2w = c2w[:3, :4]
    # print("updated c2w", c2w)
    return c2w


def render_virtual_NEU_imgs(args, pose_perturb, img_idxs, hwf, device, render_kwargs_test, world_setup_dict, epoch,
                            scene):
    ''' render nerfw imgs, save unscaled pose and results'''
    H, W, focal = hwf
    rgb_list = []
    rgbs = []
    # 创建保存文件夹
    save_folder_rgb = "Cambridge/NEUnight/train/database/rgb"
    save_folder_poses = "Cambridge/NEUnight/train/database/poses"
    save_folder_poses4 = "Cambridge/NEUnight/train/database/poses4"
    os.makedirs(save_folder_rgb, exist_ok=True)
    os.makedirs(save_folder_poses, exist_ok=True)
    os.makedirs(save_folder_poses4, exist_ok=True)
    pose_perturb = torch.Tensor(pose_perturb)
    # inference nerfw and save rgb, target, pose
    for batch_idx in range(pose_perturb.shape[0]):
        # if batch_idx % 10 == 0:
        # print("renders RVS {}/total {}".format(batch_idx, pose_perturb.shape[0]))
        pose = pose_perturb[batch_idx]
        img_idx = img_idxs[batch_idx].to(device)
        pose = pose.reshape(3, 4)
        pose_nerf = pose.clone()
        # rescale the predicted pose to nerf scales
        pose_nerf = fix_coord_supp(args, pose_nerf[None, ...].cpu(), world_setup_dict)
        # generate nerf image
        with torch.no_grad():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            """
            if args.tinyimg:
                rgb, _, _, _ = render(int(H//args.tinyscale), int(W//args.tinyscale), focal/args.tinyscale, chunk=args.chunk, c2w=pose_nerf[0,:3,:4].to(device), retraw=False, img_idx=img_idx, **render_kwargs_test)
                # convert rgb to B,C,H,W
                rgb = rgb[None,...].permute(0,3,1,2)
                # upsample rgb to hwf size
                rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
                # convert rgb back to H,W,C format
                rgb = rgb[0].permute(1,2,0)

            else:
            """
            rgb, _, _, _ = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf[0, :3, :4].to(device), retraw=False,
                                  img_idx=img_idx, **render_kwargs_test)
            rgbs.append(rgb.cpu().numpy())
            torch.set_default_tensor_type('torch.FloatTensor')
        rgb_list.append(rgb.cpu())
        # 保存图像
        rgb8_f = to8b(rgbs[-1])  # save coarse+fine img
        filename = os.path.join(save_folder_rgb, 'zadd_{}_{:05d}_{}.png'.format(scene, batch_idx, epoch))
        imageio.imwrite(filename, rgb8_f)
        # 保存位姿
        poses_filename = os.path.join(save_folder_poses, "zadd_{}_{:05d}_{}.txt".format(scene, batch_idx, epoch))
        poses4_filename = os.path.join(save_folder_poses4, "zadd_{}_{:05d}_{}.txt".format(scene, batch_idx, epoch))
        # 将中心化的坐标系转回原来的坐标系
        pose = pose.cpu().numpy()
        # pose_avg_stats_file = "data" + '/pose_avg_stats/' + scene + ".txt"
        pose_loc, pose_t = inv_concered_NEU(pose, batch_idx, scene)
        np.savetxt(poses_filename, pose_loc)
        np.savetxt(poses4_filename, pose_t)
    rgbs = torch.stack(rgb_list).detach()
    return rgbs


def render_virtual_NEU_imgs_init(args, pose_perturb, img_idxs, hwf, device, render_kwargs_test, world_setup_dict, epoch,
                                 scene):
    ''' render nerfw imgs, save unscaled pose and results'''
    H, W, focal = hwf
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])
    rgb_list = []
    rgbs = []
    # 创建保存文件夹
    save_folder_rgb = "Cambridge/NEUNerf_init/train/database/rgb"
    save_folder_poses = "Cambridge/NEUNerf_init/train/database/poses"
    save_folder_poses4 = "Cambridge/NEUNerf_init/train/database/poses4"
    os.makedirs(save_folder_rgb, exist_ok=True)
    os.makedirs(save_folder_poses, exist_ok=True)
    os.makedirs(save_folder_poses4, exist_ok=True)
    pose_perturb = torch.Tensor(pose_perturb)
    # inference nerfw and save rgb, target, pose
    for batch_idx in range(pose_perturb.shape[0]):
        # if batch_idx % 10 == 0:
        # print("renders RVS {}/total {}".format(batch_idx, pose_perturb.shape[0]))

        pose = pose_perturb[batch_idx]
        img_idx = img_idxs[batch_idx].to(device)
        pose = pose.reshape(3, 4)
        pose_nerf = pose.clone()

        # rescale the predicted pose to nerf scales
        pose_nerf = fix_coord_supp(args, pose_nerf[None, ...].cpu(), world_setup_dict)

        # generate nerf image
        with torch.no_grad():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            """
            if args.tinyimg:
                rgb, _, _, _ = render(int(H//args.tinyscale), int(W//args.tinyscale), focal/args.tinyscale, chunk=args.chunk, c2w=pose_nerf[0,:3,:4].to(device), retraw=False, img_idx=img_idx, **render_kwargs_test)
                # convert rgb to B,C,H,W
                rgb = rgb[None,...].permute(0,3,1,2)
                # upsample rgb to hwf size
                rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
                # convert rgb back to H,W,C format
                rgb = rgb[0].permute(1,2,0)

            else:
            """
            # rgb, _, _, _ = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf[0,:3,:4].to(device), retraw=False, img_idx=img_idx, **render_kwargs_test)
            pose_nerf = pose_nerf.to(device)
            rgb, _, _, _ = render_init(H, W, K, chunk=args.chunk, c2w=pose_nerf[0, :3, :4], **render_kwargs_test)
            rgbs.append(rgb.cpu().numpy())
            torch.set_default_tensor_type('torch.FloatTensor')
        rgb_list.append(rgb.cpu())
        # 保存图像
        rgb8_f = to8b(rgbs[-1])  # save coarse+fine img
        filename = os.path.join(save_folder_rgb, 'zadd_{}_{:05d}_{}.png'.format(scene, batch_idx, epoch))
        imageio.imwrite(filename, rgb8_f)
        # 保存位姿
        poses_filename = os.path.join(save_folder_poses, "zadd_{}_{:05d}_{}.txt".format(scene, batch_idx, epoch))
        poses4_filename = os.path.join(save_folder_poses4, "zadd_{}_{:05d}_{}.txt".format(scene, batch_idx, epoch))
        # 将中心化的坐标系转回原来的坐标系
        pose = pose.cpu().numpy()
        # pose_avg_stats_file = "data" + '/pose_avg_stats/' + scene + ".txt"
        pose_loc, pose_t = inv_concered_NEU(pose, batch_idx, scene)
        np.savetxt(poses_filename, pose_loc)
        np.savetxt(poses4_filename, pose_t)
    rgbs = torch.stack(rgb_list).detach()
    return rgbs


def render_virtual_SIA_imgs_init(args, pose_perturb, img_idxs, hwf, device, render_kwargs_test, world_setup_dict, epoch,
                                 scene):
    ''' render nerfw imgs, save unscaled pose and results'''
    H, W, focal = hwf
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])
    rgb_list = []
    rgbs = []
    # 创建保存文件夹
    save_folder_rgb = "Cambridge/Siasun1_3/train/database/rgb"
    save_folder_poses = "Cambridge/Siasun1_3/train/database/poses"
    save_folder_poses4 = "Cambridge/Siasun1_3/train/database/poses4"
    os.makedirs(save_folder_rgb, exist_ok=True)
    os.makedirs(save_folder_poses, exist_ok=True)
    os.makedirs(save_folder_poses4, exist_ok=True)
    pose_perturb = torch.Tensor(pose_perturb)
    # inference nerfw and save rgb, target, pose
    for batch_idx in range(pose_perturb.shape[0]):
        # if batch_idx % 10 == 0:
        # print("renders RVS {}/total {}".format(batch_idx, pose_perturb.shape[0]))

        pose = pose_perturb[batch_idx]
        img_idx = img_idxs[batch_idx].to(device)
        pose = pose.reshape(3, 4)
        pose_nerf = pose.clone()

        # rescale the predicted pose to nerf scales
        pose_nerf = fix_coord_supp(args, pose_nerf[None, ...].cpu(), world_setup_dict)

        # generate nerf image
        with torch.no_grad():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            """
            if args.tinyimg:
                rgb, _, _, _ = render(int(H//args.tinyscale), int(W//args.tinyscale), focal/args.tinyscale, chunk=args.chunk, c2w=pose_nerf[0,:3,:4].to(device), retraw=False, img_idx=img_idx, **render_kwargs_test)
                # convert rgb to B,C,H,W
                rgb = rgb[None,...].permute(0,3,1,2)
                # upsample rgb to hwf size
                rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
                # convert rgb back to H,W,C format
                rgb = rgb[0].permute(1,2,0)

            else:
            """
            # rgb, _, _, _ = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf[0,:3,:4].to(device), retraw=False, img_idx=img_idx, **render_kwargs_test)
            pose_nerf = pose_nerf.to(device)
            rgb, _, _, _ = render_init(H, W, K, chunk=args.chunk, c2w=pose_nerf[0, :3, :4], **render_kwargs_test)
            rgbs.append(rgb.cpu().numpy())
            torch.set_default_tensor_type('torch.FloatTensor')
        rgb_list.append(rgb.cpu())
        # 保存图像
        rgb8_f = to8b(rgbs[-1])  # save coarse+fine img
        filename = os.path.join(save_folder_rgb, 'zadd_{}_{:05d}_{}.png'.format(scene, batch_idx, epoch))
        imageio.imwrite(filename, rgb8_f)
        # 保存位姿
        poses_filename = os.path.join(save_folder_poses, "zadd_{}_{:05d}_{}.txt".format(scene, batch_idx, epoch))
        poses4_filename = os.path.join(save_folder_poses4, "zadd_{}_{:05d}_{}.txt".format(scene, batch_idx, epoch))
        # 将中心化的坐标系转回原来的坐标系
        pose = pose.cpu().numpy()
        # pose_avg_stats_file = "data" + '/pose_avg_stats/' + scene + ".txt"
        pose_loc, pose_t = inv_concered_SIASUN(pose, batch_idx, scene)
        np.savetxt(poses_filename, pose_loc)
        np.savetxt(poses4_filename, pose_t)
    rgbs = torch.stack(rgb_list).detach()
    return rgbs


def render_virtual_SIA_imgs(args, pose_perturb, img_idxs, hwf, device, render_kwargs_test, world_setup_dict, epoch,
                            scene):
    ''' render nerfw imgs, save unscaled pose and results'''
    H, W, focal = hwf
    rgb_list = []
    rgbs = []
    # 创建保存文件夹
    save_folder_rgb = "Cambridge/Siasun1_3/train/database/rgb"
    save_folder_poses = "Cambridge/Siasun1_3/train/database/poses"
    save_folder_poses4 = "Cambridge/Siasun1_3/train/database/poses4"
    os.makedirs(save_folder_rgb, exist_ok=True)
    os.makedirs(save_folder_poses, exist_ok=True)
    os.makedirs(save_folder_poses4, exist_ok=True)
    pose_perturb = torch.Tensor(pose_perturb)
    # inference nerfw and save rgb, target, pose
    for batch_idx in range(pose_perturb.shape[0]):
        # if batch_idx % 10 == 0:
        # print("renders RVS {}/total {}".format(batch_idx, pose_perturb.shape[0]))

        pose = pose_perturb[batch_idx]
        img_idx = img_idxs[batch_idx].to(device)
        pose = pose.reshape(3, 4)
        pose_nerf = pose.clone()

        # rescale the predicted pose to nerf scales
        pose_nerf = fix_coord_supp(args, pose_nerf[None, ...].cpu(), world_setup_dict)

        # generate nerf image
        with torch.no_grad():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            """
            if args.tinyimg:
                rgb, _, _, _ = render(int(H//args.tinyscale), int(W//args.tinyscale), focal/args.tinyscale, chunk=args.chunk, c2w=pose_nerf[0,:3,:4].to(device), retraw=False, img_idx=img_idx, **render_kwargs_test)
                # convert rgb to B,C,H,W
                rgb = rgb[None,...].permute(0,3,1,2)
                # upsample rgb to hwf size
                rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
                # convert rgb back to H,W,C format
                rgb = rgb[0].permute(1,2,0)

            else:
            """
            rgb, _, _, _ = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf[0, :3, :4].to(device), retraw=False,
                                  img_idx=img_idx, **render_kwargs_test)
            rgbs.append(rgb.cpu().numpy())
            torch.set_default_tensor_type('torch.FloatTensor')
        rgb_list.append(rgb.cpu())
        # 保存图像
        rgb8_f = to8b(rgbs[-1])  # save coarse+fine img
        filename = os.path.join(save_folder_rgb, 'zadd_{}_{:05d}_{}.png'.format(scene, batch_idx, epoch))
        imageio.imwrite(filename, rgb8_f)
        # 保存位姿
        poses_filename = os.path.join(save_folder_poses, "zadd_{}_{:05d}_{}.txt".format(scene, batch_idx, epoch))
        poses4_filename = os.path.join(save_folder_poses4, "zadd_{}_{:05d}{}.txt".format(scene, batch_idx, epoch))
        # 将中心化的坐标系转回原来的坐标系
        pose = pose.cpu().numpy()
        # pose_avg_stats_file = "data" + '/pose_avg_stats/' + scene + ".txt"
        pose_loc, pose_t = inv_concered_SIASUN(pose, batch_idx, scene)
        np.savetxt(poses_filename, pose_loc)
        np.savetxt(poses4_filename, pose_t)
    rgbs = torch.stack(rgb_list).detach()
    return rgbs


def inv_concered_NEU(pose, batch_idx, scene):
    E_pose = np.eye(4)
    E_pose[:3] = pose
    # if (batch_idx == 4):
    # print(pose_t)
    pose_t = E_pose
    import re
    pattern = re.compile(r'NEU_scan(\d+)')  # 根据nerf文件的名字该
    match = pattern.match(scene)
    if match:
        number = int(match.group(1))
    else:
        print("error")
    element_1 = pose_t[0, 3] + number * 100
    element_2 = pose_t[1, 3] + number * 100
    # 将提取的元素组成一个新的矩阵
    pose_loc = np.array([[element_1, element_2]])
    return pose_loc, pose_t


def fix_coord_NEU(poses, scenes):
    ''' fix coord for 7 Scenes to align with llff style dataset '''
    ON, N, H, W = poses.shape
    poses = poses.view(ON * N, H, W)
    # get all poses (train+val)
    all_poses = poses[:, :3, :4].reshape(poses.shape[0], 12)
    # all_poses = np.concatenate([train_poses, val_poses])
    # Center the poses for ndc
    all_poses = all_poses.reshape(all_poses.shape[0], 3, 4)
    print("all_poses[0]")
    print(all_poses[0])
    scans = [0] * 5
    for i in range(all_poses.shape[0]):
        if (scenes[i] == "scan01"):
            scans[0] = scans[0] + 1
        elif (scenes[i] == "scan02"):
            scans[1] = scans[1] + 1
        elif (scenes[i] == "scan03"):
            scans[2] = scans[2] + 1
        elif (scenes[i] == "scan04"):
            scans[3] = scans[3] + 1
        elif (scenes[i] == "scan05"):
            scans[4] = scans[4] + 1
        else:
            print("ERROR!!!!")
    idx = [scans[0], sum(scans[:2]), sum(scans[:3]), sum(scans[:4]), sum(scans[:5])]

    bounds = np.array([0, 10])  # manual tuned
    all_poses = all_poses.cpu()
    all_poses = all_poses.numpy()
    # Return all poses to dataset loaders
    all_poses = all_poses.reshape(all_poses.shape[0], 12)
    return all_poses, bounds, idx


def inv_concered_SIASUN(pose, batch_idx, scene):
    E_pose = np.eye(4)
    E_pose[:3] = pose
    # if (batch_idx == 4):
    # print(pose_t)
    pose_t = E_pose
    import re
    pattern = re.compile(r'sia_scan(\d+)')
    match = pattern.match(scene)
    if match:
        number = int(match.group(1))
    else:
        print("error")
    element_1 = pose_t[0, 3] + number * 100
    element_2 = pose_t[1, 3] + number * 100
    # 将提取的元素组成一个新的矩阵
    pose_loc = np.array([[element_1, element_2]])
    return pose_loc, pose_t


def fix_coord_SIASUN(poses, scenes):
    ''' fix coord for 7 Scenes to align with llff style dataset '''
    ON, N, H, W = poses.shape
    poses = poses.view(ON * N, H, W)
    # get all poses (train+val)
    all_poses = poses[:, :3, :4].reshape(poses.shape[0], 12)
    # all_poses = np.concatenate([train_poses, val_poses])
    # Center the poses for ndc
    all_poses = all_poses.reshape(all_poses.shape[0], 3, 4)
    print("all_poses[0]")
    print(all_poses[0])
    scans = [0] * 5
    for i in range(all_poses.shape[0]):
        if (scenes[i] == "scan01"):
            scans[0] = scans[0] + 1
        elif (scenes[i] == "scan02"):
            scans[1] = scans[1] + 1
        elif (scenes[i] == "scan03"):
            scans[2] = scans[2] + 1
        elif (scenes[i] == "scan04"):
            scans[3] = scans[3] + 1
        elif (scenes[i] == "scan05"):
            scans[4] = scans[4] + 1
        else:
            print("ERROR!!!!")
    idx = [scans[0], sum(scans[:2]), sum(scans[:3]), sum(scans[:4]), sum(scans[:5])]

    bounds = np.array([0, 10])  # manual tuned
    all_poses = all_poses.cpu()
    all_poses = all_poses.numpy()
    # Return all poses to dataset loaders
    all_poses = all_poses.reshape(all_poses.shape[0], 12)
    return all_poses, bounds, idx


def load_matrices_from_folder(folder_path):
    matrices = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as file:
                matrix = np.loadtxt(file)
                matrices.append(matrix)
    return matrices


def generate_random_matrices_with_constraints(matrices, x):
    np.random.seed(42)

    matrices_array = np.array(matrices)

    # 获取每个位置的最小值和最大值
    min_values = matrices_array.min(axis=0)
    max_values = matrices_array.max(axis=0)

    # 生成随机矩阵
    random_matrices = np.random.uniform(low=min_values, high=max_values, size=(x, 3, 4, 4))

    # 设置最后一行
    last_row = np.array([0, 0, 0, 1])
    for matrix in random_matrices:
        matrix[:, 3, :] = last_row

    return random_matrices


if __name__ == '__main__':
    """
    dir_q="E:/shujuji/nerfCambridge4VPR/CambridgeNerf_train1_4/val/database/poses4"
    none=None
    pose,scene = read_pose(dir_q)

    q_id=[1, 2, 3, 6, 12, 15, 21, 23, 26, 29, 35, 36, 37, 40, 48, 51, 54, 55, 61, 72, 77, 83, 84, 85, 86, 87, 89, 90, 93, 98, 105, 106, 108, 109, 111, 120, 124, 131, 149,
 151, 153, 154, 157, 159, 161, 162, 166, 168, 170, 175, 177, 179, 194, 195, 196, 231, 235, 237, 238, 247, 248, 249, 267, 278, 295, 305, 307, 312]
    selected_scenes = [scene[idx] for idx in q_id]
    print(selected_scenes)

    imaged = find_image(none, none, dir_q)
    for idx in q_id:
        print(idx)
        img = imaged[idx]
        img = Image.open(img)

        print(img)

    #extra_row = torch.tensor([0, 0, 0, 1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    #extra_rows = extra_row.expand(2, 1, 4)
    #print(extra_rows.shape)
    model_name = "first"
    images_ref = torch .rand((58,5,3,224,224))
    poses_ref = torch.rand((58,5,4,4))
    poses_target = torch.rand((58,6,4,4), device='cuda:0')
    predict = NBP_Cam(model_name, images_ref, poses_ref, poses_target)
    expanded_predict=predict.unsqueeze(-1).unsqueeze(-1).expand(-1, 3, 4, 4)
    selected_poses = torch.gather(poses_target, 1, expanded_predict)
    print(selected_poses.shape) #(58,3,4,4)


    # 假设你的张量是data
    data = torch.tensor([[0.0472, 0.0442, 0.0470, 0.0465, 0.0271, 0.0430]], device='cuda:0')

    # 计算按照从小到大排列的索引
    sorted_indices = torch.argsort(data)

    # 反转顺序，得到按照从高到低排列的索引
    descending_indices = sorted_indices.flip(dims=[1])
    a=descending_indices[:, :3]
    # 输出结果
    print(descending_indices)

    scene = "abc"
    batch_idx = 10

    # 格式化字符串
    filename = '{}_{:05d}.png'.format(scene, batch_idx)

    print(filename)
"""
# 测试函数
poses = torch.rand(58, 4, 4)  # 姿态张量
x = 0.1  # 平移的范围
angle = 10.0  # 角度扰动的范围
n = 5  # 生成新姿态的数量

perturbed_poses = perturb_single_render_pose(poses, x, angle, n)
print(perturbed_poses.shape)  # 输出 torch.Size([58, 5, 4, 4])
