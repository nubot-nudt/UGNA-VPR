import argparse
import json
import os
import random

import numpy as np
import torch
from genericpath import exists


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Options")
        self.parser.add_argument('--phase', type=str, default='train_tea', help='phase',
                                 choices=['train_tea', 'test_tea', 'train_stu', 'test_stu'])
        self.parser.add_argument('--dataset', type=str, default='Cambridge', help='choose dataset.')
        self.parser.add_argument('--structDir', type=str, default='Cambridge', help='Path for structure.')
        self.parser.add_argument('--imgDir', type=str, default='Cambridge/CambridgeNerf_train1_4',
                                 help='Path for images.')
        self.parser.add_argument('--com', type=str, default='', help='comment')
        self.parser.add_argument('--height', type=int, default=224, help='number of sequence to use.')
        self.parser.add_argument('--width', type=int, default=224, help='number of sequence to use.')
        self.parser.add_argument('--net', type=str, default='mixvpr', help='network')
        self.parser.add_argument('--trainer', type=str, default='trainer', help='trainer')
        self.parser.add_argument('--loss', type=str, default='tri', help='triplet loss or bayesian triplet loss',
                                 choices=['tri', 'cont', 'quad'])
        self.parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
        self.parser.add_argument('--margin2', type=float, default=0.1, help='Margin2 for quadruplet loss. Default=0.1')
        self.parser.add_argument('--output_dim', type=int, default=0, help='Number of feature dimension. Default=512')
        self.parser.add_argument('--sigma_dim', type=int, default=0, help='Number of sigma dimension. Default=512')
        self.parser.add_argument('--batchSize', type=int, default=8,
                                 help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
        self.parser.add_argument('--cacheBatchSize', type=int, default=128, help='Batch size for caching and testing')
        self.parser.add_argument('--cacheRefreshRate', type=int, default=0,
                                 help='How often to refresh cache, in number of queries. 0 for off')
        self.parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
        self.parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
        self.parser.add_argument('--cGPU', type=int, default=2, help='core of GPU to use.')
        self.parser.add_argument('--optim', type=str, default='adam', help='optimizer to use', choices=['sgd', 'adam'])
        self.parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate.')
        self.parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
        self.parser.add_argument('--lrGamma', type=float, default=0.99, help='Multiply LR by Gamma for decaying.')
        self.parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
        self.parser.add_argument('--cuda', action='store_false', help='use cuda')
        self.parser.add_argument('--d', action='store_true', help='debug mode')
        self.parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
        self.parser.add_argument('--seed', type=int, default=1234, help='Random seed to use.')
        self.parser.add_argument('--logsPath', type=str, default='./logs', help='Path to save runs to.')
        self.parser.add_argument('--runsPath', type=str, default='not defined', help='Path to save runs to.')
        self.parser.add_argument('--resume', type=str, default='',
                                 help='Path to load checkpoint from, for resuming training or testing.')
        self.parser.add_argument('--evalEvery', type=int, default=1,
                                 help='Do a validation set run, and save, every N epochs.')
        self.parser.add_argument('--cacheRefreshEvery', type=int, default=1,
                                 help='refresh embedding cache, every N epochs.')
        self.parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping. 0 is off.')
        self.parser.add_argument('--split', type=str, default='test', help='Split to use', choices=['val', 'test'])
        self.parser.add_argument('--encoder_dim', type=int, default=512,
                                 help='Number of feature dimension. Default=512')
        self.parser.add_argument('--Datasetname', type=str, default='Cambridge', help='phase',
                                 choices=['Cambridge', 'NEU', 'SIASUN'])
        # NBP
        self.parser.add_argument("--model_name", "-M", type=str, default="Cam_mixvpr_NBP",
                                 help="model name of pretrained model", )

        # Nerf-h
        self.parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
        self.parser.add_argument("--device", type=int, default=-1, help='CUDA_VISIBLE_DEVICES')
        self.parser.add_argument("--multi_gpu", action='store_true', help='use multiple gpu on the server')
        self.parser.add_argument('--config', is_config_file=True, help='config file path')
        self.parser.add_argument("--expname", type=str, default='nerfh', help='experiment name')
        self.parser.add_argument("--basedir", type=str, default='logs', help='where to store ckpts and logs')
        self.parser.add_argument("--datadir", type=str, default='./data/llff/fern', help='input data directory')

        # 7Scenes
        self.parser.add_argument("--trainskip", type=int, default=1,
                                 help='will load 1/N images from train sets, useful for large datasets like 7 Scenes')
        self.parser.add_argument("--df", type=float, default=1., help='image downscale factor')
        self.parser.add_argument("--reduce_embedding", type=int, default=-1, help='fourier embedding mode: -1: paper default, \
                                                                                0: reduce by half, 1: remove embedding, 2: DNeRF embedding')
        self.parser.add_argument("--epochToMaxFreq", type=int, default=-1, help='DNeRF embedding mode: (based on Nerfie paper): \
                                                                                hyper-parameter for when α should reach the maximum number of frequencies m')
        self.parser.add_argument("--render_pose_only", action='store_true', help='render a spiral video for 7 Scene')
        self.parser.add_argument("--save_pose_avg_stats", action='store_true',
                                 help='save a pose avg stats to unify NeRF, posenet, direct-pn training')
        self.parser.add_argument("--load_pose_avg_stats", action='store_true',
                                 help='load precomputed pose avg stats to unify NeRF, posenet, nerf tracking training')
        self.parser.add_argument("--train_local_nerf", type=int, default=-1,
                                 help='train local NeRF with ith training sequence only, ie. Stairs can pick 0~3')
        self.parser.add_argument("--render_video_train", action='store_true',
                                 help='render train set NeRF and save as video, make sure render_test is True')
        self.parser.add_argument("--render_video_test", action='store_true',
                                 help='render val set NeRF and save as video,  make sure render_test is True')
        self.parser.add_argument("--frustum_overlap_th", type=float, help='frustsum overlap threshold')
        self.parser.add_argument("--no_DNeRF_viewdir", action='store_true', default=False,
                                 help='will not use DNeRF in viewdir encoding')
        self.parser.add_argument("--load_unique_view_stats", action='store_true', help='load unique views frame index')

        # NeRF training options
        self.parser.add_argument("--netdepth", type=int, default=8, help='layers in network')
        self.parser.add_argument("--netwidth", type=int, default=128, help='channels per layer')
        self.parser.add_argument("--netdepth_fine", type=int, default=8, help='layers in fine network')
        self.parser.add_argument("--netwidth_fine", type=int, default=128, help='channels per layer in fine network')
        self.parser.add_argument("--N_rand", type=int, default=1536,
                                 help='batch size (number of random rays per gradient step)')
        self.parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
        self.parser.add_argument("--lrate_decay", type=float, default=250,
                                 help='exponential learning rate decay (in 1000 steps)')
        self.parser.add_argument("--chunk", type=int, default=1024 * 32,
                                 help='number of rays processed in parallel, decrease if running out of memory')
        self.parser.add_argument("--netchunk", type=int, default=1024 * 64,
                                 help='number of pts sent through network in parallel, decrease if running out of memory')
        self.parser.add_argument("--no_batching", action='store_true',
                                 help='only take random rays from 1 image at a time')
        self.parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
        self.parser.add_argument("--ft_path", type=str, default=None,
                                 help='specific weights npy file to reload for coarse network')
        self.parser.add_argument("--no_grad_update", action='store_true', default=False,
                                 help='do not update nerf in training')

        # NeRF-Hist training options
        self.parser.add_argument("--NeRFH", action='store_true', default=True,
                                 help='my implementation for NeRFH, to enable NeRF-Hist training, please make sure to add --encode_hist, otherwise it is similar to NeRFW')
        self.parser.add_argument("--N_vocab", type=int, default=1000,
                                 help='''number of vocabulary (number of images) 
                                        in the dataset for nn.Embedding''')
        self.parser.add_argument("--fix_index", action='store_true', help='fix training frame index as 0')
        self.parser.add_argument("--encode_hist", default=True, action='store_true',
                                 help='encode histogram instead of frame index')
        self.parser.add_argument("--hist_bin", type=int, default=10, help='image histogram bin size')
        self.parser.add_argument("--in_channels_a", type=int, default=50,
                                 help='appearance embedding dimension, hist_bin*N_a when embedding histogram')
        self.parser.add_argument("--in_channels_t", type=int, default=20,
                                 help='transient embedding dimension, hist_bin*N_tau when embedding histogram')

        # NeRF rendering options
        self.parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
        self.parser.add_argument("--N_importance", type=int, default=64,
                                 help='number of additional fine samples per ray')
        self.parser.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
        self.parser.add_argument("--use_viewdirs", default=True, action='store_true',
                                 help='use full 5D input instead of 3D')
        self.parser.add_argument("--i_embed", type=int, default=0,
                                 help='set 0 for default positional encoding, -1 for none')
        self.parser.add_argument("--multires", type=int, default=10,
                                 help='log2 of max freq for positional encoding (3D location)')
        self.parser.add_argument("--multires_views", type=int, default=4,
                                 help='log2 of max freq for positional encoding (2D direction)')
        self.parser.add_argument("--raw_noise_std", type=float, default=0.,
                                 help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
        self.parser.add_argument("--render_only", action='store_true',
                                 help='do not optimize, reload weights and render out render_poses path')
        self.parser.add_argument("--render_test", action='store_true',
                                 help='render the test set instead of render_poses path')
        self.parser.add_argument("--render_factor", type=int, default=0,
                                 help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

        # legacy mesh options
        self.parser.add_argument("--mesh_only", action='store_true',
                                 help='do not optimize, reload weights and save mesh to a file')
        self.parser.add_argument("--mesh_grid_size", type=int, default=80,
                                 help='number of grid points to sample in each dimension for marching cubes')

        # training options
        self.parser.add_argument("--precrop_iters", type=int, default=0,
                                 help='number of steps to train on central crops')
        self.parser.add_argument("--precrop_frac", type=float, default=.5,
                                 help='fraction of img taken for central crops')
        self.parser.add_argument("--epochs", type=int, default=600, help='number of epochs to train')

        # dataset options
        self.parser.add_argument("--dataset_type", type=str, default='NE', help='options: llff / 7Scenes')
        self.parser.add_argument("--testskip", type=int, default=1,
                                 help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

        ## legacy blender flags
        self.parser.add_argument("--white_bkgd", action='store_true',
                                 help='set to render synthetic data on a white bkgd (always use for dvoxels)')
        # parser.add_argument("--half_res", action='store_true', help='load blender synthetic data at 400x400 instead of 800x800')

        ## llff flags
        self.parser.add_argument("--factor", type=int, default=8, help='downsample factor for LLFF images')
        self.parser.add_argument("--no_ndc", action='store_true',
                                 help='do not use normalized device coordinates (set for non-forward facing scenes)')
        self.parser.add_argument("--lindisp", action='store_true',
                                 help='sampling linearly in disparity rather than depth')
        self.parser.add_argument("--spherify", action='store_true', help='set for spherical 360 scenes')
        self.parser.add_argument("--llffhold", type=int, default=8,
                                 help='will take every 1/N images as LLFF test set, paper uses 8')
        self.parser.add_argument("--no_bd_factor", action='store_true', default=False, help='do not use bd factor')
        # d_max
        self.parser.add_argument("--d_max", type=int, default=0.5, help='d_max')
        # logging/saving options
        self.parser.add_argument("--i_print", type=int, default=1,
                                 help='frequency of console printout and metric loggin')
        self.parser.add_argument("--i_img", type=int, default=500, help='frequency of tensorboard image logging')
        self.parser.add_argument("--i_weights", type=int, default=200, help='frequency of weight ckpt saving')
        self.parser.add_argument("--i_testset", type=int, default=200, help='frequency of testset saving')
        self.parser.add_argument("--i_video", type=int, default=50000, help='frequency of render_poses video saving')

        self.parser.add_argument("--rvs_trans", type=float, default=7.5, help='jitter range for rvs on translation')
        self.parser.add_argument("--rvs_rotation", type=float, default=0.2,
                                 help='jitter range for rvs on rotation, this is in log_10 uniform range, log(15) = 1.2')
        """
        # nerf
        self.parser.add_argument('--config', is_config_file=True,
                            help='config file path')
        self.parser.add_argument("--expname", type=str, default='nerfh', help='experiment name')
        self.parser.add_argument("--basedir", type=str, default='logs', help='where to store ckpts and logs')
        self.parser.add_argument("--datadir", type=str, default='./data/llff/fern', help='input data directory')

        # training options
        self.parser.add_argument("--netdepth", type=int, default=8,
                            help='layers in network')
        self.parser.add_argument("--netwidth", type=int, default=256,
                            help='channels per layer')
        self.parser.add_argument("--netdepth_fine", type=int, default=8,
                            help='layers in fine network')
        self.parser.add_argument("--netwidth_fine", type=int, default=256,
                            help='channels per layer in fine network')
        self.parser.add_argument("--N_rand", type=int, default=1024,
                            help='batch size (number of random rays per gradient step)')
        self.parser.add_argument("--lrate", type=float, default=5e-4,
                            help='learning rate')
        self.parser.add_argument("--lrate_decay", type=int, default=250,
                            help='exponential learning rate decay (in 1000 steps)')
        self.parser.add_argument("--chunk", type=int, default=1024 * 32,
                            help='number of rays processed in parallel, decrease if running out of memory')
        self.parser.add_argument("--netchunk", type=int, default=1024 * 64,
                            help='number of pts sent through network in parallel, decrease if running out of memory')
        self.parser.add_argument("--no_batching", action='store_true',
                            help='only take random rays from 1 image at a time')
        self.parser.add_argument("--no_reload", action='store_true',
                            help='do not reload weights from saved ckpt')
        self.parser.add_argument("--ft_path", type=str, default=None,
                            help='specific weights npy file to reload for coarse network')

        # rendering options
        self.parser.add_argument("--N_samples", type=int, default=64,
                            help='number of coarse samples per ray')
        self.parser.add_argument("--N_importance", type=int, default=64,
                            help='number of additional fine samples per ray')
        self.parser.add_argument("--perturb", type=float, default=1.,
                            help='set to 0. for no jitter, 1. for jitter')
        self.parser.add_argument("--use_viewdirs", default=True,action='store_true',
                            help='use full 5D input instead of 3D')
        self.parser.add_argument("--i_embed", type=int, default=0,
                            help='set 0 for default positional encoding, -1 for none')
        self.parser.add_argument("--multires", type=int, default=10,
                            help='log2 of max freq for positional encoding (3D location)')
        self.parser.add_argument("--multires_views", type=int, default=4,
                            help='log2 of max freq for positional encoding (2D direction)')
        self.parser.add_argument("--raw_noise_std", type=float, default=0.,
                            help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

        self.parser.add_argument("--render_only", action='store_true',
                            help='do not optimize, reload weights and render out render_poses path')
        self.parser.add_argument("--render_test", action='store_true',
                            help='render the test set instead of render_poses path')
        self.parser.add_argument("--render_factor", type=int, default=0,
                            help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

        # training options
        self.parser.add_argument("--precrop_iters", type=int, default=500,
                            help='number of steps to train on central crops')
        self.parser.add_argument("--precrop_frac", type=float,
                            default=.5, help='fraction of img taken for central crops')

        # dataset options
        self.parser.add_argument("--dataset_type",  default='Cambridge', type=str, 
                            help='options: llff / blender / deepvoxels')
        self.parser.add_argument("--testskip", type=int, default=8,
                            help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

        ## deepvoxels flags
        self.parser.add_argument("--shape", type=str, default='greek',
                            help='options : armchair / cube / greek / vase')

        ## blender flags
        self.parser.add_argument("--white_bkgd", action='store_true',
                            help='set to render synthetic data on a white bkgd (always use for dvoxels)')
        self.parser.add_argument("--half_res", default=False,action='store_true',
                            help='load blender synthetic data at 400x400 instead of 800x800')

        ## llff flags
        self.parser.add_argument("--factor", type=int, default=8,
                            help='downsample factor for LLFF images')
        self.parser.add_argument("--no_ndc", action='store_true',
                            help='do not use normalized device coordinates (set for non-forward facing scenes)')
        self.parser.add_argument("--lindisp", action='store_true',
                            help='sampling linearly in disparity rather than depth')
        self.parser.add_argument("--spherify", action='store_true',
                            help='set for spherical 360 scenes')
        self.parser.add_argument("--llffhold", type=int, default=8,
                            help='will take every 1/N images as LLFF test set, paper uses 8')

        # logging/saving options
        self.parser.add_argument("--i_print", type=int, default=100,
                            help='frequency of console printout and metric loggin')
        self.parser.add_argument("--i_img", type=int, default=500,
                            help='frequency of tensorboard image logging')
        self.parser.add_argument("--i_weights", type=int, default=10000,
                            help='frequency of weight ckpt saving')
        self.parser.add_argument("--i_testset", type=int, default=50000,
                            help='frequency of testset saving')
        self.parser.add_argument("--i_video", type=int, default=50000,
                            help='frequency of render_poses video saving')

        self.parser.add_argument("--trainskip", type=int, default=1,
                            help='will load 1/N images from train sets, useful for large datasets like 7 Scenes')
        self.parser.add_argument("--df", type=float, default=1., help='image downscale factor')
        self.parser.add_argument("--load_pose_avg_stats", action='store_true',
                            help='load precomputed pose avg stats to unify NeRF, posenet, nerf tracking training')
        self.parser.add_argument("--NeRFH", action='store_true', default=True,
                            help='new implementation for NeRFH, please add --encode_hist')
        self.parser.add_argument("--epochs", type=int, default=2000, help='number of epochs to train')
        self.parser.add_argument("--encode_hist", default=False, action='store_true',
                            help='encode histogram instead of frame index')
        self.parser.add_argument("--tinyimg", action='store_true', default=False,
                            help='render nerf img in a tiny scale image, this is a temporal compromise for direct feature matching, must FIX later')
        self.parser.add_argument("--DFNet", action='store_true', default=False, help='use DFNet')
        self.parser.add_argument("--tripletloss", action='store_true',
                            help='use triplet loss at training featurenet, this is to prevent catastophic failing')
        self.parser.add_argument("--featurenet_batch_size", type=int, default=4,
                            help='featurenet training batch size, choose smaller batch size')
        self.parser.add_argument("--rvs_refresh_rate", type=int, default=20, help='re-synthesis new views per X epochs')
        self.parser.add_argument("--d_max", type=float, default=0.5, help='rvs bounds d_max')
        self.parser.add_argument("--eval", action='store_true', help='eval model')
        self.parser.add_argument("--render_pose_only", action='store_true', help='render a spiral video for 7 Scene')
        self.parser.add_argument("--pose_only", type=int, default=1, help='posenet type to train, \
                            1: train baseline posenet, 2: posenet+nerf manual optimize, \
                            3: VLocNet, 4: DGRNet')
        self.parser.add_argument("--fix_index", action='store_true', help='fix training frame index as 0')
        self.parser.add_argument("--hist_bin", type=int, default=10, help='image histogram bin size')
        self.parser.add_argument("--finetune_unlabel", action='store_true', help='finetune unlabeled sequence like MapNet')
        self.parser.add_argument("--save_pose_avg_stats", action='store_true',
                            help='save a pose avg stats to unify NeRF, posenet, direct-pn training')
        self.parser.add_argument("--batch_size", type=int, default=1, help='dataloader batch size, Attention: this is NOT the actual training batch size, \
                                please use --featurenet_batch_size for training')
        self.parser.add_argument("--no_grad_update", action='store_true', default=False,
                            help='do not update nerf in training')
        self.parser.add_argument("--reduce_embedding", type=int, default=-1, help='fourier embedding mode: -1: paper default, \
                                                                               0: reduce by half, 1: remove embedding, 2: DNeRF embedding')
        #rvs_trans Cambridge=7.5 0.2 NEU=1 0.1 
        """

    def parse(self):
        options = self.parser.parse_args()
        return options

    def update_opt_from_json(self, flag_file, options):
        if not exists(flag_file):
            raise ValueError('{} not exist'.format(flag_file))
        # restore_var = ['runsPath', 'net', 'seqLen', 'num_clusters', 'output_dim', 'structDir', 'imgDir', 'lrStep', 'lrGamma', 'weightDecay', 'momentum', 'num_clusters', 'optim', 'margin', 'seed', 'patience']
        do_not_update_list = ['resume', 'mode', 'phase', 'optim', 'split']
        if os.path.exists(flag_file):
            with open(flag_file, 'r') as f:
                # stored_flags = {'--' + k: str(v) for k, v in json.load(f).items() if k in restore_var}
                stored_flags = {'--' + k: str(v) for k, v in json.load(f).items() if k not in do_not_update_list}
                to_del = []
                for flag, val in stored_flags.items():
                    for act in self.parser._actions:
                        if act.dest == flag[2:]:  # stored parser match current parser
                            # store_true / store_false args don't accept arguments, filter these
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ''
                            else:
                                if val == str(act.default):
                                    to_del.append(flag)

                for flag, val in stored_flags.items():
                    missing = True
                    for act in self.parser._actions:
                        if flag[2:] == act.dest:
                            missing = False
                    if missing:
                        to_del.append(flag)

                for flag in to_del:
                    del stored_flags[flag]

                train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
                print('restored flags:', train_flags)
                options = self.parser.parse_args(train_flags, namespace=options)
        return options


class FixRandom:
    def __init__(self, seed) -> None:
        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    def seed_worker(self):
        worker_seed = self.seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)
