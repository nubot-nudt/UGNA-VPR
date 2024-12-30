# %%
import importlib
import os
import pickle
import shutil
from os.path import dirname, exists, join
import h5py
import faiss
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import json
import torch.optim as optim
from torchsummary import summary
from PIL import Image

os.sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import time
import random
from options import FixRandom
from util import *
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from script.feature.options_nerf import config_parser
from script.models.nerfw import create_nerf
from color import rgb_to_yuv
from utils.util import get_image_to_tensor_balanced, coordinate_transformation
import torch.nn.functional as F
from nerf_init import create_nerf_init


def input_transform(opt=None):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        CustomTo01Transform(),
        transforms.Resize((opt.height, opt.width), interpolation=InterpolationMode.BILINEAR),
    ])


class CustomTo01Transform:
    def __call__(self, images):
        return (images + 1.0) * 0.5


class CKD_loss(nn.Module):
    def __init__(self, margin) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embs_a, embs_p, embs_n, mu_tea_a, mu_tea_p, mu_tea_n):  # (1, D)
        SaTp = torch.norm(embs_a - mu_tea_p, p=2).pow(2)
        SpTa = torch.norm(embs_p - mu_tea_a, p=2).pow(2)

        SaTn = torch.norm(embs_a - mu_tea_n, p=2).pow(2)
        SnTa = torch.norm(embs_n - mu_tea_a, p=2).pow(2)

        SaTa = torch.norm(embs_a - mu_tea_a, p=2).pow(2)
        SpTp = torch.norm(embs_p - mu_tea_p, p=2).pow(2)
        SnTn = torch.norm(embs_n - mu_tea_n, p=2).pow(2)

        dis_D = SaTp + SpTa + SaTa + SpTp + SnTn
        # dis_D=SaTp+SpTa
        loss = 0.5 * (torch.clamp(self.margin + dis_D, min=0).pow(2))

        return loss


class Trainer:
    def __init__(self, options) -> None:

        self.opt = options
        self.input_transform = input_transform(self.opt)
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.model_name = self.opt.model_name
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # r variables
        self.step = 0
        self.epoch = 0
        self.current_lr = 0
        self.best_recalls = [0, 0, 0]

        # seed
        fix_random = FixRandom(self.opt.seed)
        self.seed_worker = fix_random.seed_worker()
        self.time_stamp = datetime.now().strftime('%m%d_%H%M%S')

        # nerfh
        self.args_nerf = self.opt

        bds_dict = {
            'near': 0,
            'far': 10,
        }
        if self.args_nerf.Datasetname == "Cambridge":
            # load NeRF
            _, render_kwargs_test_GreatCourt, start_GreatCourt, _, _ = create_nerf(self.args_nerf, scenes="GreatCourt")
            # _, render_kwargs_test_GreatCourt, start_GreatCourt, _, _  = create_nerf_init(self.args_nerf, scenes="GreatCourt")
            global_step_GreatCourt = start_GreatCourt
            # render_kwargs_train.update(bds_dict)
            render_kwargs_test_GreatCourt.update(bds_dict)
            if self.args_nerf.reduce_embedding == 2:
                render_kwargs_test_GreatCourt['i_epoch'] = start_GreatCourt
            self.render_kwargs_test_GreatCourt = render_kwargs_test_GreatCourt

            _, render_kwargs_test_KingsCollege, start_KingsCollege, _, _ = create_nerf(self.args_nerf,
                                                                                       scenes="KingsCollege")
            # _, render_kwargs_test_KingsCollege, start_KingsCollege, _, _ = create_nerf_init(self.args_nerf, scenes="KingsCollege")
            global_step_KingsCollege = start_KingsCollege
            # render_kwargs_train.update(bds_dict)
            render_kwargs_test_KingsCollege.update(bds_dict)
            if self.args_nerf.reduce_embedding == 2:
                render_kwargs_test_KingsCollege['i_epoch'] = start_KingsCollege
            self.render_kwargs_test_KingsCollege = render_kwargs_test_KingsCollege

            _, render_kwargs_test_OldHospital, start_OldHospital, _, _ = create_nerf(self.args_nerf,
                                                                                     scenes="OldHospital")
            # _, render_kwargs_test_OldHospital, start_OldHospital, _, _ = create_nerf_init(self.args_nerf, scenes="OldHospital")
            global_step_OldHospital = start_OldHospital
            # render_kwargs_train.update(bds_dict)
            render_kwargs_test_OldHospital.update(bds_dict)
            if self.args_nerf.reduce_embedding == 2:
                render_kwargs_test_OldHospital['i_epoch'] = start_OldHospital
            self.render_kwargs_test_OldHospital = render_kwargs_test_OldHospital

            _, render_kwargs_test_ShopFacade, start_ShopFacade, _, _ = create_nerf(self.args_nerf, scenes="ShopFacade")
            # _, render_kwargs_test_ShopFacade, start_ShopFacade, _, _ = create_nerf_init(self.args_nerf, scenes="ShopFacade")
            global_step_ShopFacade = start_ShopFacade
            # render_kwargs_train.update(bds_dict)
            render_kwargs_test_ShopFacade.update(bds_dict)
            if self.args_nerf.reduce_embedding == 2:
                render_kwargs_test_ShopFacade['i_epoch'] = start_ShopFacade
            self.render_kwargs_test_ShopFacade = render_kwargs_test_ShopFacade

            _, render_kwargs_test_StMarysChurch, start_StMarysChurch, _, _ = create_nerf(self.args_nerf,
                                                                                         scenes="StMarysChurch")
            # _, render_kwargs_test_StMarysChurch, start_StMarysChurch, _, _ = create_nerf_init(self.args_nerf, scenes="StMarysChurch")
            global_step_StMarysChurch = start_StMarysChurch
            # render_kwargs_train.update(bds_dict)
            render_kwargs_test_StMarysChurch.update(bds_dict)

            if self.args_nerf.reduce_embedding == 2:
                render_kwargs_test_StMarysChurch['i_epoch'] = start_StMarysChurch
            self.render_kwargs_test_StMarysChurch = render_kwargs_test_StMarysChurch

        if self.args_nerf.Datasetname == "NEU":
            self.NEU_folders = ["NEU_scan01", "NEU_scan02", "NEU_scan03", "NEU_scan04", "NEU_scan05"]
            render_kwargs_test = [None] * 5
            start = [None] * 5
            global_step = [None] * 5
            self.render_kwargs_test = [None] * 5
            for i in range(5):
                _, render_kwargs_test[i], start[i], _, _ = create_nerf(self.args_nerf, scenes=self.NEU_folders[i])
                global_step[i] = start[i]
                # render_kwargs_train.update(bds_dict)
                render_kwargs_test[i].update(bds_dict)
                if self.args_nerf.reduce_embedding == 2:
                    render_kwargs_test[i]['i_epoch'] = start[i]
                self.render_kwargs_test[i] = render_kwargs_test[i]
        if self.args_nerf.Datasetname == "SIASUN":
            self.SIASUN_folders = ["sia_scan01", "sia_scan02", "sia_scan03", "sia_scan04", "sia_scan05"]
            render_kwargs_test = [None] * 5
            start = [None] * 5
            global_step = [None] * 5
            self.render_kwargs_test = [None] * 5
            for i in range(5):
                _, render_kwargs_test[i], start[i], _, _ = create_nerf(self.args_nerf, scenes=self.SIASUN_folders[i])
                global_step[i] = start[i]
                # render_kwargs_train.update(bds_dict)
                render_kwargs_test[i].update(bds_dict)
                if self.args_nerf.reduce_embedding == 2:
                    render_kwargs_test[i]['i_epoch'] = start[i]
                self.render_kwargs_test[i] = render_kwargs_test[i]

        # set device
        if self.opt.phase == 'train_tea':
            self.opt.cGPU = schedule_device()
        if self.opt.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run with --nocuda :(")
        torch.cuda.set_device(self.opt.cGPU)
        self.device = torch.device("cuda")
        print('{}:{}{}'.format('device', self.device, torch.cuda.current_device()))

        # CKD_loss
        self.CKD_loss = CKD_loss(margin=torch.tensor(self.opt.margin, device=self.device))
        # make model
        if self.opt.phase == 'train_tea':
            self.model, self.optimizer, self.scheduler, self.criterion = self.make_model()
        elif self.opt.phase == 'train_stu':
            self.teacher_net, self.student_net, self.optimizer, self.scheduler, self.criterion = self.make_model()
            self.model = self.teacher_net
        elif self.opt.phase in ['test_tea', 'test_stu']:
            self.model = self.make_model()
        else:
            raise Exception('Undefined phase :(')

        # make folders
        self.make_folders()
        # make dataset
        self.make_dataset()
        # online logs
        if self.opt.phase in ['train_tea', 'train_stu']:
            wandb.init(project="TSCM", config=vars(self.opt),
                       name=f"{self.opt.loss}_{self.opt.phase}_{self.time_stamp}")

    def make_folders(self):
        ''' create folders to store tensorboard files and a copy of networks files
        '''
        if self.opt.phase in ['train_tea', 'train_stu']:
            self.opt.runsPath = join(self.opt.logsPath, f"{self.opt.loss}_{self.opt.phase}_{self.time_stamp}")
            if not os.path.exists(join(self.opt.runsPath, 'models')):
                os.makedirs(join(self.opt.runsPath, 'models'))

            if not os.path.exists(join(self.opt.runsPath, 'transformed')):
                os.makedirs(join(self.opt.runsPath, 'transformed'))

            for file in [__file__, 'datasets/{}.py'.format(self.opt.dataset), 'networks/{}.py'.format(self.opt.net)]:
                shutil.copyfile(file, os.path.join(self.opt.runsPath, 'models', file.split('/')[-1]))

            with open(join(self.opt.runsPath, 'flags.json'), 'w') as f:
                f.write(json.dumps({k: v for k, v in vars(self.opt).items()}, indent=''))

    def make_dataset(self):
        ''' make dataset
        '''
        if self.opt.phase in ['train_tea', 'train_stu']:
            assert os.path.exists(f'datasets/{self.opt.dataset}.py'), 'Cannot find ' + f'{self.opt.dataset}.py :('
            self.dataset = importlib.import_module('datasets.' + self.opt.dataset)
        elif self.opt.phase in ['test_tea', 'test_stu']:
            self.dataset = importlib.import_module('tmp.models.{}'.format(self.opt.dataset))

        # for emb cache
        self.whole_train_set = self.dataset.get_whole_training_set(self.opt)
        self.whole_training_data_loader = DataLoader(dataset=self.whole_train_set, num_workers=self.opt.threads,
                                                     batch_size=self.opt.cacheBatchSize, shuffle=False,
                                                     pin_memory=self.opt.cuda, worker_init_fn=self.seed_worker)
        self.whole_val_set = self.dataset.get_whole_val_set(self.opt)
        self.whole_val_data_loader = DataLoader(dataset=self.whole_val_set, num_workers=self.opt.threads,
                                                batch_size=self.opt.cacheBatchSize, shuffle=False,
                                                pin_memory=self.opt.cuda, worker_init_fn=self.seed_worker)
        self.whole_test_set = self.dataset.get_whole_test_set(self.opt)
        self.whole_test_data_loader = DataLoader(dataset=self.whole_test_set, num_workers=self.opt.threads,
                                                 batch_size=self.opt.cacheBatchSize, shuffle=False,
                                                 pin_memory=self.opt.cuda, worker_init_fn=self.seed_worker)

        self.train_set = self.dataset.get_training_query_set(self.opt, self.opt.margin)
        self.training_data_loader = DataLoader(dataset=self.train_set, num_workers=8, batch_size=self.opt.batchSize,
                                               shuffle=True, collate_fn=self.dataset.collate_fn,
                                               worker_init_fn=self.seed_worker)
        print('{}:{}, {}:{}, {}:{}, {}:{}, {}:{}'.format('dataset', self.opt.dataset, 'database',
                                                         self.whole_train_set.dbStruct.numDb, 'train_set',
                                                         self.whole_train_set.dbStruct.numQ, 'val_set',
                                                         self.whole_val_set.dbStruct.numQ, 'test_set',
                                                         self.whole_test_set.dbStruct.numQ))
        print('{}:{}, {}:{}'.format('cache_bs', self.opt.cacheBatchSize, 'tuple_bs', self.opt.batchSize))

    def make_model(self):
        '''build model
        '''
        if self.opt.phase == 'train_tea':
            # build teacher net
            assert os.path.exists(f'networks/{self.opt.net}.py'), 'Cannot find ' + f'{self.opt.net}.py :('
            network = importlib.import_module('networks.' + self.opt.net)
            model = network.deliver_model(self.opt, 'tea')
            model = model.to(self.device)
            outputs = model(torch.rand((2, 3, self.opt.height, self.opt.width), device=self.device))
            self.opt.output_dim = \
                model(torch.rand((2, 3, self.opt.height, self.opt.width), device=self.device))[0].shape[-1]
            self.opt.sigma_dim = \
                model(torch.rand((2, 3, self.opt.height, self.opt.width), device=self.device))[1].shape[
                    -1]  # place holder
        elif self.opt.phase == 'train_stu':  # load teacher net
            assert self.opt.resume != '', 'You need to define the teacher/resume path :('
            if exists('tmp'):
                shutil.rmtree('tmp')
            os.mkdir('tmp')
            shutil.copytree(join(dirname(self.opt.resume), 'models'), join('tmp', 'models'))
            network = importlib.import_module(f'tmp.models.{self.opt.net}')
            model_tea = network.deliver_model(self.opt, 'tea').to(self.device)
            checkpoint = torch.load(self.opt.resume)
            model_tea.load_state_dict(checkpoint['state_dict'])
            # build student net
            assert os.path.exists(f'networks/{self.opt.net}.py'), 'Cannot find ' + f'{self.opt.net}.py :('
            network = importlib.import_module('networks.' + self.opt.net)
            model = network.deliver_model(self.opt, 'stu').to(self.device)
            # checkpointS=torch.load('logs/tri_train_stu_0804_180109/ckpt_e_1.pth.tar')
            # model.load_state_dict(checkpointS['state_dict'])
            self.opt.output_dim = \
                model(torch.rand((2, 3, self.opt.height, self.opt.width), device=self.device))[0].shape[-1]
            self.opt.sigma_dim = \
                model(torch.rand((2, 3, self.opt.height, self.opt.width), device=self.device))[1].shape[-1]
        elif self.opt.phase in ['test_tea', 'test_stu']:
            # load teacher or student net
            assert self.opt.resume != '', 'You need to define a teacher/resume path :('
            if exists('tmp'):
                shutil.rmtree('tmp')
            os.mkdir('tmp')
            shutil.copytree(join(dirname(self.opt.resume), 'models'), join('tmp', 'models'))
            network = importlib.import_module('tmp.models.{}'.format(self.opt.net))
            model = network.deliver_model(self.opt, self.opt.phase[-3:]).to(self.device)
            checkpoint = torch.load(self.opt.resume)
            model.load_state_dict(checkpoint['state_dict'])

        print('{}:{}, {}:{}, {}:{}'.format(model.id, self.opt.net, 'loss', self.opt.loss, 'mu_dim', self.opt.output_dim,
                                           'sigma_dim', self.opt.sigma_dim if self.opt.phase[-3:] == 'stu' else '-'))

        if self.opt.phase in ['train_tea', 'train_stu']:
            # optimizer
            if self.opt.optim == 'adam':
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), self.opt.lr,
                                       weight_decay=self.opt.weightDecay)
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.opt.lrGamma, last_epoch=-1, verbose=False)
            elif self.opt.optim == 'sgd':
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.opt.lr,
                                      momentum=self.opt.momentum, weight_decay=self.opt.weightDecay)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.lrStep, gamma=self.opt.lrGamma)
            else:
                raise NameError('Undefined optimizer :(')

            criterion = nn.TripletMarginLoss(margin=self.opt.margin, p=2, reduction='sum').to(self.device)

        if self.opt.nGPU > 1:
            model = nn.DataParallel(model)

        if self.opt.phase == 'train_tea':
            return model, optimizer, scheduler, criterion
        elif self.opt.phase == 'train_stu':
            return model_tea, model, optimizer, scheduler, criterion
        elif self.opt.phase in ['test_tea', 'test_stu']:
            return model
        else:
            raise NameError('Undefined phase :(')

    def build_embedding_cache(self):
        '''build embedding cache, such that we can find the corresponding (p) and (n) with respect to (a) in embedding space
        '''
        self.train_set.cache = os.path.join(self.opt.runsPath, self.train_set.whichSet + '_feat_cache.hdf5')
        with h5py.File(self.train_set.cache, mode='w') as h5:
            h5feat = h5.create_dataset("features", [len(self.whole_train_set), self.opt.output_dim], dtype=np.float32)
            with torch.no_grad():
                for iteration, (input, indices) in enumerate(tqdm(self.whole_training_data_loader), 1):
                    input = input.to(self.device)  # torch.Size([32, 3, 154, 154]) ([32, 5, 3, 200, 200])
                    emb, _ = self.model(input)
                    h5feat[indices.detach().numpy(), :] = emb.detach().cpu().numpy()
                    del input, emb

    def build_embedding_cache_stu(self):
        '''build embedding cache, such that we can find the corresponding (p) and (n) with respect to (a) in embedding space
        '''
        self.train_set.cache = os.path.join(self.opt.runsPath, self.train_set.whichSet + '_feat_cache.hdf5')
        with h5py.File(self.train_set.cache, mode='w') as h5:
            h5feat = h5.create_dataset("features", [len(self.whole_train_set), self.opt.output_dim], dtype=np.float32)
            with torch.no_grad():
                for iteration, (input, indices) in enumerate(tqdm(self.whole_training_data_loader), 1):
                    input = input.to(self.device)  # torch.Size([32, 3, 154, 154]) ([32, 5, 3, 200, 200])
                    emb, _ = self.student_net(input)
                    h5feat[indices.detach().numpy(), :] = emb.detach().cpu().numpy()
                    del input, emb

    def process_batch(self, batch_inputs):
        '''
        process a batch of input
        '''

        anchor, positives, negatives, neg_counts, indices = batch_inputs

        # in case we get an empty batch
        if anchor is None:
            return None, None

        # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor, where N = batchSize * (nQuery + nPos + n_neg)
        B = anchor.shape[0]  # ([8, 1, 3, 200, 200])
        n_neg = torch.sum(neg_counts)  # tensor(80) = torch.sum(torch.Size([8]))

        input = torch.cat([anchor, positives, negatives])  # ([B, C, H, 200])

        input = input.to(self.device)  # ([96, 1, C, H, W])
        embs, vars = self.model(input)  # ([96, D])

        tuple_loss = 0
        # Standard triplet loss (via PyTorch library)
        if self.opt.loss == 'tri':
            embs_a, embs_p, embs_n = torch.split(embs, [B, B, n_neg])
            for i, neg_count in enumerate(neg_counts):
                for n in range(neg_count):
                    negIx = (torch.sum(neg_counts[:i]) + n).item()
                    tuple_loss += self.criterion(embs_a[i:i + 1], embs_p[i:i + 1], embs_n[negIx:negIx + 1])
            tuple_loss /= n_neg.float().to(self.device)

        del input, embs, embs_a, embs_p, embs_n
        del anchor, positives, negatives

        return tuple_loss, n_neg

    def process_batch_stu(self, batch_inputs):
        '''
        process a batch of input
        '''
        anchor, positives, negatives, neg_counts, indices = batch_inputs

        # in case we get an empty batch
        if anchor is None:
            return None, None

        # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor, where N = batchSize * (nQuery + nPos + n_neg)
        B = anchor.shape[0]  # ([8, 1, 3, 200, 200])
        n_neg = torch.sum(neg_counts)  # tensor(80) = torch.sum(torch.Size([8]))

        input = torch.cat([anchor, positives, negatives])  # ([B, C, H, 200])

        input = input.to(self.device)  # ([96, 1, C, H, W])
        embs, vars = self.student_net(input)  # ([96, D])

        anchor = anchor.to(self.device)
        with torch.no_grad():
            mu_tea, _ = self.teacher_net(input)  # ([B, D])
        # mu_stu, log_sigma_sq = self.student_net(anchor)  # ([B, D]), ([B, D])

        tuple_loss = 0
        CKDloss = 0

        # Standard triplet loss (via PyTorch library)
        if self.opt.loss == 'tri':
            embs_a, embs_p, embs_n = torch.split(embs, [B, B, n_neg])
            vars_a, vars_p, vars_n = torch.split(vars, [B, B, n_neg])
            mu_tea_a, mu_tea_p, mu_tea_n = torch.split(mu_tea, [B, B, n_neg])
            for i, neg_count in enumerate(neg_counts):
                for n in range(neg_count):
                    negIx = (torch.sum(neg_counts[:i]) + n).item()
                    tuple_loss += self.criterion(embs_a[i:i + 1], embs_p[i:i + 1], embs_n[negIx:negIx + 1])
                    CKDloss += self.CKD_loss(embs_a[i:i + 1], embs_p[i:i + 1], embs_n[negIx:negIx + 1],
                                             mu_tea_a[i:i + 1], mu_tea_p[i:i + 1], mu_tea_n[negIx:negIx + 1])

            tuple_loss /= n_neg.float().to(self.device)
            CKDloss /= n_neg.float().to(self.device)
        del input, embs, embs_a, embs_p, embs_n
        del anchor, positives, negatives
        return tuple_loss + CKDloss, n_neg

    def train(self):
        not_improved = 0
        for epoch in range(self.opt.nEpochs):
            # make dataset
            self.make_dataset()
            self.epoch = epoch
            self.current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            # build embedding cache
            if self.epoch % self.opt.cacheRefreshEvery == 0:
                self.model.eval()
                self.build_embedding_cache()
                self.model.train()

            # train
            tuple_loss_sum = 0
            for _, batch_inputs in enumerate(tqdm(self.training_data_loader)):
                self.step += 1

                self.optimizer.zero_grad()
                tuple_loss, n_neg = self.process_batch(batch_inputs)
                if tuple_loss is None:
                    continue
                tuple_loss.backward()
                self.optimizer.step()
                tuple_loss_sum += tuple_loss.item()

                if self.step % 10 == 0:
                    wandb.log({'train_tuple_loss': tuple_loss.item()}, step=self.step)
                    wandb.log({'train_batch_num_neg': n_neg}, step=self.step)

            n_batches = len(self.training_data_loader)
            wandb.log({'train_avg_tuple_loss': tuple_loss_sum / n_batches}, step=self.step)
            torch.cuda.empty_cache()
            self.scheduler.step()

            # val every x epochs
            if (self.epoch % self.opt.evalEvery) == 0:
                recalls = self.val(self.model, self.epoch)
                if recalls[0] > self.best_recalls[0]:
                    self.best_recalls = recalls
                    not_improved = 0
                else:
                    if recalls[0] == self.best_recalls[0]:
                        self.save_model(self.model, is_best=False, save_every_epoch=True)
                    not_improved += self.opt.evalEvery
                # light log
                vars_to_log = [
                    'e={:>2d},'.format(self.epoch),
                    'lr={:>.8f},'.format(self.current_lr),
                    'tl={:>.4f},'.format(tuple_loss_sum / n_batches),
                    'r@1/5/10={:.2f}/{:.2f}/{:.2f}'.format(recalls[0], recalls[1], recalls[2]),
                    '\n' if not_improved else ' *\n',
                ]
                light_log(self.opt.runsPath, vars_to_log)
            else:
                recalls = None
            self.save_model(self.model, is_best=not not_improved)

            # stop when not improving for a period
            if self.opt.phase == 'train_tea':
                if self.opt.patience > 0 and not_improved > self.opt.patience:
                    print('terminated because performance has not improve for', self.opt.patience, 'epochs')
                    break

        self.save_model(self.model, is_best=False)
        print('best r@1/5/10={:.2f}/{:.2f}/{:.2f}'.format(self.best_recalls[0], self.best_recalls[1],
                                                          self.best_recalls[2]))

        return self.best_recalls

    def train_student(self):
        not_improved = 0
        for epoch in range(self.opt.nEpochs):
            self.epoch = epoch
            self.current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            # build embedding cache
            if self.epoch % self.opt.cacheRefreshEvery == 0:
                self.student_net.eval()
                self.build_embedding_cache()
                self.student_net.train()
                # train
                tuple_loss_sum = 0
                for _, batch_inputs in enumerate(tqdm(self.training_data_loader)):
                    self.step += 1

                    self.optimizer.zero_grad()
                    tuple_loss, n_neg = self.process_batch_stu(batch_inputs)
                    if tuple_loss is None:
                        continue
                    tuple_loss.backward()
                    self.optimizer.step()
                    tuple_loss_sum += tuple_loss.item()
                    loss_sum = tuple_loss_sum
                    if self.step % 10 == 0:
                        wandb.log({'train_tuple_loss': tuple_loss.item()}, step=self.step)
                        wandb.log({'train_batch_num_neg': n_neg}, step=self.step)

                n_batches = len(self.training_data_loader)
                wandb.log({'train_avg_tuple_loss': tuple_loss_sum / n_batches}, step=self.step)
                wandb.log({'student/epoch_loss': loss_sum / n_batches}, step=self.step)
                torch.cuda.empty_cache()
                self.scheduler.step()

            # val
            if (self.epoch % self.opt.evalEvery) == 0:
                recalls = self.val(self.student_net)
                if recalls[0] > self.best_recalls[0]:
                    self.best_recalls = recalls
                    not_improved = 0
                else:
                    not_improved += self.opt.evalEvery

                light_log(self.opt.runsPath, [
                    f'e={self.epoch:>2d},',
                    f'lr={self.current_lr:>.8f},',
                    f'tl={loss_sum / n_batches:>.4f},',
                    f'r@1/5/10={recalls[0]:.2f}/{recalls[1]:.2f}/{recalls[2]:.2f}',
                    '\n' if not_improved else ' *\n',
                ])
            else:
                recalls = None

            self.save_model(self.student_net, is_best=False, save_every_epoch=True)
            if self.opt.patience > 0 and not_improved > self.opt.patience:
                print('terminated because performance has not improve for', self.opt.patience, 'epochs')
                break

        print('best r@1/5/10={:.2f}/{:.2f}/{:.2f}'.format(self.best_recalls[0], self.best_recalls[1],
                                                          self.best_recalls[2]))
        return self.best_recalls

    def val(self, model, epoch):
        mode = "val"
        recalls, _ = self.get_recall(model, mode, epoch)

        for i, n in enumerate([1, 5, 10]):
            wandb.log({'{}/{}_r@{}'.format(model.id, self.opt.split, n): recalls[i]}, step=self.step)
            # self.writer.add_scalar('{}/{}_r@{}'.format(model.id, self.opt.split, n), recalls[i], self.epoch)

        return recalls

    def test(self):
        mode = "test"
        epoch = None
        recalls, _ = self.get_recall(self.model, mode, epoch, save_embs=True)
        print('best r@1/5/10={:.2f}/{:.2f}/{:.2f}'.format(recalls[0], recalls[1], recalls[2]))
        # summary(self.model, input_size=(3, 224, 224))

        return recalls

    def save_model(self, model, is_best=False, save_every_epoch=False):
        if is_best:
            torch.save({
                'epoch': self.epoch,
                'step': self.step,
                'state_dict': model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }, os.path.join(self.opt.runsPath, 'ckpt_best.pth.tar'))

        if save_every_epoch:
            torch.save({
                'epoch': self.epoch,
                'step': self.step,
                'state_dict': model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }, os.path.join(self.opt.runsPath, 'ckpt_e_{}.pth.tar'.format(self.epoch)))

    def get_recall(self, model, mode, epoch, save_embs=False):
        model.eval()

        if self.opt.split == 'val':
            eval_dataloader = self.whole_val_data_loader
            eval_set = self.whole_val_set
            eval_dataloader_train = self.whole_training_data_loader
            eval_set_train = self.whole_train_set
        elif self.opt.split == 'test':
            eval_dataloader = self.whole_test_data_loader
            eval_set = self.whole_test_set
        # print(f"{self.opt.split} len:{len(eval_set)}")
        # print(len(eval_set))
        # val
        whole_mu = torch.zeros((len(eval_set), self.opt.output_dim), device=self.device)  # (N, D)
        whole_var = torch.zeros((len(eval_set), self.opt.sigma_dim), device=self.device)  # (N, D)
        gt = eval_set.get_positives()  # (N_q, n_pos)
        if self.opt.split == 'val':
            # train
            whole_mu_train = torch.zeros((len(eval_set_train), self.opt.output_dim), device=self.device)  # (N, D)
            whole_var_train = torch.zeros((len(eval_set_train), self.opt.sigma_dim), device=self.device)  # (N, D)
            gt_train = eval_set_train.get_positives()  # (N_q, n_pos)
        # print(len(gt))
        start_time = time.time()
        with torch.no_grad():
            for iteration, (input, indices) in enumerate(tqdm(eval_dataloader), 1):
                input = input.to(self.device)
                # print(iteration)
                # print(input.shape)
                # print(indices)
                mu, _ = model(input)  # (B, D)
                # summary(self.model, input_size=input.shape[1:])
                # print(input.shape)
                # var = torch.exp(var)
                whole_mu[indices, :] = mu
                # whole_var[indices, :] = var
                del input, mu
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed Time:", elapsed_time)
        if self.opt.split == 'val':
            with torch.no_grad():
                for iteration, (input, indices) in enumerate(tqdm(eval_dataloader_train), 1):
                    input = input.to(self.device)
                    # print(iteration)
                    # print(input.shape)
                    # print(indices)
                    mu, _ = model(input)  # (B, D)
                    # summary(self.model, input_size=input.shape[1:])
                    # print(input.shape)
                    # var = torch.exp(var)
                    whole_mu_train[indices, :] = mu
                    # whole_var[indices, :] = var
                    del input, mu

        n_values = [1, 5, 10]

        whole_mu = whole_mu.cpu().numpy()
        mu_q = whole_mu[eval_set.dbStruct.numDb:].astype('float32')
        mu_db = whole_mu[:eval_set.dbStruct.numDb].astype('float32')
        faiss_index = faiss.IndexFlatL2(mu_q.shape[1])
        faiss_index.add(mu_db)
        dists, preds = faiss_index.search(mu_q, max(n_values))  # the results is sorted
        # cull queries without any ground truth positives in the database
        val_inds = [True if len(gt[ind]) != 0 else False for ind in range(len(gt))]
        val_inds = np.array(val_inds)
        mu_q = mu_q[val_inds]
        preds = preds[val_inds]
        dists = dists[val_inds]
        gt = gt[val_inds]
        recall_at_k, q_id = cal_recall(preds, gt, n_values)
        if self.opt.split == 'val':
            # train result for augmentation
            whole_mu_train = whole_mu_train.cpu().numpy()
            mu_q_train = whole_mu_train[eval_set_train.dbStruct.numDb:].astype('float32')
            mu_db_train = whole_mu_train[:eval_set_train.dbStruct.numDb].astype('float32')
            faiss_index_train = faiss.IndexFlatL2(mu_q_train.shape[1])
            faiss_index_train.add(mu_db_train)
            dists_train, preds_train = faiss_index_train.search(mu_q_train, max(n_values))  # the results is sorted
            # cull queries without any ground truth positives in the database
            val_inds_train = [True if len(gt_train[ind]) != 0 else False for ind in range(len(gt_train))]
            val_inds_train = np.array(val_inds_train)
            mu_q_train = mu_q_train[val_inds_train]
            preds_train = preds_train[val_inds_train]  # (n_q, 10)
            dists_train = dists_train[val_inds_train]  # (n_q, 10) 从近到远
            gt_train = gt_train[val_inds_train]
            recall_at_k_train, q_id_train = cal_recall(preds_train, gt_train, n_values)

        if mode == "val" and self.opt.Datasetname == "Cambridge" and len(q_id_train) > 0 and epoch > 5:
            device = torch.device("cuda")
            dir_q = os.path.join(self.opt.imgDir, "train", "query")
            # dir_q = os.path.join("E:/shujuji/nerfCambridge4VPR/CambridgeNerf_train1_4", "val", "query")
            poseq, scenesq = gen_pose(q_id_train, preds_train, dir_q)  # 需要增加数据的pose，使用的q_id_train筛选
            print(poseq.shape)  # [58, 4, 4]
            print(len(scenesq))  # 58
            # print(q_id_train)
            # todo [58, 4, 4]->[58, n, 4, 4]
            # determine bounding box
            b_min = [poseq[:, 0, 3].min() - self.args_nerf.d_max, poseq[:, 1, 3].min() - self.args_nerf.d_max,
                     poseq[:, 2, 3].min() - self.args_nerf.d_max]
            b_max = [poseq[:, 0, 3].max() + self.args_nerf.d_max, poseq[:, 1, 3].max() + self.args_nerf.d_max,
                     poseq[:, 2, 3].max() + self.args_nerf.d_max]
            poses_target = perturb_single_render_pose(poseq, self.args_nerf.rvs_trans, self.args_nerf.rvs_rotation, 10)
            # poses_target = poseq.unsqueeze(1)  #测试需要删去
            # poses_target = poses_target.repeat(1, 3, 1, 1) #测试需要删去
            poses_target = poses_target.to("cuda")
            for i in range(poses_target.shape[0]):
                for j in range(poses_target.shape[1]):
                    for k in range(3):
                        if poses_target[i, j, k, 3] < b_min[k]:
                            poses_target[i, j, k, 3] = b_min[k]
                        elif poses_target[i, j, k, 3] > b_max[k]:
                            poses_target[i, j, k, 3] = b_max[k]
            # print(poses_target)
            dir_d = os.path.join(self.opt.imgDir, "train", "database")
            imageq = find_image(dir_q)
            imaged = find_image(dir_d)
            dir_d_pose = os.path.join(dir_d, "poses4")
            pose_database, _ = read_pose(dir_d_pose)  # (1342,4,4)
            # print(pose_database.shape)
            image_database = []
            img_idx_list = []
            # print(len(imaged))
            for idx in range(len(imaged)):
                img = Image.open(imaged[idx])
                # img = self.input_transform(img)
                img = self.image_to_tensor(img)
                image_database.append(img)

            for idx in range(len(imageq)):
                img = load_image(imageq[idx])
                img = self.data_transform(img)
                yuv = rgb_to_yuv(img)
                y_img = yuv[0]  # extract y channel only
                hist = torch.histc(y_img, bins=10, min=0., max=1.)  # compute intensity histogram
                hist = hist / (hist.sum()) * 100  # convert to histogram density, in terms of percentage per bin
                hist = torch.round(hist)
                hist = hist.unsqueeze(0)
                # print(hist.shape)
                img_idx_list.append(hist.cpu())

            img_idxs = torch.stack(img_idx_list).detach()

            # print(img_idxs.shape)
            selected_img_idxs = img_idxs[q_id_train]

            img_idxs_repeated = torch.repeat_interleave(selected_img_idxs, repeats=3, dim=0)
            # print(img_idxs_repeated.shape)
            img_idxs = img_idxs_repeated
            # print(image_database.shape)
            imgs_database = torch.stack(image_database, dim=0)  # (n_database, 3, 224, 224)
            imgs_database = F.interpolate(imgs_database, size=[224, 224], mode="area")

            imgs = torch.zeros((len(q_id_train), 5, 3, 224, 224))
            # 截取每个数组的前5个元素
            gt_train = gt_train = np.array([
                arr[:5] if len(arr) >= 5 else np.pad(arr, (0, 5 - len(arr)), mode='wrap')
                for arr in gt_train
            ])
            # 转换为二维数组，形状为 [len(gt), 5]
            gt_train = np.array(gt_train)
            index_array = gt_train[q_id_train, :5]
            # print(index_array.shape)
            for i, indices in enumerate(index_array):
                imgs[i] = imgs_database[indices]
            # imgs = imgs_database[preds_train[q_id_train, :5]] # [58, 5, 3, 224, 224]
            images_ref = imgs
            print(images_ref.shape)
            # images_ref = imgs[:20]               # [20, 5, 3, 224, 224]

            pose = torch.zeros((len(q_id_train), 5, 4, 4))
            index_array = gt_train[q_id_train, :5]
            for i, indices in enumerate(index_array):
                pose[i] = pose_database[indices]
            # pose = pose_database[preds_train[q_id_train, :5]] # [58, 5, 4, 4]
            poses_ref = pose
            # poses_ref = pose[:20] #  [20, 5, 4, 4]

            print("img shape")
            # print(imgs.shape)
            print(images_ref.shape)
            print("pose shape")
            # print(pose.shape)
            print(poses_ref.shape)

            predict = NBP_Cam(self.model_name, images_ref, poses_ref, poses_target) #[58 3]

            predict = predict.to(self.device)
            del images_ref, poses_ref
            expanded_predict = predict.unsqueeze(-1).unsqueeze(-1).expand(-1, 3, 4, 4)
            selected_poses = torch.gather(poses_target, 1, expanded_predict)  # (58,n,4,4)->(58,3,4,4)

            selected_scene = [item for sublist in scenesq for item in [sublist] * 3]  # scenes重塑为了后续fixcoord

            # print(selected_poses.shape)
            hwf = [480, 854, 744.0]
            world_setup_dict = {
                'pose_scale': 0.3027,
                'pose_scale2': 0.2,
                'move_all_cam_vec': [0.0, 0.0, 0.0],
            }
            world_setup_dict_OldHospital = {
                'pose_scale': 0.3027,
                'pose_scale2': 0.2,
                'move_all_cam_vec': [0.0, 0.0, 5.0],
            }

            world_setup_dict_ShopFacade = {
                'pose_scale': 0.3027,
                'pose_scale2': 0.32,
                'move_all_cam_vec': [0.0, 0.0, 2.5],
            }
            # print(selected_poses[0,0])
            # print("after_fixcoord")
            # print(selected_scene)
            selected_poses, bounds, idx = fix_coord(selected_poses, selected_scene,
                                                    pose_avg_stats_file='data/poses_avg_stats')

            save_folder_rgb="Cambridge/CambridgeNerf_train1_4/train/database/rgb"
            save_folder_poses = "Cambridge/CambridgeNerf_train1_4/train/database/poses"
            save_folder_poses4 = "Cambridge/CambridgeNerf_train1_4/train/database/poses4"
            os.makedirs(save_folder_rgb, exist_ok=True)
            os.makedirs(save_folder_poses, exist_ok=True)
            os.makedirs(save_folder_poses4, exist_ok=True)
            import glob
            image_remove_files = glob.glob(os.path.join(save_folder_rgb, 'zadd*.png'))
            poses_remove_files = glob.glob(os.path.join(save_folder_poses, 'zadd*.txt'))
            poses4_remove_files = glob.glob(os.path.join(save_folder_poses4, 'zadd*.txt'))
            # 合并所有要删除的文件列表
            files_to_delete = image_remove_files + poses_remove_files + poses4_remove_files
            for file in files_to_delete:
                try:
                    os.remove(file)
                    print(f'Successfully deleted: {file}')
                except Exception as e:
                    print(f'Error deleting file {file}: {e}')

            # print(selected_poses[0])
            # print(img_idxs[0])
            if (idx[0] > 0):
                # GreatCourt
                render_virtual_Cam_imgs(self.args_nerf, selected_poses[:idx[0]], img_idxs[:idx[0]], hwf, self.device,
                                        self.render_kwargs_test_GreatCourt, world_setup_dict, epoch, scene="GreatCourt")
            if ((idx[1] - idx[0]) > 0):
                # KingsCollege
                render_virtual_Cam_imgs(self.args_nerf, selected_poses[idx[0]:idx[1]], img_idxs[idx[0]:idx[1]], hwf,
                                        self.device,
                                        self.render_kwargs_test_KingsCollege, world_setup_dict, epoch,
                                        scene="KingsCollege")
            if ((idx[2] - idx[1]) > 0):
                # OldHospital
                render_virtual_Cam_imgs(self.args_nerf, selected_poses[idx[1]:idx[2]], img_idxs[idx[1]:idx[2]], hwf,
                                        self.device,
                                        self.render_kwargs_test_OldHospital, world_setup_dict_OldHospital, epoch,
                                        scene="OldHospital")
            if ((idx[3] - idx[2]) > 0):
                # ShopFacade
                render_virtual_Cam_imgs(self.args_nerf, selected_poses[idx[2]:idx[3]], img_idxs[idx[2]:idx[3]], hwf,
                                        self.device,
                                        self.render_kwargs_test_ShopFacade, world_setup_dict_ShopFacade, epoch,
                                        scene="ShopFacade")
            if ((idx[4] - idx[3]) > 0):
                # StMarysChurch
                render_virtual_Cam_imgs(self.args_nerf, selected_poses[idx[3]:idx[4]], img_idxs[idx[3]:idx[4]], hwf,
                                        self.device,
                                        self.render_kwargs_test_StMarysChurch, world_setup_dict, epoch,
                                        scene="StMarysChurch")
            del poses_target, imgs_database
            torch.cuda.empty_cache()
        if mode == "val" and self.opt.Datasetname == "NEU" and len(q_id_train) > 0:
            device = torch.device("cuda")
            dir_q = os.path.join(self.opt.imgDir, "train", "query")
            # dir_q = os.path.join("E:/shujuji/nerfCambridge4VPR/CambridgeNerf_train1_4", "val", "query")
            poseq, scenesq = gen_pose(q_id_train, preds_train, dir_q)  # 需要增加数据的pose，使用的q_id_train筛选
            print(poseq.shape)  # [58, 4, 4]
            print(len(scenesq))  # 58
            # print(q_id_train)
            # todo [58, 4, 4]->[58, n, 4, 4]
            # determine bounding box
            b_min = [poseq[:, 0, 3].min() - self.args_nerf.d_max, poseq[:, 1, 3].min() - self.args_nerf.d_max,
                     poseq[:, 2, 3].min() - self.args_nerf.d_max]
            b_max = [poseq[:, 0, 3].max() + self.args_nerf.d_max, poseq[:, 1, 3].max() + self.args_nerf.d_max,
                     poseq[:, 2, 3].max() + self.args_nerf.d_max]
            poses_target = perturb_single_render_pose(poseq, self.args_nerf.rvs_trans, self.args_nerf.rvs_rotation, 10)
            # poses_target = poseq.unsqueeze(1)  #测试需要删去
            # poses_target = poses_target.repeat(1, 3, 1, 1) #测试需要删1
            poses_target = poses_target.to("cuda")
            for i in range(poses_target.shape[0]):
                for j in range(poses_target.shape[1]):
                    for k in range(3):
                        if poses_target[i, j, k, 3] < b_min[k]:
                            poses_target[i, j, k, 3] = b_min[k]
                        elif poses_target[i, j, k, 3] > b_max[k]:
                            poses_target[i, j, k, 3] = b_max[k]
            # print(poses_target)
            dir_d = os.path.join(self.opt.imgDir, "train", "database")
            imageq = find_image(dir_q)
            imaged = find_image(dir_d)
            dir_d_pose = os.path.join(dir_d, "poses4")
            pose_database, _ = read_pose(dir_d_pose)  # (1342,4,4)
            # print(pose_database.shape)
            image_database = []
            img_idx_list = []
            # print(len(imaged))
            for idx in range(len(imaged)):
                img = Image.open(imaged[idx])
                # img = self.input_transform(img)
                img = self.image_to_tensor(img)
                image_database.append(img)

            for idx in range(len(imageq)):
                img = load_image(imageq[idx])
                img = self.data_transform(img)
                yuv = rgb_to_yuv(img)
                y_img = yuv[0]  # extract y channel only
                hist = torch.histc(y_img, bins=10, min=0., max=1.)  # compute intensity histogram
                hist = hist / (hist.sum()) * 100  # convert to histogram density, in terms of percentage per bin
                hist = torch.round(hist)
                hist = hist.unsqueeze(0)
                # print(hist.shape)
                img_idx_list.append(hist.cpu())

            img_idxs = torch.stack(img_idx_list).detach()

            selected_img_idxs = img_idxs[q_id_train]

            img_idxs_repeated = torch.repeat_interleave(selected_img_idxs, repeats=3, dim=0)
            # print(img_idxs_repeated.shape)
            img_idxs = img_idxs_repeated
            # print(image_database.shape)
            imgs_database = torch.stack(image_database, dim=0)  # (n_database, 3, 224, 224)
            imgs_database = F.interpolate(imgs_database, size=[224, 224], mode="area")

            imgs = torch.zeros((len(q_id_train), 5, 3, 224, 224))
            # 截取每个数组的前5个元素
            gt_train = gt_train = np.array([
                arr[:5] if len(arr) >= 5 else np.pad(arr, (0, 5 - len(arr)), mode='wrap')
                for arr in gt_train
            ])
            # 转换为二维数组，形状为 [len(gt), 5]
            gt_train = np.array(gt_train)
            index_array = gt_train[q_id_train, :5]
            # print(index_array.shape)
            for i, indices in enumerate(index_array):
                imgs[i] = imgs_database[indices]
            # imgs = imgs_database[preds_train[q_id_train, :5]] # [58, 5, 3, 224, 224]
            images_ref = imgs
            print(images_ref.shape)
            # images_ref = imgs[:20]               # [20, 5, 3, 224, 224]

            pose = torch.zeros((len(q_id_train), 5, 4, 4))
            index_array = gt_train[q_id_train, :5]
            for i, indices in enumerate(index_array):
                pose[i] = pose_database[indices]
            # pose = pose_database[preds_train[q_id_train, :5]] # [58, 5, 4, 4]
            poses_ref = pose
            # poses_ref = pose[:20] #  [20, 5, 4, 4]

            print("img shape")
            # print(imgs.shape)
            print(images_ref.shape)
            print("pose shape")
            # print(pose.shape)
            print(poses_ref.shape)

            predict = NBP_NEU(self.model_name, images_ref, poses_ref, poses_target, self.device)  # [58 3]
            predict = predict.to(self.device)
            del images_ref, poses_ref
            expanded_predict = predict.unsqueeze(-1).unsqueeze(-1).expand(-1, 3, 4, 4)
            selected_poses = torch.gather(poses_target, 1, expanded_predict)  # (58,n,4,4)->(58,3,4,4)
            selected_scene = [item for sublist in scenesq for item in [sublist] * 3]  # scenes重塑为了后续fixcoord
            # print(selected_scene)
            hwf = [270, 480, 211.]
            world_setup_dict = {
                'pose_scale': 1,
                'pose_scale2': 1,
                'move_all_cam_vec': [0.0, 0.0, 0.0],
            }

            selected_poses, bounds, idx = fix_coord_NEU(selected_poses, selected_scene)

            save_folder_rgb = "Cambridge/NEUnightNerf/train/database/rgb"
            save_folder_poses = "Cambridge/NEUnightNerf/train/database/poses"
            save_folder_poses4 = "Cambridge/NEUnightNerf/train/database/poses4"
            os.makedirs(save_folder_rgb, exist_ok=True)
            os.makedirs(save_folder_poses, exist_ok=True)
            os.makedirs(save_folder_poses4, exist_ok=True)
            import glob
            image_remove_files = glob.glob(os.path.join(save_folder_rgb, 'zadd*.png'))
            poses_remove_files = glob.glob(os.path.join(save_folder_poses, 'zadd*.txt'))
            poses4_remove_files = glob.glob(os.path.join(save_folder_poses4, 'zadd*.txt'))
            # 合并所有要删除的文件列表
            files_to_delete = image_remove_files + poses_remove_files + poses4_remove_files
            for file in files_to_delete:
                try:
                    os.remove(file)
                    print(f'Successfully deleted: {file}')
                except Exception as e:
                    print(f'Error deleting file {file}: {e}')

            for i in range(5):
                if i == 0:
                    if (idx[0] > 0):
                        render_virtual_NEU_imgs(self.args_nerf, selected_poses[:idx[0]], img_idxs[:idx[0]], hwf,
                                                self.device,
                                                self.render_kwargs_test[0], world_setup_dict, epoch,
                                                scene=self.NEU_folders[0]
                                                )
                elif i > 0:
                    if ((idx[i] - idx[i - 1]) > 0):
                        render_virtual_NEU_imgs(self.args_nerf, selected_poses[idx[i - 1]:idx[i]],
                                                img_idxs[idx[i - 1]:idx[i]], hwf,
                                                self.device, self.render_kwargs_test[i], world_setup_dict, epoch,
                                                scene=self.NEU_folders[i]
                                                )
        if mode == "val" and self.opt.Datasetname == "SIASUN" and len(q_id_train) > 0 and epoch > 25:
            device = torch.device("cuda")
            dir_q = os.path.join(self.opt.imgDir, "train", "query")
            # dir_q = os.path.join("E:/shujuji/nerfCambridge4VPR/CambridgeNerf_train1_4", "val", "query")
            poseq, scenesq = gen_pose(q_id_train, preds_train, dir_q)  # 需要增加数据的pose，使用的q_id_train筛选
            print(poseq.shape)  # [58, 4, 4]
            print(len(scenesq))  # 58
            # print(q_id_train)
            # todo [58, 4, 4]->[58, n, 4, 4]
            # determine bounding box
            b_min = [poseq[:, 0, 3].min() - self.args_nerf.d_max, poseq[:, 1, 3].min() - self.args_nerf.d_max,
                     poseq[:, 2, 3].min() - self.args_nerf.d_max]
            b_max = [poseq[:, 0, 3].max() + self.args_nerf.d_max, poseq[:, 1, 3].max() + self.args_nerf.d_max,
                     poseq[:, 2, 3].max() + self.args_nerf.d_max]
            poses_target = perturb_single_render_pose(poseq, self.args_nerf.rvs_trans, self.args_nerf.rvs_rotation, 10)
            # poses_target = poseq.unsqueeze(1)  #测试需要删去
            # poses_target = poses_target.repeat(1, 3, 1, 1) #测试需要删去
            poses_target = poses_target.to("cuda")
            for i in range(poses_target.shape[0]):
                for j in range(poses_target.shape[1]):
                    for k in range(3):
                        if poses_target[i, j, k, 3] < b_min[k]:
                            poses_target[i, j, k, 3] = b_min[k]
                        elif poses_target[i, j, k, 3] > b_max[k]:
                            poses_target[i, j, k, 3] = b_max[k]
            # print(poses_target)
            dir_d = os.path.join(self.opt.imgDir, "train", "database")
            imageq = find_image(dir_q)
            imaged = find_image(dir_d)
            dir_d_pose = os.path.join(dir_d, "poses4")
            pose_database, _ = read_pose(dir_d_pose)  # (1342,4,4)
            # print(pose_database.shape)
            image_database = []
            img_idx_list = []
            # print(len(imaged))
            for idx in range(len(imaged)):
                img = Image.open(imaged[idx])
                # img = self.input_transform(img)
                img = self.image_to_tensor(img)
                image_database.append(img)

            for idx in range(len(imageq)):
                img = load_image(imageq[idx])
                img = self.data_transform(img)
                yuv = rgb_to_yuv(img)
                y_img = yuv[0]  # extract y channel only
                hist = torch.histc(y_img, bins=10, min=0., max=1.)  # compute intensity histogram
                hist = hist / (hist.sum()) * 100  # convert to histogram density, in terms of percentage per bin
                hist = torch.round(hist)
                hist = hist.unsqueeze(0)
                # print(hist.shape)
                img_idx_list.append(hist.cpu())

            img_idxs = torch.stack(img_idx_list).detach()

            # print(img_idxs.shape)
            selected_img_idxs = img_idxs[q_id_train]

            img_idxs_repeated = torch.repeat_interleave(selected_img_idxs, repeats=3, dim=0)
            # print(img_idxs_repeated.shape)
            img_idxs = img_idxs_repeated
            # print(image_database.shape)
            imgs_database = torch.stack(image_database, dim=0)  # (n_database, 3, 224, 224)
            imgs_database = F.interpolate(imgs_database, size=[224, 224], mode="area")

            imgs = torch.zeros((len(q_id_train), 5, 3, 224, 224))
            # 截取每个数组的前5个元素
            gt_train = gt_train = np.array([
                arr[:5] if len(arr) >= 5 else np.pad(arr, (0, 5 - len(arr)), mode='wrap')
                for arr in gt_train
            ])
            # 转换为二维数组，形状为 [len(gt), 5]
            gt_train = np.array(gt_train)
            index_array = gt_train[q_id_train, :5]
            # print(index_array.shape)
            for i, indices in enumerate(index_array):
                imgs[i] = imgs_database[indices]
            # imgs = imgs_database[preds_train[q_id_train, :5]] # [58, 5, 3, 224, 224]
            images_ref = imgs
            print(images_ref.shape)
            # images_ref = imgs[:20]               # [20, 5, 3, 224, 224]

            pose = torch.zeros((len(q_id_train), 5, 4, 4))
            index_array = gt_train[q_id_train, :5]
            for i, indices in enumerate(index_array):
                pose[i] = pose_database[indices]
            # pose = pose_database[preds_train[q_id_train, :5]] # [58, 5, 4, 4]
            poses_ref = pose
            # poses_ref = pose[:20] #  [20, 5, 4, 4]

            print("img shape")
            # print(imgs.shape)
            print(images_ref.shape)
            print("pose shape")
            # print(pose.shape)
            print(poses_ref.shape)

            predict = NBP_SIASUN(self.model_name, images_ref, poses_ref, poses_target, self.device)  # [58 3]



            predict = predict.to(self.device)
            del images_ref, poses_ref
            expanded_predict = predict.unsqueeze(-1).unsqueeze(-1).expand(-1, 3, 4, 4)
            selected_poses = torch.gather(poses_target, 1, expanded_predict)  # (58,n,4,4)->(58,3,4,4)
            # 需要复原以上
            selected_scene = [item for sublist in scenesq for item in [sublist] * 3]  # scenes重塑为了后续fixcoord

            hwf = [270, 480, 396.]
            world_setup_dict = {
                'pose_scale': 1,
                'pose_scale2': 1,
                'move_all_cam_vec': [0.0, 0.0, 0.0],
            }

            selected_poses, bounds, idx = fix_coord_SIASUN(selected_poses, selected_scene)

            save_folder_rgb="Cambridge/Siasun1_3Nerf/train/database/rgb"
            save_folder_poses = "Cambridge/Siasun1_3Nerf/train/database/poses"
            save_folder_poses4 = "Cambridge/Siasun1_3Nerf/train/database/poses4"
            os.makedirs(save_folder_rgb, exist_ok=True)
            os.makedirs(save_folder_poses, exist_ok=True)
            os.makedirs(save_folder_poses4, exist_ok=True)
            import glob
            image_remove_files = glob.glob(os.path.join(save_folder_rgb, 'zadd*.png'))
            poses_remove_files = glob.glob(os.path.join(save_folder_poses, 'zadd*.txt'))
            poses4_remove_files = glob.glob(os.path.join(save_folder_poses4, 'zadd*.txt'))
            # 合并所有要删除的文件列表
            files_to_delete = image_remove_files + poses_remove_files + poses4_remove_files
            for file in files_to_delete:
                try:
                    os.remove(file)
                    print(f'Successfully deleted: {file}')
                except Exception as e:
                    print(f'Error deleting file {file}: {e}')

            for i in range(5):
                if i == 0:
                    if (idx[0] > 0):
                        render_virtual_SIA_imgs(self.args_nerf, selected_poses[:idx[0]], img_idxs[:idx[0]], hwf,
                                                self.device,
                                                self.render_kwargs_test[0], world_setup_dict, epoch,
                                                scene=self.SIASUN_folders[0]
                                                )
                elif i > 0:
                    if ((idx[i] - idx[i - 1]) > 0):
                        render_virtual_SIA_imgs(self.args_nerf, selected_poses[idx[i - 1]:idx[i]],
                                                img_idxs[idx[i - 1]:idx[i]], hwf,
                                                self.device, self.render_kwargs_test[i], world_setup_dict, epoch,
                                                scene=self.SIASUN_folders[i]
                                                )
            del poses_target, imgs_database
            torch.cuda.empty_cache()
        if save_embs:
            with open(join(self.opt.runsPath, '{}_db_embeddings_{}.pickle'.format(self.opt.split,
                                                                                  self.opt.resume.split('.')[-3].split(
                                                                                      '_')[-1])), 'wb') as handle:
                pickle.dump(mu_q, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(mu_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # pickle.dump(sigma_q, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # pickle.dump(sigma_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(preds, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(dists, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(gt, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(whole_mu, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(whole_var, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('embeddings saved for post processing')

        return recall_at_k, None


if __name__ == '__main__':
    from options import Options

    options_handler = Options()
    options = options_handler.parse()
    options.phase = "test_tea"
    options.resume = "logs/ckpt_best.pth.tar"
    options.cGPU = 1
    options.nGPU = 0
    tr = Trainer(options)
    tr.test()