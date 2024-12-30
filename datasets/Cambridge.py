from collections import namedtuple
from os.path import join

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
from torchvision.transforms import InterpolationMode
import os

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

def get_whole_training_set(opt, onlyDB=False, forCluster=False, return_labels=False):
    return WholeDatasetFromStruct(opt, join(opt.structDir, 'CambridgeNerf_train1_4/train'), join(opt.imgDir,'train'), input_transform=input_transform(opt), onlyDB=onlyDB, forCluster=forCluster, return_labels=return_labels)


def get_whole_val_set(opt, return_labels=False):
    return WholeDatasetFromStruct(opt, join(opt.structDir, 'CambridgeNerf_train1_4/val'), join(opt.imgDir, 'val'), input_transform=input_transform(opt), return_labels=return_labels)


def get_whole_test_set(opt, return_labels=False):
    return WholeDatasetFromStruct(opt, join(opt.structDir, 'CambridgeNerf_train1_4/test'), join(opt.imgDir, 'test'), input_transform=input_transform(opt), return_labels=return_labels)


def get_training_query_set(opt, margin=0.1):
    return QueryDatasetFromStruct(opt, join(opt.structDir, 'CambridgeNerf_train1_4/train'), join(opt.imgDir,'train'), input_transform=input_transform(opt), margin=margin)


def get_val_query_set(opt, margin=0.1):
    return QueryDatasetFromStruct(opt, join(opt.structDir, 'CambridgeNerf_train1_4/val'), join(opt.imgDir, 'val'), input_transform=input_transform(opt), margin=margin)


def get_quad_set(opt, margin, margin2):
    return QuadrupletDataset(opt, join(opt.structDir, 'CambridgeNerf_train1_4/train'), join(opt.imgDir,'train'), input_transform=input_transform(opt), margin=margin, margin2=margin2)


dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ', 'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])


def parse_dbStruct(path):

    data_dir, spilt = os.path.split(path)

    dbpath=join(path, 'database/rgb')
    dbfiles = os.listdir(dbpath)
    dbImage = []

    # 遍历文件夹中的所有文件
    for file in dbfiles:
        # 检查文件是否为PNG图片
        if file.endswith(".png"):
            # 将图片名字添加到列表中
            dbImage.append(file)

    qpath=join(path, 'query/rgb')
    qfiles = os.listdir(qpath)
    qImage = []

    # 遍历文件夹中的所有文件
    for file in qfiles:
        # 检查文件是否为PNG图片
        if file.endswith(".png"):
            # 将图片名字添加到列表中
            qImage.append(file)

    dbnumpath=join(path, 'database/poses')
    # 获取文件夹中所有的txt文件名
    txt_files = [file for file in os.listdir(dbnumpath) if file.endswith('.txt')]

    # 创建一个空列表用于存储所有的数据
    dbnumdata = []

    # 依次读取每个txt文件中的数据
    for file in txt_files:
        with open(os.path.join(dbnumpath, file), 'r') as f:
            # 读取一行数据，并转换为列表
            line = f.readline().strip()  # 读取一行数据并去除空白字符
            if line:  # 确保读取到的行不为空
                line = list(map(float, line.split()))
                 # 将数据列表添加到总列表中
                dbnumdata.append(line)
    utmDb=np.array(dbnumdata)

    qnumpath=join(path, 'query/poses')
    # 获取文件夹中所有的txt文件名
    txt_files = [file for file in os.listdir(qnumpath) if file.endswith('.txt')]

    # 创建一个空列表用于存储所有的数据
    qnumdata = []

    # 依次读取每个txt文件中的数据
    for file in txt_files:
        with open(os.path.join(qnumpath, file), 'r') as f:
            line = f.readline().strip()  # 读取一行数据并去除空白字符
            if line:  # 确保读取到的行不为空
            # 读取一行数据，并转换为列表
                line = list(map(float, line.split()))
                # 将数据列表添加到总列表中
                qnumdata.append(line)
    utmQ=np.array(qnumdata)

    dataset = 'nuscenes'

    whichSet = spilt


    numDb = len(dbImage)
    numQ = len(qImage)

    posDistThr = 25
    posDistSqThr = 625
    nonTrivPosDistSqThr = 100
    #print(whichSet)
    #print(dbImage)
    #print(utmDb)
    #print(qImage)
    #print(utmQ)
    print(numDb)
    #print(numQ)
    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, utmQ, numDb, numQ, posDistThr, posDistSqThr, nonTrivPosDistSqThr)


class WholeDatasetFromStructForCluster(data.Dataset):
    def __init__(self, opt, structFile, img_dir, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(structFile)

        self.images = [join(img_dir, 'database', 'rgb', dbIm) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += [join(img_dir, 'query', 'rgb', qIm) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None
        print("num images")
        print(len(self.images))
    def __getitem__(self, index):
        img = Image.open(self.images[index])
        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)
            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ, radius=self.dbStruct.nonTrivPosDistSqThr**0.5)       # TODO: sort!!

        return self.positives


class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, opt, structFile, img_dir, input_transform=None, onlyDB=False, forCluster=False, return_labels=False):
        super().__init__()
        self.opt = opt
        self.forCluster = forCluster
        self.return_labels = return_labels

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(structFile)

        self.images = [join(img_dir, 'database','rgb',  dbIm) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += [join(img_dir, 'query','rgb',  qIm) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

    def load_images(self, index):
        filename = self.images[index]
        # imgs = []
        img = Image.open(filename)
        if self.input_transform:
            img = self.input_transform(img)
        # imgs.append(img)
        # imgs = torch.stack(imgs, 0)

        return img, index

    def __getitem__(self, index):
        if self.forCluster:
            img = Image.open(self.images[index])
            if self.input_transform:
                img = self.input_transform(img)

            return img, index
        else:
            if self.return_labels:
                imgs, index = self.load_images(index)
                return imgs, index, self.dbStruct.utmQ[index]
            else:
                imgs, index = self.load_images(index)
                return imgs, index

    def __len__(self):
        return len(self.images)

    def get_databases(self):
        return self.dbStruct.utmDb

    def get_queries(self):
        return self.dbStruct.utmQ

    def get_positives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)
            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ, radius=self.dbStruct.nonTrivPosDistSqThr**0.5)       # TODO: sort!!

        return self.positives


def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

    Args:
        data: list of tuple (query, positive, negatives).
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)                                                 # ([8, 3, 200, 200]) = [(3, 200, 200), (3, 200, 200), ..  ]     ([8, 1, 3, 200, 200])
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)                                                            # ([80, 3, 200, 200]) ([80, 1, 3, 200, 200])
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices


class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, opt, structFile, img_dir, nNegSample=1000, nNeg=10, margin=0.1, input_transform=None):
        super().__init__()
        self.opt = opt
        self.img_dir = img_dir
        self.input_transform = input_transform
        self.margin = margin

        self.dbStruct = parse_dbStruct(structFile)
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = nNegSample             # number of negatives to randomly sample
        self.nNeg = nNeg                         # number of negatives used for training

        # potential positives are those within nontrivial threshold range
        # fit NN to find them, search by radius
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)

        # TODO use sqeuclidean as metric?
        # 搜索100之内的
        self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ, radius=self.dbStruct.nonTrivPosDistSqThr**0.5, return_distance=False))
        # radius returns unsorted, sort once now so we dont have to later
        for i, posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)
        # its possible some queries don't have any non trivial potential positives
        # lets filter those out
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives]) > 0)[0]

        # potential negatives are those outside of posDistThr range
        #搜索25之类的
        potential_positives = knn.radius_neighbors(self.dbStruct.utmQ, radius=self.dbStruct.posDistThr, return_distance=False)

        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb), pos, assume_unique=True))

        self.cache = None                        # filepath of HDF5 containing feature vectors for images

        self.negCache = [np.empty((0, ), dtype=np.int64) for _ in range(self.dbStruct.numQ)]

    def load_images(self, filename):
        # imgs = []
        img = Image.open(filename)
        if self.input_transform:
            img = self.input_transform(img)
        # imgs.append(img)
        # imgs = torch.stack(imgs, 0)

        return img

    def __getitem__(self, index):

        index = self.queries[index]              # re-map index to match dataset
        with h5py.File(self.cache, mode='r') as h5:
            h5feat = h5.get("features")
            qOffset = self.dbStruct.numDb

            qFeat = h5feat[index + qOffset]
            posFeat = h5feat[self.nontrivial_positives[index].tolist()]
            qFeat = torch.tensor(qFeat)
            posFeat = torch.tensor(posFeat)
            dist = torch.norm(qFeat - posFeat, dim=1, p=None)
            result = dist.topk(1, largest=False)
            dPos, posNN = result.values, result.indices
            posIndex = self.nontrivial_positives[index][posNN].item()

            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)         # randomly choose potential_negatives
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))               # remember negSamples history for each query

            negFeat = h5feat[negSample.tolist()]
            negFeat = torch.tensor(negFeat)
            dist = torch.norm(qFeat - negFeat, dim=1, p=None)
            result = dist.topk(self.nNeg * 10, largest=False)
            dNeg, negNN = result.values, result.indices

            if self.opt.loss == 'cont':
                violatingNeg = dNeg.numpy() < self.margin**0.5
            else:
                violatingNeg = dNeg.numpy() < dPos.numpy() + self.margin**0.5

            if np.sum(violatingNeg) < 1:
                return None

            negNN = negNN.numpy()
            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = negIndices

        query = self.load_images(join(self.img_dir, 'query', 'rgb', self.dbStruct.qImage[index]))
        positive = self.load_images(join(self.img_dir, 'database','rgb',  self.dbStruct.dbImage[posIndex]))

        negatives = []
        for negIndex in negIndices:
            negative = self.load_images(join(self.img_dir, 'database', 'rgb', self.dbStruct.dbImage[negIndex]))
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)    # ([10, 3, 200, 200])
        return query, positive, negatives, [index, posIndex] + negIndices.tolist()

    def __len__(self):
        return len(self.queries)


class QuadrupletDataset(data.Dataset):
    def __init__(self, opt, structFile, img_dir, nNegSample=1000, nNeg=10, margin=0.1, margin2=0.05, input_transform=None):
        super().__init__()
        self.opt = opt
        self.img_dir = img_dir
        self.input_transform = input_transform
        self.margin = margin
        self.margin2 = margin2

        self.dbStruct = parse_dbStruct(structFile)
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = nNegSample             # number of negatives to randomly sample
        self.nNeg = nNeg                         # number of negatives used for training

        # potential positives are those within nontrivial threshold range, fit NN to find them, search by radius
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)

        self.db_potential_positives = knn.radius_neighbors(self.dbStruct.utmDb, radius=self.dbStruct.posDistThr, return_distance=False)    # 6312
        self.db_potential_negatives = []
        for pos in self.db_potential_positives:
            self.db_potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb), pos, assume_unique=True))

        # TODO use sqeuclidean as metric?
        self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ, radius=self.dbStruct.nonTrivPosDistSqThr**0.5, return_distance=False)) # 7075
                                                                                                                                                         # radius returns unsorted, sort once now so we dont have to later
        for i, posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)
                                                                                                                                                         # its possible some queries don't have any non trivial potential positives, lets filter those out
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives]) > 0)[0]

        # potential negatives are those outside of posDistThr range
        self.potential_positives = knn.radius_neighbors(self.dbStruct.utmQ, radius=self.dbStruct.posDistThr, return_distance=False)

        self.potential_negatives = []
        for pos in self.potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb), pos, assume_unique=True))

        self.cache = None                        # filepath of HDF5 containing feature vectors for images

        self.negCache = [np.empty((0, ), dtype=np.int64) for _ in range(self.dbStruct.numQ)]

    def load_images(self, filename):
        img = Image.open(filename)
        if self.input_transform:
            img = self.input_transform(img)
        return img

    def __getitem__(self, index):
        index = self.queries[index]              # re-map index to match dataset
        with h5py.File(self.cache, mode='r') as h5:
            h5feat = h5.get("features")
            qOffset = self.dbStruct.numDb

            qFeat = h5feat[index + qOffset]
            tmp = self.nontrivial_positives[index]
            tmp = tmp.tolist()
            posFeat = h5feat[self.nontrivial_positives[index].tolist()]
            qFeat = torch.tensor(qFeat)
            posFeat = torch.tensor(posFeat)
            dist = torch.norm(qFeat - posFeat, dim=1, p=None)
            result = dist.topk(1, largest=False)                                                   # choose the closet positive
            dPos, posNN = result.values, result.indices
            posIndex = self.nontrivial_positives[index][posNN].item()

            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)         # randomly choose potential_negatives
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))               # encourage to sample from last negIndices + current last negIndices

            negFeat = h5feat[negSample.tolist()]
            negFeat = torch.tensor(negFeat)
            dist = torch.norm(qFeat - negFeat, dim=1, p=None)
            result = dist.topk(self.nNeg * 10, largest=False)
            dNeg, negNN = result.values, result.indices

            # try to find negatives that are within margin, if there aren't any return none
            violatingNeg = dNeg.numpy() < dPos.numpy() + self.margin**0.5

            if np.sum(violatingNeg) < 1:
                # if none are violating then skip this query
                return None

            negNN = negNN.numpy()
            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = negIndices

        query = self.load_images(join(self.img_dir, 'query', 'rgb', self.dbStruct.qImage[index]))
        positive = self.load_images(join(self.img_dir, 'database', 'rgb', self.dbStruct.dbImage[posIndex]))

        negatives = []
        negatives2 = []
        negIndices2 = []
        for negIndex in negIndices:
            anchor_neg_negs = np.random.choice(self.db_potential_negatives[negIndex], 1000, replace=False)
            anchor_poss = self.potential_positives[index]
            anchor_neg_negs_clean = np.setdiff1d(anchor_neg_negs, anchor_poss, assume_unique=True)
            anchor_neg_negs_clean = np.sort(anchor_neg_negs_clean)
            with h5py.File(self.cache, mode='r') as h5:
                h5feat = h5.get("features")
                negFeat = h5feat[anchor_neg_negs_clean.tolist()]
                negFeat = torch.tensor(negFeat)
                dist = torch.norm(qFeat - negFeat, dim=1, p=None)
                result = dist.topk(self.nNeg * 10, largest=False)
                dNeg, negNN = result.values, result.indices
                violatingNeg = dNeg.numpy() < dPos.numpy() + self.margin2**0.5                     # increase negative samples by using **0.5
                if np.sum(violatingNeg) < 1:
                    return None
                negNN = negNN.numpy()
                negNN = negNN[violatingNeg][:1]
                neg2Index = anchor_neg_negs_clean[negNN].astype(np.int32)[0]

            negative = self.load_images(join(self.img_dir, 'database', 'rgb', self.dbStruct.dbImage[negIndex]))
            negative2 = self.load_images(join(self.img_dir, 'database', 'rgb', self.dbStruct.dbImage[neg2Index]))
            negatives.append(negative)
            negatives2.append(negative2)
            negIndices2.append(neg2Index)

        negatives = torch.stack(negatives, 0)    # ([num_neg, C, H, W])
        negatives2 = torch.stack(negatives2, 0)  # ([num_neg, C, H, W])
        return query, positive, negatives, negatives2, [index, posIndex] + negIndices.tolist() + negIndices2

    def __len__(self):
        return len(self.queries)


def collate_quad_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

    Args:
        data: list of tuple (query, positive, negatives).
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
            - negative2: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None, None, None, None, None

    query, positive, negatives, negatives2, indices = zip(*batch)

    query = data.dataloader.default_collate(query)                                                 # ([8, 3, 200, 200]) = [(3, 200, 200), (3, 200, 200), ..  ]     ([8, 1, 3, 200, 200])
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)                                                            # ([80, 3, 200, 200]) ([80, 1, 3, 200, 200])
    negatives2 = torch.cat(negatives2, 0)                                                          # ([80, 3, 200, 200]) ([80, 1, 3, 200, 200])
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negatives2, negCounts, indices


if __name__ == '__main__':
    structFile="E:/shujuji/Cambridge_train1_4/train"
    dbStruct = parse_dbStruct(structFile)