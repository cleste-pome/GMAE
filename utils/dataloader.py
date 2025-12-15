import csv
import random
import torch
import sklearn
import scipy

import numpy as np

from torch.utils.data import Dataset
from scipy.io import loadmat


def loadData(data_name):
    data = scipy.io.loadmat(data_name)
    features = data['X']
    gnd = data['Y']
    # 返回一个折叠成一维的数组，只能适用于numpy对象，即array或者mat
    gnd = gnd.flatten()
    return features, gnd


class AnyDataset(Dataset):

    def __init__(self, dataname):
        self.features, self.gnd = loadData('./data/' + dataname + '.mat')
        self.v = self.features.shape[1]
        # 数据归一化
        for i in range(0, self.v):
            minmax = sklearn.preprocessing.MinMaxScaler()
            self.features[0][i] = minmax.fit_transform(self.features[0][i])
        # 单位矩阵
        self.iden = torch.tensor(np.identity(self.features[0][0].shape[0])).float()
        self.dataname = dataname

        self.X = self.features[0]

    def __len__(self):
        return self.gnd.shape[0]

    def __getitem__(self, idx):
        """
        return torch.from_numpy(np.array(self.features[0][:][:,idx])), torch.from_numpy(
            np.array(self.gnd[idx])), torch.from_numpy(np.array(idx))
        """
        if (self.v == 2):
            return list([torch.from_numpy(np.array(self.features[0][0][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][1][idx], dtype=np.float32))]), torch.from_numpy(
                np.array(self.gnd[idx])), \
                torch.from_numpy(np.array(idx)), torch.from_numpy(np.array(self.iden[idx]))
        if (self.v == 3):
            return list([torch.from_numpy(np.array(self.features[0][0][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][1][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][2][idx], dtype=np.float32))]), torch.from_numpy(
                np.array(self.gnd[idx])), torch.from_numpy(np.array(idx)), torch.from_numpy(np.array(self.iden[idx]))
        if (self.v == 4):
            return list([torch.from_numpy(np.array(self.features[0][0][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][1][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][2][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][3][idx], dtype=np.float32))]), torch.from_numpy(
                np.array(self.gnd[idx])), torch.from_numpy(np.array(idx)), torch.from_numpy(np.array(self.iden[idx]))
        if (self.v == 5):
            return list([torch.from_numpy(np.array(self.features[0][0][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][1][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][2][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][3][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][4][idx], dtype=np.float32))]), torch.from_numpy(
                np.array(self.gnd[idx])), torch.from_numpy(np.array(idx)), torch.from_numpy(np.array(self.iden[idx]))
        if (self.v == 6):
            return list([torch.from_numpy(np.array(self.features[0][0][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][1][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][2][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][3][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][4][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][5][idx], dtype=np.float32))]), torch.from_numpy(
                np.array(self.gnd[idx])), torch.from_numpy(np.array(idx)), torch.from_numpy(np.array(self.iden[idx]))
        if (self.v == 12):
            return list([torch.from_numpy(np.array(self.features[0][0][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][1][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][2][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][3][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][4][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][5][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][6][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][7][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][8][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][9][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][10][idx], dtype=np.float32)),
                         torch.from_numpy(np.array(self.features[0][11][idx], dtype=np.float32))]), torch.from_numpy(
                np.array(self.gnd[idx])), torch.from_numpy(np.array(idx)), torch.from_numpy(np.array(self.iden[idx]))

    def addMissing(self, index, ratio):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            # 从视图的总数中随机选择一部分视图
            elements = list(range(self.v))  # 生成一个包含0到self.v-1的列表
            random.seed()  # 确保每次运行时生成不同的随机数
            length = random.randint(1, self.v)  # views数量为随机选取的该列表的子集长度
            views = random.sample(elements, length)  # 从该列表中随机选取views个不重复的元素
            print(f'add missing[{i}]: {views}')
            for v in views:
                self.X[v][i] = 0
        print(f'1. Add Missing completed[ratio: {ratio}]')
        pass

    def addConflict(self, index, ratio):
        Y = self.gnd
        Y = np.squeeze(Y)
        if np.min(Y) == 1:
            Y = Y - 1
        Y = Y.astype(dtype=np.int64)
        num_classes = len(np.unique(Y))

        # 初始化一个字典来记录每个类别的某个代表性数据的视图值
        records = dict()
        # 遍历每个类别
        for c in range(num_classes):
            # 找到类别为c的第一个数据的索引
            i = np.where(Y == c)[0][0]
            # 初始化一个临时字典来存储当前类别的数据的各视图值
            temp = dict()
            # 遍历所有视图
            for v in range(self.v):
                # 记录当前视图下，当前类别的第一个数据的值
                temp[v] = self.X[v][i]
            # 将当前类别的数据视图值存储到records字典
            records[c] = temp
        # 随机选择一部分数据索引用于添加冲突，选择的数量由比例ratio和索引总数决定
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        # 对每一个被选中添加冲突的数据索引
        for i in selects:
            # 随机选择一个视图
            v = np.random.randint(self.v)
            # 修改当前选择的数据索引i的视图v的值，将其设置为当前数据的类别+1后的类别对应的视图值
            # 这里使用模运算保证类别编号是循环的（即如果当前类别是最后一个类别，+1后变成第一个类别）
            self.X[v][i] = records[(Y[i] + 1) % num_classes][v]
        print(f'2. Add Conflict completed: {ratio}]')
        pass

    def addNoise(self, index, ratio, sigma):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            elements = list(range(self.v))  # 生成一个包含0到self.v-1的列表
            random.seed()  # 确保每次运行时生成不同的随机数
            length = random.randint(1, self.v)  # views数量为随机选取的该列表的子集长度
            views = random.sample(elements, length)  # 从该列表中随机选取views个不重复的元素
            print(f'add noise[{i}]: {views}')
            for v in views:
                self.X[v][i] = np.random.normal(self.X[v][i], sigma)
        print(f'3. Add Noise completed: {ratio}]')
        pass


# TODO 1.数据集加载 Dataloader
def dataset_with_info(dataname, FileDatasetInfo):
    features, gnd = loadData('./data/' + dataname + '.mat')
    views = max(features.shape[0], features.shape[1])
    input_num = features[0][0].shape[0]
    datasetforuse = AnyDataset(dataname)
    nc = len(np.unique(gnd))
    input_dims = []
    for v in range(views):
        dim = features[0][v].shape[1]
        input_dims.append(dim)
    print("Data: " + dataname + ", number of data: " + str(input_num) + ", views: " + str(views) + ", clusters: " +
          str(nc) + ", each view: ", input_dims)

    # TODO 3.保存数据集信息
    with open(FileDatasetInfo, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入数据
        data = [dataname, str(input_num), str(views), str(nc), input_dims]
        writer.writerow(data)

    return datasetforuse, input_num, views, nc, input_dims, gnd
