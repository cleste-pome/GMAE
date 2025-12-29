import argparse
import csv
import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from loss import orthogonal_loss, contrastive_loss
from models import GMAEModel
from utils import Logger
from utils.dataloader import dataset_with_info
from utils.metric import compute_metric
from utils.plot import plot_acc, print_metrics_table


def seed_setting(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # 禁用 cudnn 优化
    torch.backends.cudnn.deterministic = True  # 保证每次执行相同的结果
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':

    # ===================================================================
    # 使用argparse解析命令行超参数
    # ===================================================================
    parser = argparse.ArgumentParser(description='GMAE Model Training')
    parser.add_argument('--log_path', default='1.logs_clustering', type=str, help='Path to save logs')
    parser.add_argument('--img_path', default='2.imgs_clustering', type=str, help='Path to save imgs')
    parser.add_argument('--folder_path', default='dataset', type=str, help='Dataset folder path')
    parser.add_argument('--do_plot', default=True, type=bool, help='Whether to plot the results')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use for training')
    # TODO 1.超参数
    parser.add_argument('--train_epoch', default=500, type=int, help='Number of training epochs') # 500
    parser.add_argument('--eval_interval', default=10, type=int, help='Interval for evaluation')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dimensions')
    parser.add_argument('--lambda_ma', default=0.01, type=float, help='Lambda for mutual alignment loss')
    parser.add_argument('--lambda_con', default=0.01, type=float, help='Lambda for contrastive loss')
    parser.add_argument('--pos_num', default=21, type=int, help='Positive sample number')
    parser.add_argument('--do_contrast', default=True, type=bool, help='Whether to use contrastive loss')
    # TODO 2.数据处理
    parser.add_argument('--ratio_noise', default=0.0, type=float, help='Noise ratio')
    parser.add_argument('--ratio_conflict', default=0.0, type=float, help='Conflict ratio')
    parser.add_argument('--missing_ratio', default=0.0, type=float, help='Missing ratio')

    args = parser.parse_args()  # 解析参数
    seed_setting(args.seed)  # 设置随机种子

    # ===================================================================
    # 创建日志目录
    # ===================================================================
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
        print("Logs directory created at:", args.log_path)
    else:
        print("Logs directory already exists at:", args.log_path)

    # ===================================================================
    # 创建并写入数据集信息的CSV文件
    # ===================================================================
    file_datasetInfo = f'{args.log_path}/datasetInfo.csv'
    headers = ['Dataname', 'number of data', 'views', 'clusters', 'each view']
    with open(file_datasetInfo, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    print(f"{file_datasetInfo} has been created")

    # ===================================================================
    # 遍历数据集文件夹并加载数据
    # ===================================================================
    file_names = os.listdir(args.folder_path)
    data_iter = 1 # 数据集位次
    for dataset_name in file_names:
        if dataset_name.endswith(".mat"):
            dataset_name = dataset_name[:-4]  # 去掉文件名后缀，得到数据集名称
            print(f'----------------------------------{dataset_name}[{data_iter}]----------------------------------')

            # 初始化日志记录器
            logger = Logger.get_logger(__file__, dataset_name, args.log_path)

            # 获取数据集信息
            dataset, ins_num, view_num, nc, input_dims, _ = dataset_with_info(
                dataset_name, file_datasetInfo, args.folder_path)

            # ===================================================================
            # 对数据集进行数据增强处理
            # ===================================================================
            index = np.arange(len(dataset))
            dataset.addMissing(index, args.missing_ratio)  # 添加缺失数据
            dataset.addConflict(index, args.ratio_conflict)  # 添加冲突数据
            dataset.addNoise(index, args.ratio_noise, sigma=0.5)  # 添加噪声

            # 设置DataLoader，提供训练数据和测试数据
            train_loader = DataLoader(dataset, batch_size=ins_num, shuffle=False)
            test_loader = DataLoader(dataset, batch_size=ins_num, shuffle=False)

            # ===================================================================
            # 初始化邻居信息
            # ===================================================================
            neighbors_num = int(ins_num / 4)
            neg_num = int((neighbors_num - args.pos_num - 1) / 2)
            nbr_idx, neg_idx = [], []

            # 计算每个视图的邻居和负样本索引
            for v in range(view_num):
                X_np = np.array(dataset.features[0][v])  # 获取第v个视图的数据
                nbrs_v = np.zeros((ins_num, args.pos_num - 1))  # 初始化正样本邻居数组
                neg_v = np.zeros((ins_num, neg_num))  # 初始化负样本邻居数组
                # GMAE中对比约束的不是不同视图，而是最终表征中的邻居与否。
                nbrs = NearestNeighbors(n_neighbors=neighbors_num, algorithm='auto').fit(X_np)  # 计算邻居
                dis, idx = nbrs.kneighbors(X_np)  # 获取距离和索引
                for i in range(ins_num):
                    for j in range(args.pos_num - 1):
                        nbrs_v[i][j] += idx[i][j + 1]  # 正样本邻居索引
                    for j in range(neg_num):
                        neg_v[i][j] += idx[i][neighbors_num - j - 1]  # 负样本邻居索引
                nbr_idx.append(torch.LongTensor(nbrs_v))  # 将邻居索引转换为张量并添加到列表
                neg_idx.append(torch.LongTensor(neg_v))  # 将负样本索引转换为张量并添加到列表

            nbr_idx = torch.cat(nbr_idx, dim=-1)  # 将所有视图的正样本邻居索引拼接在一起
            neg_idx = torch.cat(neg_idx, dim=-1)  # 将所有视图的负样本邻居索引拼接在一起

            # ===================================================================
            # 选择训练设备（GPU或CPU）
            # ===================================================================
            device = args.device
            h_dims = [500, 200]
            model = GMAEModel(input_dims, view_num, args.feature_dim, h_dims, nc).to(device)
            mse_loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            # ===================================================================
            # 训练过程
            # ===================================================================
            acc_list, nmi_list, pur_list, ari_list = [], [], [], []
            for epoch in tqdm(range(args.train_epoch)):
                criterion = nn.CrossEntropyLoss()
                for x, y, train_idx, pu in train_loader:
                    optimizer.zero_grad()
                    model.train()

                    # 将数据移到设备
                    for v in range(view_num):
                        x[v] = x[v].to(device)

                    hidden_share, hidden_specific, hidden, recs, _ = model(x)
                    loss_rec, loss_mi, loss_ad, loss_class = 0, 0, 0, 0

                    if y.min() == 1:
                        y = (y - 1).long().to(device)
                    elif y.min() == 0:
                        y = y.long().to(device)

                    for v in range(view_num):
                        loss_rec += mse_loss_fn(recs[v], x[v])
                        loss_mi += orthogonal_loss(hidden_share, hidden_specific[v])
                        loss_ad += model.discriminators_loss(hidden_specific, v)

                    # 对比损失
                    loss_con = contrastive_loss(args, hidden, nbr_idx, neg_idx, train_idx)
                    total_loss = loss_rec + args.lambda_ma * (
                                loss_mi + loss_ad) + args.lambda_con * loss_con + loss_class
                    total_loss.backward()
                    optimizer.step()

                # ===================================================================
                # 评估过程
                # ===================================================================
                if (epoch + 1) % args.eval_interval == 0:
                    with torch.no_grad():  # 不需要计算梯度
                        for x, y, idx, pu in train_loader:
                            # 将输入数据移到指定设备
                            for v in range(view_num):
                                x[v] = x[v].to(device)
                            model.eval()  # 设置模型为评估模式

                            # 获取模型输出
                            hidden_share, hidden_specific, hidden, recs, classes = model(x)
                            label = np.array(y)

                            # 使用K-means进行聚类并计算指标
                            y_pred = KMeans(n_clusters=nc, n_init=50).fit_predict(hidden.cpu().numpy())
                            ACC, NMI, PUR, ARI, F_score, Precision, Recall = compute_metric(label, y_pred)

                            # 打包评估结果
                            info = {"epoch": epoch, "acc": ACC, "Nmi": NMI, "ari": ARI, "Purity": PUR,
                                    "Fscore": F_score, "Precision": Precision, "recall": Recall}
                            logger.info(str(info))

                            # 记录指标
                            acc_list.append(ACC)
                            nmi_list.append(NMI)
                            pur_list.append(PUR)
                            ari_list.append(ARI)

            # ===================================================================
            # 绘图
            # ===================================================================
            if args.do_plot:
                plot_acc(acc_list, dataset_name, 'acc', args.imgs_path)
                plot_acc(nmi_list, dataset_name, 'nmi', args.imgs_path)
                plot_acc(pur_list, dataset_name, 'pur', args.imgs_path)
                plot_acc(ari_list, dataset_name, 'ari', args.imgs_path)

        else:
            print(f'Non-MAT file. Please convert the dataset to multi-view one-dimensional MAT format.')
        data_iter += 1