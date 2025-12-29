import torch
import torch.nn as nn


# 编码器（Encoder）类
class Encoder(nn.Module):
    def __init__(self, dims):
        super(Encoder, self).__init__()
        self.dims = dims  # 输入维度列表
        models = []
        # 1. 构建编码器的各层
        for i in range(len(self.dims) - 1):
            models.append(nn.Linear(self.dims[i], self.dims[i + 1]))  # 添加全连接层
            # 2. 对中间层应用 ReLU 激活函数
            if i != len(self.dims) - 2:
                models.append(nn.ReLU())
            # 3. 最后一层使用 Dropout 防止过拟合
            else:
                models.append(nn.Dropout(p=0.5))
        # 4. 使用 Sequential 将所有层顺序组合
        self.models = nn.Sequential(*models)

    def forward(self, X):
        # 前向传播：通过顺序模型传递数据
        return self.models(X)


# 解码器（Decoder）类
class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        self.dims = dims  # 输入维度列表
        models = []
        # 1. 构建解码器的各层
        for i in range(len(self.dims) - 1):
            models.append(nn.Linear(self.dims[i], self.dims[i + 1]))  # 添加全连接层
            # 2. 对中间层应用 ReLU 激活函数
            if i == len(self.dims) - 2:
                models.append(nn.Dropout(p=0.5))  # 最后一层 Dropout
                models.append(nn.Sigmoid())  # 最后一层使用 Sigmoid 激活
            else:
                models.append(nn.ReLU())
        # 3. 使用 Sequential 将所有层顺序组合
        self.models = nn.Sequential(*models)

    def forward(self, X):
        # 前向传播：通过顺序模型传递数据
        return self.models(X)


# 判别器（Discriminator）类
class Discriminator(nn.Module):
    def __init__(self, input_dim, feature_dim=64):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim  # 输入维度
        self.feature_dim = feature_dim  # 特征维度
        # 1. 定义判别器的网络结构
        self.discriminator = nn.Sequential(
            nn.Linear(self.input_dim, self.feature_dim),  # 输入层到隐藏层
            nn.LeakyReLU(),  # 激活函数
            nn.Linear(self.feature_dim, 1),  # 输出层
            nn.Sigmoid()  # 输出概率
        )

    def forward(self, x):
        # 2. 前向传播
        return self.discriminator(x)


# 计算判别器的损失函数
def discriminator_loss(real_out, fake_out, lambda_dis=1):
    # 1. 计算真实样本损失，目标是 1
    real_loss = nn.BCEWithLogitsLoss()(real_out, torch.ones_like(real_out))
    # 2. 计算假样本损失，目标是 0
    fake_loss = nn.BCEWithLogitsLoss()(fake_out, torch.zeros_like(fake_out))
    # 3. 返回总损失，乘以超参数 lambda_dis
    return lambda_dis * (real_loss + fake_loss)


# MvAE 模型类
class GMAEModel(nn.Module):
    def __init__(self, input_dims, view_num, out_dims, h_dims, num_classes):
        super().__init__()
        # 1. 初始化各类参数
        self.input_dims = input_dims
        self.view_num = view_num
        self.out_dims = out_dims
        self.h_dims = h_dims
        self.num_classes = num_classes

        # 2. 创建判别器：每个视图一个
        self.discriminators = nn.ModuleList([Discriminator(out_dims) for _ in range(view_num)])

        # 3. 创建特定视图的编码器和解码器
        h_dims_reverse = list(reversed(h_dims))
        self.encoders_specific = nn.ModuleList([Encoder([input_dims[v]] + h_dims + [out_dims]) for v in range(view_num)])
        self.decoders_specific = nn.ModuleList([Decoder([out_dims * 2] + h_dims_reverse + [input_dims[v]]) for v in range(view_num)])

        # 4. 创建共享编码器：输入为所有视图特征拼接
        self.encoder_share = Encoder([sum(input_dims)] + h_dims + [out_dims])

        # 5. 创建分类头：输出类别数
        hidden_dim = out_dims * (view_num + 1)
        mid = min(256, hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mid, num_classes),
        )

        # 6. 创建 LayerNorm 层
        self.block_norm = nn.LayerNorm(out_dims)

    # 计算判别器的损失函数
    def discriminators_loss(self, hidden_specific, i, lambda_dis=1):
        discriminate_loss = 0.
        # 1. 遍历每个视图，计算其损失
        for j in range(self.view_num):
            if j != i:
                real_out = self.discriminators[i](hidden_specific[i])  # 当前视图判别器预测真实输出
                fake_out = self.discriminators[i](hidden_specific[j])  # 其他视图的判别器预测假输出
                discriminate_loss += discriminator_loss(real_out, fake_out, lambda_dis)
        return discriminate_loss

    def forward(self, x_list):
        # 1. 拼接所有视图特征
        x_total = torch.cat(x_list, dim=-1)
        # 2. 计算共享潜表示
        hidden_share = self.encoder_share(x_total)
        recs = []  # 存储每个视图的重构结果
        hidden_specific = []  # 存储每个视图的特定潜表示

        # 3. 逐视图进行编码和解码
        for v in range(self.view_num):
            z_v = self.encoders_specific[v](x_list[v])  # 获取当前视图的潜表示
            hidden_specific.append(z_v)
            z_cat = torch.cat([hidden_share, z_v], dim=-1)  # 拼接共享潜表示和视图特定潜表示
            recs.append(self.decoders_specific[v](z_cat))  # 解码得到重构结果

        # 4. 对共享表示和每个视图的特定表示进行归一化
        hidden_list = [self.block_norm(hidden_share)] + [self.block_norm(h) for h in hidden_specific]
        hidden = torch.cat(hidden_list, dim=-1)  # 拼接所有表示

        # 5. 使用分类器进行分类
        class_output = self.classifier(hidden)

        return hidden_share, hidden_specific, hidden, recs, class_output