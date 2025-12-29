import torch
import torch.nn as nn


# 定义编码器类，继承自nn.Module
class Encoder(nn.Module):
    def __init__(self, dims):
        super(Encoder, self).__init__()  # 调用父类的初始化方法
        self.dims = dims  # 存储输入的维度列表
        models = []  # 用于存储模型层的列表
        for i in range(len(self.dims) - 1):  # 遍历维度列表以构建网络层
            models.append(nn.Linear(self.dims[i], self.dims[i + 1]))  # 添加全连接层
            if i != len(self.dims) - 2:  # 如果不是最后一层
                models.append(nn.ReLU())  # 添加ReLU激活函数
            else:
                models.append(nn.Dropout(p=0.5))  # 在最后一层之前添加Dropout层
                models.append(nn.Softmax(dim=1))  # 在最后一层添加Softmax激活函数
        self.models = nn.Sequential(*models)  # 将层列表转换为顺序模型

    def forward(self, X):
        return self.models(X)  # 定义前向传播，直接调用顺序模型


# 定义解码器类，继承自nn.Module
class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()  # 调用父类的初始化方法
        self.dims = dims  # 存储输入的维度列表
        models = []  # 用于存储模型层的列表
        for i in range(len(self.dims) - 1):  # 遍历维度列表以构建网络层
            models.append(nn.Linear(self.dims[i], self.dims[i + 1]))  # 添加全连接层
            if i == len(self.dims) - 2:  # 如果是最后一层
                models.append(nn.Dropout(p=0.5))  # 在最后一层之前添加Dropout层
                models.append(nn.Sigmoid())  # 添加Sigmoid激活函数
            else:
                models.append(nn.ReLU())  # 对于其他层，添加ReLU激活函数
        self.models = nn.Sequential(*models)  # 将层列表转换为顺序模型

    def forward(self, X):
        return self.models(X)  # 定义前向传播，直接调用顺序模型


# 定义判别器类，继承自nn.Module
class Discriminator(nn.Module):
    def __init__(self, input_dim, feature_dim=64):
        super(Discriminator, self).__init__()  # 调用父类的初始化方法
        self.input_dim = input_dim  # 输入维度
        self.feature_dim = feature_dim  # 特征维度
        self.discriminator = nn.Sequential(
            nn.Linear(self.input_dim, self.feature_dim),  # 全连接层，将输入映射到特征空间
            nn.LeakyReLU(),  # LeakyReLU激活函数
            nn.Linear(self.feature_dim, 1),  # 全连接层，将特征映射到单个输出
            nn.Sigmoid()  # Sigmoid激活函数，用于二分类
        )

    def forward(self, x):
        return self.discriminator(x)  # 定义前向传播，直接调用顺序模型


# 定义判别器损失函数
def discriminator_loss(real_out, fake_out, lambda_dis=1):
    real_loss = nn.BCEWithLogitsLoss()(real_out, torch.ones_like(real_out))  # 计算真实样本的二元交叉熵损失
    fake_loss = nn.BCEWithLogitsLoss()(fake_out, torch.zeros_like(fake_out))  # 计算假样本的二元交叉熵损失
    return lambda_dis * (real_loss + fake_loss)  # 返回加权的总损失


# 定义多视角自编码器模型类，继承自nn.Module
class MvAEModel(nn.Module):
    def __init__(self, input_dims, view_num, out_dims, h_dims):
        super().__init__()  # 调用父类的初始化方法
        self.input_dims = input_dims  # 输入维度列表
        self.view_num = view_num  # 视角数量
        self.out_dims = out_dims  # 输出维度
        self.h_dims = h_dims  # 隐藏层维度
        self.discriminators = nn.ModuleList()  # 用于存储判别器的模块列表
        for v in range(view_num):
            self.discriminators.append((Discriminator(out_dims)))  # 为每个视角创建一个判别器
        h_dims_reverse = list(reversed(h_dims))  # 反转隐藏层维度，用于解码器
        self.encoders_specific = nn.ModuleList()  # 用于存储特定编码器的模块列表
        self.decoders_specific = nn.ModuleList()  # 用于存储特定解码器的模块列表
        for v in range(self.view_num):
            # 为每个视角创建一个特定编码器
            self.encoders_specific.append(Encoder([input_dims[v]] + h_dims + [out_dims]))
            # 为每个视角创建一个特定解码器
            self.decoders_specific.append(Decoder([out_dims * 2] + h_dims_reverse + [input_dims[v]]))
        d_sum = 0  # 计算所有输入维度的总和
        for d in input_dims:
            d_sum += d
        # 创建共享编码器，输入为所有视角的拼接，输出为共享表示
        self.encoder_share = Encoder([d_sum] + h_dims + [out_dims])
        # TODO 添加分类头
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
        # LayerNorm
        self.block_norm = nn.LayerNorm(out_dims)

    # 定义判别器损失的计算函数
    def discriminators_loss(self, hidden_specific, i, LAMB_DIS=1):
        discriminate_loss = 0.  # 初始化判别损失
        for j in range(self.view_num):
            if j != i:  # 对于其他视角，计算判别损失
                real_out = self.discriminators[i](hidden_specific[i])  # 对第i个视角的特定表示进行判别
                fake_out = self.discriminators[i](hidden_specific[j])  # 将第j个视角的特定表示作为假样本输入
                discriminate_loss += discriminator_loss(real_out, fake_out, LAMB_DIS)  # 累积损失
        return discriminate_loss  # 返回总的判别损失

    # 定义前向传播函数
    def forward(self, x_list):
        x_total = torch.cat(x_list, dim=-1)  # 将所有视角的输入拼接在一起
        hidden_share = self.encoder_share(x_total)  # 通过共享编码器获取共享表示
        recs = []  # 用于存储重构输出的列表
        hidden_specific = []  # 用于存储特定表示的列表
        for v in range(self.view_num):
            x = x_list[v]  # 获取第v个视角的输入
            hidden_specific_v = self.encoders_specific[v](x)  # 通过特定编码器获取特定表示
            hidden_specific.append(hidden_specific_v)  # 将特定表示添加到列表
            hidden_v = torch.cat((hidden_share, hidden_specific_v), dim=-1)  # 拼接共享表示和特定表示
            rec = self.decoders_specific[v](hidden_v)  # 通过特定解码器获取重构输出
            recs.append(rec)  # 将重构输出添加到列表
            
        hidden_list = [self.block_norm(hidden_share)] + [self.block_norm(h) for h in hidden_specific] # 创建包含共享和特定表示的列表
        hidden = torch.cat(hidden_list, dim=-1) # 拼接所有隐藏表示
        class_output = self.classifier(hidden)
            
        return hidden_share, hidden_specific, hidden, recs  # 返回共享表示、特定表示、拼接表示和重构输出
