CV 大作业：Fashion-mnist分类任务
===============================


# 1 赛题描述


[经典的MNIST数据集](http://yann.lecun.com/exdb/mnist/)包含了大量的手写数字。十几年来，来自机器学习、机器视觉、人工智能、深度学习领域的研究员们把这个数据集作为衡量算法的基准之一。你会在很多的会议，期刊的论文中发现这个数据集的身影。实际上，MNIST数据集已经成为算法作者的必测的数据集之一。有人曾调侃道："如果一个算法在MNIST不work, 那么它就根本没法用；而如果它在MNIST上work, 它在其他数据上也可能不work！"
 

`Fashion-MNIST`的目的是要成为MNIST数据集的一个直接替代品。作为算法作者，你不需要修改任何的代码，就可以直接使用这个数据集。`Fashion-MNIST`的图片大小，训练、测试样本数及类别数与经典MNIST**完全相同**。

这个数据集的样子大致如下（每个类别占三行）：

![](https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png)


## 1.1 类别标注

在Fashion-mnist数据集中，每个训练样本都按照以下类别进行了标注：

| 标注编号 | 描述 |
| --- | --- |
| 0 | T-shirt/top（T恤）|
| 1 | Trouser（裤子）|
| 2 | Pullover（套衫）|
| 3 | Dress（裙子）|
| 4 | Coat（外套）|
| 5 | Sandal（凉鞋）|
| 6 | Shirt（汗衫）|
| 7 | Sneaker（运动鞋）|
| 8 | Bag（包）|
| 9 | Ankle boot（踝靴）|


## 1.2 任务描述


`Fashion-MNIST`是一个替代[MNIST手写数字集](http://yann.lecun.com/exdb/mnist/)的图像数据集。 它是由Zalando（一家德国的时尚科技公司）旗下的[研究部门](https://research.zalando.com/)提供。其涵盖了来自10种类别的共7万个不同商品的正面图片。Fashion-MNIST的大小、格式和训练集/测试集划分与原始的MNIST完全一致。60000/10000的训练测试数据划分，28x28的灰度图片。你可以直接用它来测试你的机器学习和深度学习算法性能，且**不需要**改动任何的代码。

本次任务需要针对`Fashion-MNIST`数据集，设计、搭建、训练机器学习模型，能够尽可能准确地分辨出测试数据地标签。


# 2 代码
```python
# -*- coding: utf-8 -*-
import os
import sys
import time
import math
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms


os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # TODO:


class GlobalAvgPool2d(nn.Module):
    """
    全局平均池化层
    可通过将普通的平均池化的窗口形状设置成输入的高和宽实现
    """
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class Residual(nn.Module): 
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        """
            use_1×1conv: 是否使用额外的1x1卷积层来修改通道数
            stride: 卷积层的步幅, resnet使用步长为2的卷积来替代pooling的作用，是个很赞的idea
        """
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    '''
    resnet block

    num_residuals: 当前block包含多少个残差块
    first_block: 是否为第一个block

    一个resnet block由num_residuals个残差块组成
    其中第一个残差块起到了通道数的转换和pooling的作用
    后面的若干残差块就是完成正常的特征提取
    '''
    if first_block:
        assert in_channels == out_channels # 第一个模块的输出通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


# 定义resnet模型结构
net = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),   # TODO: 缩小感受野, 缩channel
        nn.BatchNorm2d(32),
        nn.ReLU())
        #nn.ReLU(),
        #nn.MaxPool2d(kernel_size=2, stride=2))   # TODO：去掉maxpool缩小感受野

# 然后是连续4个block
net.add_module("resnet_block1", resnet_block(32, 32, 2, first_block=True))   # TODO: channel统一减半
net.add_module("resnet_block2", resnet_block(32, 64, 2))
net.add_module("resnet_block3", resnet_block(64, 128, 2))
net.add_module("resnet_block4", resnet_block(128, 256, 2))
# global average pooling
net.add_module("global_avg_pool", GlobalAvgPool2d()) 
# fc layer
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(256, 10)))


print('打印网络结构(主要是为了确认如何调整)')
print(net)


print('打印 1*1*28*28 输入经过每个模块后的shape')
X = torch.rand((1, 1, 28, 28))
for name, layer in net.named_children():
    X = layer(X)
    print(name, ' output shape:\t', X.shape)


# 定义加载数据集的函数
def load_data_fashion_mnist(batch_size, root='/home/kesci/work/dataset', use_normalize=False, mean=None, std=None):
    """Download the fashion mnist dataset and then load into memory."""

    if use_normalize:
        normalize = transforms.Normalize(mean=[mean], std=[std])
        train_augs = transforms.Compose([transforms.RandomCrop(28, padding=2),
                    transforms.RandomHorizontalFlip(), # 数据增强
                    # transforms.RandomVerticalFlip
                    # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
                    transforms.ToTensor(), 
                    normalize])
        test_augs = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        train_augs = transforms.Compose([transforms.ToTensor()])
        test_augs = transforms.Compose([transforms.ToTensor()])
    
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=train_augs)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=test_augs)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter



print('计算数据集均值标准差')
batch_size = 64  
train_iter, test_iter = load_data_fashion_mnist(batch_size, root='/home/kesci/work/dataset', use_normalize=False)
# 求整个数据集的均值
temp_sum = 0
cnt = 0
for X, y in train_iter:
    if y.shape[0] != batch_size:
        break   # 最后一个batch不足batch_size,这里就忽略了
    channel_mean = torch.mean(X, dim=(0,2,3))  # 按channel求均值(不过这里只有1个channel)
    cnt += 1   # cnt记录的是batch的个数，不是图像
    temp_sum += channel_mean[0].item()
dataset_global_mean = temp_sum / cnt
print('整个数据集的像素均值:{}'.format(dataset_global_mean))
# 求整个数据集的标准差
cnt = 0
temp_sum = 0
for X, y in train_iter:
    if y.shape[0] != batch_size:
        break   # 最后一个batch不足batch_size,这里就忽略了
    residual = (X - dataset_global_mean) ** 2
    channel_var_mean = torch.mean(residual, dim=(0,2,3))  
    cnt += 1   # cnt记录的是batch的个数，不是图像
    temp_sum += math.sqrt(channel_var_mean[0].item())
dataset_global_std = temp_sum / cnt
print('整个数据集的像素标准差:{}'.format(dataset_global_std))


# 重新获取应用了归一化的数据集迭代器
batch_size = 64  
train_iter, test_iter = load_data_fashion_mnist(batch_size, root='/home/kesci/work/dataset', use_normalize=True,
                        mean = dataset_global_mean, std = dataset_global_std)


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    net.eval() 
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
    net.train() # 改回训练模式
    return acc_sum / n


def train_model(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs, lr, lr_period, lr_decay):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    best_test_acc = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        if epoch > 0 and epoch % lr_period == 0:  # 每lr_period个epoch，学习率衰减一次
            lr = lr * lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.4f, test acc %.4f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        if test_acc > best_test_acc:
            print('find best! save at model/best.pth')
            best_test_acc = test_acc
            torch.save(net.state_dict(), 'model/best.pth')


print('训练...')
num_epochs = 70
lr = 0.01
lr_period = 40
lr_decay = 0.1
#optimizer = optim.Adam(net.parameters(), lr=lr)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)   # TODO:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_model(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs, lr, lr_period, lr_decay)


print('加载最优模型')
net.load_state_dict(torch.load('model/best.pth'))
net = net.to(device)

print('inference测试集')
net.eval() 
id = 0
preds_list = []
with torch.no_grad():
    for X, y in test_iter:
        batch_pred = list(net(X.to(device)).argmax(dim=1).cpu().numpy())
        for y_pred in batch_pred:
            preds_list.append((id, y_pred))
            id += 1

print('生成提交结果文件')
with open('submission.csv', 'w') as f:
    f.write('ID,Prediction\n')
    for id, pred in preds_list:
        f.write('{},{}\n'.format(id, pred))
```

代码参照了助教提供的baseline，主要是修改了数据增强、Epoch、学习率和优化器，最终成绩为0.94450。第一次用pytorch，也是第一次玩深度学习和CV，学习到很多，也有很多不足，后面继续加油！

## 参考资源

[1] Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747  
[2] https://github.com/zalandoresearch/fashion-mnist/  
[3] https://github.com/monkeyDemon/Learn_Dive-into-DL-PyTorch/tree/master/assignment1_FashionMNIST_Classification
