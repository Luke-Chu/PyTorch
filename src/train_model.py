import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Model
import time

# 准备数据集
train_data = torchvision.datasets.CIFAR10("../datasets", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("../datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
train_size = len(train_data)
test_size = len(test_data)
print("训练数据集大小：{}".format(train_size))
print("测试数据集大小：{}".format(test_size))

# 加载数据
train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# 定义训练设备
# device = torch.device("cuda:0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 创建网络模型
model = Model().to(device)
# 使用GPU
# model = model.cuda()

# 损失函数
loss_fun = nn.CrossEntropyLoss().to(device)
# loss_fun = loss_fun.cuda()
# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置网络训练参数
total_train_times = 0  # 总训练次数
total_test_times = 0  # 总测试次数
epoch = 30  # 训练轮数

# 使用tensorboard
writer = SummaryWriter("../model_logs")


for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i + 1))
    # i轮开始训练
    start_time = time.time()
    model.train()  # 在该模型中不是必须
    for data in train_loader:
        images, targets = data
        # images = images.cuda()
        # targets = targets.cuda()
        images = images.to(device)
        targets = targets.to(device)

        out = model(images)
        loss = loss_fun(out, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 打印输出
        total_train_times += 1
        if total_train_times % 100 == 0:
            print("train_times: {}, loss: {}".format(total_train_times, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_times)

    end_time = time.time()
    print("第{}轮训练用时：{}".format(i + 1, end_time - start_time))
    # 测试集评价效果
    # 对于分类问题使用正确率更直观
    total_accuracy = 0
    model.eval()  # 在该模型中不是必须
    with torch.no_grad():
        total_test_loss = 0
        for data in test_loader:
            images, targets = data
            # images = images.cuda()
            # targets = targets.cuda()
            images = images.to(device)
            targets = targets.to(device)

            out = model(images)
            total_test_loss += loss_fun(out, targets).item()
            # 正确率
            accuracy = (out.argmax(1) == targets).sum()
            total_accuracy += accuracy
        # 打印输出
        print("--------第{}轮训练结束，测试集整体误差：{}".format(i + 1, total_test_loss))
        print("--------第{}轮训练结束，测试集整体正确率：{}".format(i + 1, total_accuracy / test_size))
        writer.add_scalar("test_loss", total_test_loss, total_test_times)
        writer.add_scalar("test_accuracy", total_accuracy / test_size, total_test_times)
        total_test_times += 1

    # 每轮保存模型, 参数保存为字典类型
    torch.save(model.state_dict(), "model_{}.pth".format(i + 1))

writer.close()
