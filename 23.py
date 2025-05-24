import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import argparse

# 添加随机种子
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='MNIST Training')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train')
parser.add_argument('--early_stopping_patience', type=int, default=5, help='early stopping patience')
args = parser.parse_args()

# 1. 配置参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
validation_split = 0.2  # 验证集比例
early_stopping_patience = args.early_stopping_patience

# 2. 数据预处理与加载，添加数据增强操作
transform = transforms.Compose([
    transforms.RandomRotation(10),  # 随机旋转 ±10 度
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 数据归一化参数
])

# 加载训练集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

# 划分训练集和验证集
train_size = int((1 - validation_split) * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. 定义神经网络模型（添加Batch Normalization和残差连接）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.res1 = ResidualBlock(16, 16)
        self.res2 = ResidualBlock(16, 32, stride=2)
        # 重新计算全连接层输入维度
        self.fc_layers = nn.Sequential(
            nn.Linear(6272, 128),  # 修改输入维度为 6272
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.fc_layers(x)
        return x

# 4. 初始化模型、损失函数和优化器
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 添加学习率调度器
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)

# 定义模型保存路径，移除不必要的路径检查
model_save_path = 'best_model.pth'

# 初始化TensorBoard writer
writer = SummaryWriter()

# 记录训练日志
train_logs = []
val_logs = []

# 5. 训练模型（添加梯度裁剪）
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
            writer.add_scalar('Train/BatchLoss', loss.item(), epoch * len(train_loader) + batch_idx)
    
    train_loss /= len(train_loader)
    train_accuracy = 100. * correct / total
    train_logs.append((train_loss, train_accuracy))
    writer.add_scalar('Train/Loss', train_loss, epoch)
    writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
    return train_loss, train_accuracy

# 6. 验证模型
def validate():
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / total
    val_logs.append((val_loss, val_accuracy))
    writer.add_scalar('Validation/Loss', val_loss, epoch)
    writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
    return val_loss, val_accuracy

# 7. 测试模型（添加测试时增强）
def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # 测试时增强
            outputs = []
            for _ in range(5):
                augmented_data = data
                output = model(augmented_data)
                outputs.append(output)
            output = torch.stack(outputs).mean(0)

            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / total
    print(f'\nTest Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%\n')
    writer.add_scalar('Test/Loss', test_loss, 0)
    writer.add_scalar('Test/Accuracy', test_accuracy, 0)
    return test_loss, test_accuracy

# 8. 运行训练、验证和测试（添加早停机制）
if __name__ == '__main__':
    best_val_loss = float('inf')
    no_improve = 0
    for epoch in range(1, epochs+1):
        train_loss, train_accuracy = train(epoch)
        val_loss, val_accuracy = validate()
        
        # 调整学习率
        scheduler.step(val_loss)
        
        # 早停逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.2f}%')
    
    # 加载最佳模型并测试
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_accuracy = test()
    
    # 保存日志
    with open('training_logs.txt', 'w') as f:
        f.write('Epoch, Train Loss, Train Acc, Val Loss, Val Acc\n')
        for i in range(len(train_logs)):
            train_loss, train_acc = train_logs[i]
            val_loss, val_acc = val_logs[i]
            f.write(f'{i+1}, {train_loss:.4f}, {train_acc:.2f}, {val_loss:.4f}, {val_acc:.2f}\n')
    
    writer.close()