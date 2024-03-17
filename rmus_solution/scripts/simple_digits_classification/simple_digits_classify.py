import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# 定义神经网络模型
class CNN_digits(nn.Module):
    def __init__(self, C_in, H_in, W_in, n_classes):
        super(CNN_digits, self).__init__()
        self.C_in = C_in
        self.H_in = H_in
        self.W_in = W_in
        self.n_classes = n_classes
        # self.norm = nn.BatchNorm2d(C_in)
        self.conv1 = nn.Conv2d(C_in, 8, kernel_size=3)
        self.conv1_drop = nn.Dropout2d(0)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        # calc fc1's input size
        tmp = torch.zeros(1, 3, H_in, W_in)
        tmp = F.relu(F.max_pool2d(self.conv1(tmp), 2))
        tmp = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(tmp)), 2))
        tmp = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(tmp)), 2))
        d = tmp.flatten().shape[0]
        # define fully connected layers
        self.fc1 = nn.Linear(d, 128)
        self.fc1_drop = nn.Dropout()
        self.fc2 = nn.Linear(128, 64)
        self.fc2_drop = nn.Dropout()
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        ## x: (batch_size, C_in, H_in, W_in)
        assert x.shape[-3:] == (self.C_in, self.H_in, self.W_in)
        x = torch.Tensor(x).view(-1, self.C_in, self.H_in, self.W_in)
        # note that
        # x = self.norm(x)
        
        # imgs = x.permute(0, 2, 3, 1).detach().numpy()
        # import cv2
        # import numpy as np
        # for img in imgs:
        #     print(np.max(img), np.min(img))
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     cv2.imshow("img", img)
        #     if cv2.waitKey(0) == ord('q'):
        #         break
        
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.flatten(start_dim=-3, end_dim=-1)
        x = self.fc1_drop(F.relu(self.fc1(x)))
        x = self.fc2_drop(F.relu(self.fc2(x)))
        logits = self.fc3(x)
        return logits


# 定义训练循环
def train(model:CNN_digits, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        # print(target)
        # if epoch % 20 == 0:
        #     cv2.imshow("data", data[0].permute(1, 2, 0).numpy())
        #     cv2.waitKey(0)
            
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.mean(F.cross_entropy(output, target))
        loss.backward()
        optimizer.step()
    if epoch % 20 == 0:
        print(f'Train Epoch: {epoch}\tLoss: {loss.item():.6f}')

def rotate_images():
    from PIL import Image
    # 指定图像文件夹路径
    folder_path = 'templates'

    # 递归遍历文件夹中的所有文件
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if not(filename.endswith('.png') or filename.endswith('.jpg')):  # 检查文件是否为图像
                continue
            if filename.find("rotated") != -1:  # 检查文件是否已经旋转过
                continue
            else:
                # 读取图像
                img = Image.open(os.path.join(dirpath, filename))
            
                # 旋转图像并保存
                for angle in [90, 180, 270]:
                    img_rotated = img.rotate(angle)
                    # 生成新的文件名
                    new_filename = os.path.splitext(filename)[0] + '_rotated_' + str(angle) + os.path.splitext(filename)[1]
                    img_rotated.save(os.path.join(dirpath, new_filename))

def main():
    # 定义数据加载器
    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor()
    ])
    train_data = datasets.ImageFolder(root='templates', transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    # 初始化模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_digits(C_in=3, H_in=50, W_in=50, n_classes=9).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    # 开始训练
    if not os.path.exists("model.pth"):
        for epoch in range(1, 1600 + 1):
            train(model, device, train_loader, optimizer, epoch)
            if epoch % 200 == 0:
                torch.save(model.state_dict(), "model.pth")
    
    # 加载模型
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        print(f"Predictions\t: {pred.flatten().tolist()}")
        print(f"Targets    \t: {target.flatten().tolist()}")
        print(f"acc: {torch.sum(pred.flatten() == target.flatten()).item() / len(pred.flatten())}")

if __name__ == '__main__':
    rotate_images()
    main()