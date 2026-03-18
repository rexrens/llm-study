from cnn import NetCNN
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset

def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("using {} device.".format(devi'
    # ce))
    device = torch.device("cuda:0")

    # data load and preprocess
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1,1]
    ])

    # load MNIST data set
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    # 获取一半的数据量
    half_size = len(train_dataset) // 2
    # 随机选择一半的索引
    indices = torch.randperm(len(train_dataset))[:half_size]
    # 创建子集
    half_train_dataset = Subset(train_dataset, indices)

    train_loader = torch.utils.data.DataLoader(half_train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = NetCNN()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    num_epochs = 5

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}] | Loss: {total_loss / len(train_loader):.4f}")

    # validate
    model.eval()
    correct = 0  # accumulate accurate number / epoch
    total = 0

    with torch.no_grad():
       for images, labels in test_loader:
           outputs = model(images.to(device))
           _, predicted = torch.max(outputs, 1)
           total += labels.size(0)
           correct += (predicted == labels.to(device)).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Finished Training, Test Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), './cNN_model.pth')

if __name__ == '__main__':
    main()