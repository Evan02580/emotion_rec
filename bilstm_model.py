import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('./train/word_vector/kaggle_train_data_vector.pkl', 'rb') as f:
    contents = pickle.load(f)

with open('./train/word_vector/kaggle_train_data_label.pkl', 'rb') as f:
    labels = pickle.load(f)


# 定义自定义数据集
class SentimentDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)  # 将数据转换为张量类型
        self.labels = torch.tensor(labels, dtype=torch.long)  # 将标签转换为张量类型

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 定义 BiLSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # 双全连接层
        # self.fnn = nn.Linear(hidden_size * 2, hidden_size)
        # self.relu = nn.ReLU()
        # self.classifier = nn.Linear(hidden_size, output_size)

        # 单全连接层
        self.classifier = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # out = self.relu(self.fnn(out))  # 双全连接层
        out = self.classifier(out)
        return out


# 定义训练函数
def train(model, train_loader, criterion, optimizer, epochs, test_loader):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            model = model.to(device)
            output = model(data)

            # 计算loss
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 计算准确率
        accuracy = test(test_loader)
        # 计算平均loss
        avg_loss = total_loss / len(train_loader)

        print(f'Epoch {epoch + 1}, Loss: {avg_loss}, Accuracy: {accuracy}')

    torch.save(model, './nn_model/bilstm_model_{}.pth'.format("3_1"))
    # torch.save(model, './nn_model/lstm_model_{}.pth'.format("1"))
    print("模型训练完成~")


def test(test_loader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)  # 将数据移动到设备上
            output = model(data)
            _, predicted = torch.max(output, 1)  # 获取预测的类别
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy


# 将数据分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(contents, labels, test_size=0.2, random_state=22)

# 定义数据集
train_dataset = SentimentDataset(x_train, y_train)
test_dataset = SentimentDataset(x_test, y_test)

batch_size = 256

# 加载到dataloader类
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
data_loader = DataLoader(SentimentDataset(contents, labels), batch_size=batch_size, shuffle=True)

# 定义模型参数
input_size = len(contents[0])
hidden_size = 128
num_layers = 2
output_size = 3  # 积极、消极、中性
num_epochs = 100
learning_rate = 0.01

# 初始化模型、损失函数和优化器
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
# train = False
# if train == True:
# train(model, data_loader, criterion, optimizer, num_epochs, test_loader)
