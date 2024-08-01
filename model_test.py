import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import warnings
from bilstm_model import LSTMModel

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('./nn_model/bilstm_model_1_2.pth', map_location=device)
with open('./train/word_vector/kaggle_train_data_vector.pkl', 'rb') as f:
    contents = pickle.load(f)
with open('./train/word_vector/kaggle_train_data_label.pkl', 'rb') as f:
    labels = pickle.load(f)


class SentimentDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)  # 将数据转换为张量类型
        self.labels = torch.tensor(labels, dtype=torch.long)  # 将标签转换为张量类型

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


dataset = SentimentDataset(contents, labels)
test_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
# print(dataloader)


def test():
    model.eval()
    predictions = []
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for data, labels in test_dataloader:
            data, labels = data.to(device), labels.to(device)  # 将数据移动到设备上
            output = model(data)
            _, predicted = torch.max(output, 1)  # 获取预测的类别
            predictions.extend(predicted.cpu().tolist())

            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    return predictions, accuracy


if __name__ == '__main__':
    r, acc = test()
    print(f"Accuracy: {acc}")


