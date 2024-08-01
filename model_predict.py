import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import warnings
import pandas as pd
import csv
import os
from bilstm_model import LSTMModel

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('./nn_model/bilstm_model_1_2.pth', map_location=device)

data_path = "./test1/word_vector"
save_path = "./test1/predict"
filename = "test_data.csv"
with open(f'{data_path}/{filename}_vector.pkl', 'rb') as f:
    contents = pickle.load(f)


class SentimentDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)  # 将数据转换为张量类型

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


dataset = SentimentDataset(contents)
test_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
# print(dataloader)


def predict():
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in test_dataloader:
            data = data.to(device)  # 将数据移动到设备上
            output = model(data)
            _, predicted = torch.max(output, 1)  # 获取预测的类别
            predictions.extend(predicted.cpu().tolist())

    return predictions


if __name__ == '__main__':
    r = predict()

    content = pd.read_csv(f'./test1/fenci_data/{filename}')['content']
    # 如果文件夹不存在，则创建
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(f'{save_path}/{filename}', "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "content"])
        for i in range(len(r)):
            writer.writerow([r[i], content[i]])
    print("完成文件预测的写入")



