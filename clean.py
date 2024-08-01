import csv
import os
import re


class Clean:
    data_path = "./test1/weibo_data"
    save_path = "./test1/clean_data"

    def __init__(self):
        # 所有文件名
        files = os.listdir(self.data_path)

        # 对每个文件读取评论文本并清洗
        for file in files:
            text_list = self.read_file(file)
            self.write_file(file, text_list)

    # 读取文件
    def read_file(self, file):
        file_path = f"{self.data_path}/{file}"
        text_list = []  # [[情感, 文本],[],...]
        # 读取文件
        with open(file_path, "r", encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                row[1] = re.findall('[\u4e00-\u9fa5]|["，","。","！","？","、"," "]+', row[1], re.S)
                row[1] = "".join(row[1])
                row[1] = re.sub("@[\u4e00-\u9fa5a-zA-Z0-9_-]{2,30}:", "", row[1])
                row[1] = re.sub("#[^#]+#", "", row[1])
                row[1] = re.sub("""[ ,/,",',=,.,\],\[,\-,_,;,:,?,%,&,+]""", "", row[1])
                text_list.append(row)
        return text_list

    # 写入文件
    def write_file(self, file, text_list):
        # 如果文件夹不存在，则创建
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        file_path = f"{self.save_path}/{file}"
        # 写入清洗后的数据
        with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            for text in text_list:
                if len(text[1]) > 1:
                    writer.writerow(text)


if __name__ == "__main__":
    Clean()
