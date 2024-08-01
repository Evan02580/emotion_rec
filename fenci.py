import jieba
import jieba.analyse
import os
import csv
from tqdm import tqdm


class Fenci:
    data_path = "./test1/clean_data"
    save_path = "./test1/fenci_data"
    # all_words_path = "./train/word"
    stopwords = []

    def __init__(self):
        # 所有文件名
        files = os.listdir(self.data_path)

        # 对每个文件进行操作
        for file in files:
            # 储存评论数据
            text_list = self.read_file(file)
            # jieba分词
            words_list = self.jieba_fenci(text_list)
            print("分词完成...")
            # 写入csv文件
            self.write_file(file, words_list)

    # 读取用户词典文件、停用词文件、数据文件
    def read_file(self, file):
        # 用户词典读取
        jieba.load_userdict("./jieba/dict.txt")

        # 停用词读取
        with open("./jieba/stopwords_7352.txt", "r", encoding='utf-8') as f:
            self.stopwords = f.read().split("\n")

        # 从文件中读取评论文本
        file_path = f"{self.data_path}/{file}"
        text_list = []
        with open(file_path, "r", encoding='utf-8-sig') as f:
            for row in csv.reader(f):
                text_list.append(row)
        return text_list

    # 使用jieba进行分词，并删除停用词，存入列表
    def jieba_fenci(self, text_list):
        words_list = []
        for text in tqdm(text_list):
            # jieba对一条评论分词
            words = jieba.lcut(text[1])
            # 保留不在停用词中的词
            cut_stop_words = [word for word in words
                              if word not in self.stopwords and len(word) > 1]
            # 将[情感, 分词结果]存入列表
            words_list.append([text[0], cut_stop_words])

        return words_list

    # 存入文件
    def write_file(self, file, words_list):
        # 如果文件夹不存在，则创建
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        file_path = f"{self.save_path}/{file}"
        with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "content"])
            for words in words_list:
                if len(words[1]) > 0:
                    writer.writerow([words[0], " ".join(words[1])])
        print(f"完成文件{file}的写入")

        # file_path = f"{self.all_words_path}/word.txt"
        # with open(file_path, "a", encoding="utf-8") as f:
        #     for words in words_list:
        #         if len(words) > 0:
        #             f.write(" ".join(words))
        #             f.write("\n")


if __name__ == "__main__":
    Fenci()
