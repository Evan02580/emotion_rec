import numpy as np
import pandas as pd
import jieba
from gensim.models import Word2Vec
import pickle
from tqdm import tqdm
import os

word2vec_model = Word2Vec.load("./word2vec_model/word2vec.bin")


def generate_vector_avg(data):
    encoded_contents = []
    for content in tqdm(data['content']):
        # words = jieba.lcut(str(content), cut_all=False)
        words = []
        words.extend(content.split(' '))
        sentences_vector = np.zeros(word2vec_model.vector_size)

        num_words = 0
        for word in words:
            if word in word2vec_model.wv:
                sentences_vector += word2vec_model.wv[word]
                num_words += 1
            else:
                # 如果单词不在词汇表中，随机初始化一个向量
                random_vector = np.random.uniform(-1, 1, word2vec_model.vector_size)
                sentences_vector += random_vector
                num_words += 1
            # sentences_vector = torch.from_numpy(sentences_vector)

        if num_words > 0:
            sentences_vector /= num_words  # 对一个句子的词向量进行平均
        # sentences_vector = torch.from_numpy(sentences_vector)
        encoded_contents.append(sentences_vector)
    print("处理成功")
    return encoded_contents


def generate_vector_plus(data):
    # data = pd.read_csv('./data/nCov_10k_test_result.csv')

    encoded_contents_plus = []
    for content in tqdm(data['content']):
        # words = jieba.lcut(str(content), cut_all=False)
        words = []
        words.extend(content.split(' '))
        sentences_vector = np.zeros((len(words), word2vec_model.vector_size))

        # print(content)
        num_words = 0

        for i, word in enumerate(words):
            if word in word2vec_model.wv:
                sentences_vector[i] = word2vec_model.wv[word]
                num_words += 1
            else:
                # 如果单词不在词汇表中，随机初始化一个向量
                random_vector = np.random.uniform(-1, 1, word2vec_model.vector_size)
                sentences_vector[i] = random_vector
                num_words += 1
            # sentences_vector = torch.from_numpy(sentences_vector)

        # sentences_vector = torch.from_numpy(sentences_vector)
        encoded_contents_plus.append(sentences_vector)
    print("处理成功")
    return encoded_contents_plus


def generate_label(data):
    label_data = []
    for content in tqdm(data['label']):
        # 0：中性 1：积极 2：消极
        if int(content) == -1:
            content = 2
        label_data.append(int(content))
    return label_data


if __name__ == '__main__':
    # with open('./data/test/test_vector.pkl', 'wb') as f:
    #     encoded_contents = generate_vector_one()
    #     pickle.dump(encoded_contents, f)
    data_path = "./test1/fenci_data"
    save_path = "./test1/word_vector"
    files = os.listdir(data_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for file in files:
        data = pd.read_csv(f"{data_path}/{file}")

        file_path = f"{save_path}/{file}_vector.pkl"
        with open(file_path, 'wb') as f:
            # 向量化
            encoded_contents_avg = generate_vector_avg(data)
            pickle.dump(encoded_contents_avg, f)
            print('文本编码成功~')

        # 训练数据需要label
        # with open('./train/word_vector/kaggle_train_data_label.pkl', 'wb') as f:
        #     label_data = generate_label(data)
        #     pickle.dump(label_data, f)
        #     print('标签编码成功~')
