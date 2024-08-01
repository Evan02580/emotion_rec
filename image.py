# coding=utf-8
from wordcloud import WordCloud
from collections import Counter
from bilstm_model import LSTMModel
import matplotlib.pyplot as plt
import pandas as pd
import jieba
import collections
import re
from model_predict import predict


def draw_pie(predictions):
    label_map = {0: 'neutral', 1: 'positive', 2: 'negative'}

    # 将标签列表中的类别进行替换
    labels_mapped = [label_map[label] for label in predictions]
    # 统计每个类别的个数
    label_counts = Counter(labels_mapped)

    # 获取类别标签和对应的计数
    categories = label_counts.keys()
    counts = label_counts.values()

    # 创建饼状图
    plt.pie(counts, labels=categories, autopct='%1.1f%%')
    plt.axis('equal')  # 保持饼状图为正圆形
    plt.show()
    return labels_mapped


def statistical_number(labels_mapped):
    positive_list = []
    neutral_list = []
    negative_list = []
    for i, item in enumerate(labels_mapped):
        if item == 'positive':
            positive_list.append(i)
        elif item == 'neutral':
            neutral_list.append(i)
        else:
            negative_list.append(i)
    return positive_list, neutral_list, negative_list


def concate_content(data, positive_list, neutral_list, negative_list):
    contents = data['content']
    positive_content = ''
    for i in positive_list:
        positive_content += str(contents[i])

    neutral_content = ''
    for i in neutral_list:
        neutral_content += str(contents[i])

    negative_content = ''
    for i in negative_list:
        negative_content += str(contents[i])
    return positive_content, neutral_content, negative_content


def draw_word_cloud(content, title):
    content = str(content)
    content = re.sub(r"[^\w\s]", "", content)
    words = jieba.cut(content)
    word_counts = collections.Counter(words)
    top_words = word_counts.most_common(500)
    word_freq = {}
    for word, count in top_words:
        word_freq[word] = count
    wordcloud = WordCloud(font_path='C:\Windows\Fonts\simhei.ttf', width=800, height=400, max_words=100,
                          background_color='white')
    wordcloud.generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.show()


if __name__ == '__main__':
    predictions = predict()

    filename = 'test_data.csv'
    data = pd.read_csv(f'./test1/fenci_data/{filename}')
    labels_mapped = draw_pie(predictions)
    positive_list, neutral_list, negative_list = statistical_number(labels_mapped)
    positive_content, neutral_content, negative_content = concate_content(data, positive_list, neutral_list,
                                                                          negative_list)
    '''
    画词云图，分为三个
    '''
    draw_word_cloud(positive_content, "positive")
    draw_word_cloud(negative_content, "neutral")
    draw_word_cloud(neutral_content, "negative")
