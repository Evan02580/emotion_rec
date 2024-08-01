from gensim.models import Word2Vec


model = Word2Vec.load('./word2vec_model/word2vec.bin')
while True:
    word = input("输入需要查询的词：\n")
    if word == "0":
        break
    for words in model.wv.most_similar(word, topn=10):
        print(words)


# print(nn_model.wv.similarity("肺炎", "疫情"))

