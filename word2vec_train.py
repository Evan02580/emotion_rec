import logging
import multiprocessing
import os.path
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences

if __name__ == '__main__':
    # 日志信息输出
    # program = os.path.basename(sys.argv[0])
    # logger = logging.getLogger(program)
    # logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    # logging.root.setLevel(level=logging.INFO)
    # logger.info("running %s" % ' '.join(sys.argv))

    input_dir = './train/data_for_word2vec'
    outp1 = "./word2vec_model/word2vec.bin"
    outp2 = "./word2vec_model/word2vec.csv"
    fileNames = os.listdir(input_dir)
    # 训练模型
    # 输入语料目录:PathLineSentences(input_dir)
    # embedding size:256，共现窗口大小:8，去除出现次数5以下的词，迭代10次，多线程运行
    model = Word2Vec(PathLineSentences(input_dir),
                     vector_size=256, window=8, min_count=5, epochs=10,
                     workers=multiprocessing.cpu_count())
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
