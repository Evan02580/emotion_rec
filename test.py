# import pandas as pd
import pickle
# import torch
# print(torch.cuda.is_available())
with open('./train/word_vector/kaggle_train_data_label.pkl', 'rb') as f:
    contents = pickle.load(f)
print(len(contents))

with open('./train/word_vector/kaggle_train_data_vector.pkl', 'rb') as f:
    contents = pickle.load(f)
print(len(contents))