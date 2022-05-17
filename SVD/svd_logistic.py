import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import io
import scipy.io
import pandas as pd


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

file_name_labels = '/Users/serhat/Desktop/NYU/Ders/Deep_Learning/Mini_Projects/Mini_Project_2/SVDMeta/SVD Meta Embeddings/labels_all.mat'
labels_file = scipy.io.loadmat(file_name_labels)
train_Y = labels_file['label_train'].reshape(-1)
val_Y = labels_file['label_valid'].reshape(-1)
family_Y = labels_file['label_test_family'].reshape(-1)
superfamily_Y = labels_file['label_test_superfamily'].reshape(-1)
fold_Y = labels_file['label_test_fold'].reshape(-1)
d = list(np.arange(4,15)*250)
pandas_ls = []
for l in d:
  file_name = '/Users/serhat/Desktop/NYU/Ders/Deep_Learning/Mini_Projects/Mini_Project_2/SVDMeta/SVD Meta Embeddings/meta_svd_%d.mat'%l
  print(l)
  mat_file = scipy.io.loadmat(file_name)
  train_X = mat_file['meta_train']
  val_X = mat_file['meta_valid']
  family_X = mat_file['meta_test_family']
  superfamily_X = mat_file['meta_test_superfamily']
  fold_X = mat_file['meta_test_fold']
  scaler = StandardScaler().fit(train_X)
  train_X = scaler.transform(train_X)
  val_X = scaler.transform(val_X)
  fold_X = scaler.transform(fold_X)
  family_X = scaler.transform(family_X)
  superfamily_X = scaler.transform(superfamily_X)
  clf = LogisticRegression(random_state=0, max_iter=5000).fit(train_X, train_Y)
  train_acc = clf.score(train_X, train_Y)
  val_acc = clf.score(val_X, val_Y)
  fold_acc = clf.score(fold_X, fold_Y)
  family_acc = clf.score(family_X, family_Y)
  superfamily_acc = clf.score(superfamily_X, superfamily_Y)
  results = [train_acc, val_acc, fold_acc, family_acc, superfamily_acc] 

  pandas_ls.append(results)
  df = pd.DataFrame(data = pandas_ls, columns = ['train', 'val', 'fold', 'family', 'superfamily'])
  df.to_csv('meta_svd_all.csv')
