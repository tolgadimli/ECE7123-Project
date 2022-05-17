import numpy as np
from utils import *
from sklearn.preprocessing import normalize
from scipy import interpolate
import pickle

suffixes = ['elmo', 'transformer', 'unirep']

dataset_dict = {}
files = ['ordered_rh_train_', 'ordered_rh_valid_', 'ordered_rh_test_fold_',
            'ordered_rh_test_family_', 'ordered_rh_test_superfamily_']

keys = ['train', 'valid', 'test_fold', 'test_family', 'test_superfamily']

for k, f in zip(keys, files):
    x_avg_ls = []
    x_lens = []
    for suffix in suffixes:
        f_tmp = f + suffix
        x, y = get_embd_and_label(f_tmp)
        x = normalize(x, norm = 'l2', axis = 1)

        x_lens.append(x.shape[1])
        x_avg_ls.append(x)
    max_len = max(x_lens)
    exclude_i = x_lens.index(max_len)

    #interpolation
    x_inter_ls = [] 
    for i in range(len(x_avg_ls)):
        if i != exclude_i:
            arr_tmp = x_avg_ls[i]
            arr_inter = np.zeros([arr_tmp.shape[0], max_len])

            t = np.linspace(0, max_len, arr_tmp.shape[1])
            #t_new = np.arange(0, max_len)
            t_new = np.linspace(0, max_len, max_len)

            for j in range(arr_tmp.shape[0]):
                row = arr_tmp[j,:]
                f = interpolate.interp1d(t, row)
                new_row = f(t_new)
                arr_inter[j,:] = new_row

            x_inter_ls.append(arr_inter)
        else:
            arr_tmp = x_avg_ls[i]
            x_inter_ls.append(arr_tmp)

    #x_avg = sum(x_inter_ls)/len(x_inter_ls)
    dataset_dict[k] = (x_inter_ls, y)


with open('interpolated', 'wb') as fp:
    pickle.dump(dataset_dict, fp)
