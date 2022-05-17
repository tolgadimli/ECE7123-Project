import numpy as np
from utils import *
from sklearn.preprocessing import normalize
from scipy import interpolate
import os

"""
Python script that includes different ensembling methods.
"""

"""
indiv - 3
pairwise - 3
concatenate
weighted concat
"""

def get_datasets_dict(suffix):
    dataset_dict = {}
    files = ['ordered_rh_train_', 'ordered_rh_valid_', 'ordered_rh_test_fold_',
              'ordered_rh_test_family_', 'ordered_rh_test_superfamily_']

    keys = ['train', 'valid', 'test_fold', 'test_family', 'test_superfamily']
    for k, f in zip(keys, files):
        f = f + suffix
        a, b = get_embd_and_label(f)
        a = normalize(a, norm = 'l2', axis = 1)
        dataset_dict[k] = (a,b)

    return dataset_dict


def get_concat_datasets_dict( suffixes ):
    dataset_dict = {}
    files = ['ordered_rh_train_', 'ordered_rh_valid_', 'ordered_rh_test_fold_',
              'ordered_rh_test_family_', 'ordered_rh_test_superfamily_']

    keys = ['train', 'valid', 'test_fold', 'test_family', 'test_superfamily']

    for k, f in zip(keys, files):
        x_concat = []
        for suffix in suffixes:
            f_tmp = f + suffix
            x, y = get_embd_and_label(f_tmp)
            x = normalize(x, norm = 'l2', axis = 1)
            x_concat.append(x)
        dataset_dict[k] = (np.concatenate(x_concat, axis = 1), y)

    return dataset_dict


def get_padded_average_dict( suffixes ):
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

        #padding
        x_padded_ls = [] 
        for i in range(len(x_avg_ls)):
            if i != exclude_i:
                arr_tmp = x_avg_ls[i]
                pad = np.zeros([arr_tmp.shape[0],max_len - arr_tmp.shape[1]])
                arr_padded = np.concatenate((arr_tmp, pad), axis = 1)
                x_padded_ls.append(arr_padded)
            else:
                arr_tmp = x_avg_ls[i]
                x_padded_ls.append(arr_tmp)

        x_avg = sum(x_padded_ls)/len(x_padded_ls)
        dataset_dict[k] = (x_avg, y)

    return dataset_dict



def get_interpolated_avg_dict( suffixes ):
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
        x_padded_ls = [] 
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

                x_padded_ls.append(arr_inter)
            else:
                arr_tmp = x_avg_ls[i]
                x_padded_ls.append(arr_tmp)

        x_avg = sum(x_padded_ls)/len(x_padded_ls)
        dataset_dict[k] = (x_avg, y)

    return dataset_dict



def get_dict_from_mat(mat_file, file_dir = 'data/1toN'):
    """Function to get embeddings and labels from the pickle file"""
  

    mat_dir = os.path.join(file_dir, mat_file)
    mat = scipy.io.loadmat(mat_dir)

    mat_dict = {}
    source_keys = ['meta_train', 'meta_valid', 'meta_test_fold', 'meta_test_family', 'meta_test_superfamily']
    target_keys = ['train', 'valid', 'test_fold', 'test_family', 'test_superfamily']
    for tk, sk in zip(target_keys, source_keys):
        mat_dict[tk] = mat[sk]

    reference_dict = get_transformer()
    
    dataset_dict = {}
    for key, value in reference_dict.items():
        embds =   mat_dict[key]
        labs = value[1]
        dataset_dict[key] = (embds, labs)

    return dataset_dict


# individuals
def get_elmo():
    suffix = 'elmo'
    return get_datasets_dict(suffix)

def get_unirep():
    suffix = 'unirep'
    return get_datasets_dict(suffix)

def get_transformer():
    suffix = 'transformer'
    return get_datasets_dict(suffix)



# concats
def get_elmo_and_unirep_concat():
    suffixes = ['elmo', 'unirep']
    return get_concat_datasets_dict(suffixes)


def get_elmo_and_transformer_concat():
    suffixes = ['elmo', 'transformer']
    return get_concat_datasets_dict(suffixes)

def get_transformer_and_unirep_concat():
    suffixes = ['transformer', 'unirep']
    return get_concat_datasets_dict(suffixes)

def get_all_concat():
    suffixes = ['elmo', 'unirep', 'transformer']
    return get_concat_datasets_dict(suffixes)


# padded averages
def get_elmo_and_unirep_padded_avg():
    suffixes = ['elmo', 'unirep']
    return get_padded_average_dict(suffixes)

def get_elmo_and_transformer_padded_avg():
    suffixes = ['elmo', 'transformer']
    return get_padded_average_dict(suffixes)

def get_transformer_and_unirep_padded_avg():
    suffixes = ['transformer', 'unirep']
    return get_padded_average_dict(suffixes)

def get_all_padded_avg():
    suffixes = ['elmo', 'unirep', 'transformer']
    return get_padded_average_dict(suffixes)


# interpolated averages
def get_elmo_and_unirep_inter_avg():
    suffixes = ['elmo', 'unirep']
    return get_interpolated_avg_dict(suffixes)

def get_elmo_and_transformer_inter_avg():
    suffixes = ['elmo', 'transformer']
    return get_interpolated_avg_dict(suffixes)

def get_transformer_and_unirep_inter_avg():
    suffixes = ['transformer', 'unirep']
    return get_interpolated_avg_dict(suffixes)

def get_all_inter_avg():
    suffixes = ['elmo', 'unirep', 'transformer']
    return get_interpolated_avg_dict(suffixes)



def get_1toN_x(x, file_dir = 'data/1toN'):
    mat_file = 'meta_%d'%x
    return get_dict_from_mat(mat_file, file_dir)


def get_SVD_x(x, file_dir = 'data/SVD'):
    mat_file = 'meta_svd_%d'%x
    return get_dict_from_mat(mat_file, file_dir)




if __name__ == '__main__':
    # d1 = get_elmo()
    # print(d1['train'][0].shape[1])

    # d2 = get_unirep()
    # print(d2['train'][0].shape[1])

    # d = get_transformer()
    # print(d['train'][0].shape[1])

    # d = get_elmo_and_unirep_concat()
    # print(d['train'][0].shape[1])

    # d = get_elmo_and_transformer_concat()
    # print(d['train'][0].shape[1])

    # d = get_transformer_and_unirep_concat()
    # print(d['train'][0].shape[1])

    # d = get_all_concat()
    # print(d['train'][0].shape[1])

    # d = get_interpolated_avg_dict(['elmo', 'unirep'])
    # print(d['train'][0].shape[1])

    d = get_1toN_x(1000)
    print(d['train'][0].shape[1])