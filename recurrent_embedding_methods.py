import numpy as np
from recurrent_utils import *
from sklearn.preprocessing import normalize
from scipy import interpolate

MIN_SEQ = 17

def get_unified_sequence( ls, seq_len = MIN_SEQ): 
    """ls is list of arrays"""
    print("Reccurence model will be used.")
    for i, arr in enumerate(ls):
        L = len(arr)
        pad_amount = seq_len - L % seq_len
        embd_size = ls[0].shape[1]
        pad_zeros = np.zeros([pad_amount, embd_size])
        padded_residues =  np.concatenate((arr, pad_zeros), axis = 0)
        padded_residue_seq = padded_residues.reshape([-1, 17, embd_size])
        padded_residue_seq = np.mean(padded_residue_seq, axis=0) # 17 X M ARRAY
        padded_residue_seq = normalize(padded_residue_seq, norm = 'l2', axis = 1)
        ls[i] = padded_residue_seq
    return ls



def get_datasets_dict(suffix):
    dataset_dict = {}
    files = ['ordered_rh_train_', 'ordered_rh_valid_', 'ordered_rh_test_fold_',
              'ordered_rh_test_family_', 'ordered_rh_test_superfamily_']

    keys = ['train', 'valid', 'test_fold', 'test_family', 'test_superfamily']
    for k, f in zip(keys, files):
        f = f + suffix
        e, l = get_embd_and_label(f)
        get_unified_sequence(e)
        dataset_dict[k] = (np.stack(e),l)

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
            get_unified_sequence(x)
            x_concat.append(x)
        dataset_dict[k] = (np.concatenate(x_concat, axis = 2), y)

    return dataset_dict


# individuals
def get_elmo():
    suffix = 'elmo_residue'
    return get_datasets_dict(suffix)

def get_unirep():
    suffix = 'unirep_residue'
    return get_datasets_dict(suffix)

def get_transformer():
    suffix = 'transformer_residue'
    return get_datasets_dict(suffix)

# concats
def get_elmo_and_unirep_concat():
    suffixes = ['elmo_residue', 'unirep_residue']
    return get_concat_datasets_dict(suffixes)


def get_elmo_and_transformer_concat():
    suffixes = ['elmo_residue', 'transformer_residue']
    return get_concat_datasets_dict(suffixes)

def get_transformer_and_unirep_concat():
    suffixes = ['transformer_residue', 'unirep_residue']
    return get_concat_datasets_dict(suffixes)

def get_all_concat():
    suffixes = ['elmo_residue', 'unirep_residue', 'transformer_residue']
    return get_concat_datasets_dict(suffixes)




if __name__ == '__main__':
    d = get_elmo()
    #print(d1['train'][0].shape[1])


    # d = get_transformer()
    # print(d['train'][0].shape[1])

    # d = get_elmo_and_unirep_concat()
    # print(d['train'][0].shape[1])

    # d = get_elmo_and_transformer_concat()
    # print(d['train'][0].shape[1])

    # d = get_transformer_and_unirep_concat()
    # print(d['train'][0].shape[1])

    d = get_all_concat()
    print(d['train'][0].shape[1])

    # d = get_interpolated_avg_dict(['elmo', 'unirep'])
    # print(d['train'][0].shape[1])