import os
import numpy as np
import pickle
from embedding_class import EmbeddingSet
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import StandardScaler

MIN_SEQ = 17

# Also gets 17 sequences from all residues
def get_embd_and_label(file_name, file_dir = 'data/pickles_residue'):
    """Function to get embeddings and labels from the pickle file"""

    if 'unirep' in file_name or 'transformer' in file_name:
        print("sa")
        file = os.path.join(file_dir,file_name)
        with open(file, 'rb') as f:
            x_dict = pickle.load(f)

        embds = []
        labels = []
        for _, value in x_dict.items():
            e = value[0]
            e = e[1:-1]

            t = value[2]
            t = t.astype(np.int64)
            embds.append(e)
            labels.append(t)

        labels = np.array(labels)

    elif 'elmo' in file_name:
        print("as")

        file = os.path.join(file_dir,file_name)
        with open(file, 'rb') as f:
            x_dict = pickle.load(f)

        embds = []
        labels = []
        for _, value in x_dict.items():
            e = value[0]

            t = value[2]
            t = t.astype(np.int64)
            embds.append(e)
            labels.append(t)

        labels = np.array(labels)

    return embds, labels 





def get_dataloaders_from_dict( data_dict, bs=128 ):

    dataloaders = []
    # training set
    (emb, lab) = data_dict['train']
    mean = np.mean(emb, axis = 0)
    print(mean.shape)
    std = np.std(emb, axis = 0) + 1e-8 # for numerical stability
    print(std.shape)
    emb_norm = (emb - mean) / std
    dataset = EmbeddingSet(emb_norm,lab)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    dataloaders.append(loader) 

    #remaining ones
    keys = ['valid', 'test_fold', 'test_family', 'test_superfamily']
    for k in keys:
        (emb, lab) = data_dict[k]
        emb_norm = (emb - mean) / std
        dataset = EmbeddingSet(emb_norm,lab)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        dataloaders.append(loader)

    return dataloaders, emb.shape[2]


def batch_correct(pred, target, device):
    _, predicted = torch.max(pred.data, 1)
    correct = (predicted.to(device) == target).sum().item()
    return correct


def test_epoch(model, test_loader, criterion, device):

    all_correct = 0
    all_loss = 0
    total_size = 0
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, target) in enumerate(test_loader):
            l = len(target)
            total_size += l

            data, target = data.to(device), target.to(device)
            pred = model(data)
            loss = criterion(pred, target)
            all_loss += loss.item()
            
            correct = batch_correct(pred, target, device)
            all_correct += correct

    all_correct = all_correct / total_size
    all_loss = all_loss / total_size
    all_error = (1 - all_correct) * 100
    all_correct = all_correct * 100
    
    return all_loss, all_correct


# def test_epoch_recurrent(model, inp_size, test_loader, criterion, device):

#     all_correct = 0
#     all_loss = 0
#     total_size = 0
#     with torch.no_grad():
#         model.eval()
#         for batch_idx, (data, target) in enumerate(test_loader):
#             l = len(target)
#             total_size += l

#             data, target = data.to(device), target.to(device)
#             data = data.reshape(-1, inp_size, 1) #only for Recurrent types
#             pred = model(data)
#             loss = criterion(pred, target)
#             all_loss += loss.item()
            
#             correct = batch_correct(pred, target, device)
#             all_correct += correct

#     all_correct = all_correct / total_size
#     all_loss = all_loss / total_size
#     all_error = (1 - all_correct) * 100
#     all_correct = all_correct * 100
    
#     return all_loss, all_correct



if __name__ == '__main__':

    e, t = get_embd_and_label('ordered_rh_train_elmo_residue')