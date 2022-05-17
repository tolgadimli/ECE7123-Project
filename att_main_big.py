import torch
import torch.nn as nn
from models.classAttModel2 import ClassAttBig
from embedding_methods import *
from utils import get_dataloaders_from_dict, batch_correct, test_epoch
from tqdm import tqdm
import pandas as pd

OUT_SIZE = 1195
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(231)

# dataset_dictionaries1 = [get_elmo_and_unirep_concat(),
#                         get_elmo_and_transformer_concat(),   get_transformer_and_unirep_concat() , get_all_concat() ]

data = get_all_concat()
loaders, e = get_dataloaders_from_dict(data, bs = 128)

train_loader = loaders[0]
val_loader = loaders[1]
test_fold_loader = loaders[2]
test_family_loader = loaders[3]
test_superfamily_loader = loaders[4]




#optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-3, weight_decay=1e-5)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

pandas_ls = []
M_list = list(np.arange(4,15)*250)
print(M_list)
for M in M_list:
    hid1 = 2048*2
    hid2 = 2048*2
    rep_len = 32
    model = ClassAttBig(e, rep_len, M, hid1, hid2, OUT_SIZE) 
    model.to(device)

    criterion = nn.CrossEntropyLoss(reduction = 'sum')
    criterion.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, 
                                    momentum = 0.9, nesterov = True, weight_decay=1e-4)
    best_val_acc = 0
    row_ls = [M]
    for epoch in tqdm(range(1,101)):
        

        train_correct = 0
        total_size = 0
        train_loss = 0.0

        model.train()
        for i, (input, target) in enumerate(train_loader):

            bs = len(target)
            input, target = input.to(device), target.to(device)
            #input = input.reshape(-1, 32, 32)
            #print(input.shape)
            optimizer.zero_grad()


            pred = model(input)
            loss = criterion(pred, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            bc = batch_correct(pred, target, device)
            train_correct += bc
            total_size += bs
        
        train_loss = train_loss / total_size
        train_correct = train_correct / total_size
        epoch_err = (1 - train_correct)/100

        if epoch % 30 == 0:
            print("Learning rate drop!!!!!")
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*0.1


        val_loss, val_acc = test_epoch(model, val_loader, criterion, device)

        if val_acc > best_val_acc and epoch > 15:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model_params.pt')

        
    print("TEST PHASE ===============================")
    #model.load_state_dict(torch.load('model_params.pt'))
    test_loss, test_acc = test_epoch(model, train_loader, criterion, device)
    #print(test_acc)
    row_ls.append(round(test_acc*100)/100)

    test_loss, test_acc = test_epoch(model, val_loader, criterion, device)
    #print(test_acc)
    row_ls.append(round(test_acc*100)/100)

    test_loss, test_acc = test_epoch(model, test_fold_loader, criterion, device)
    #print(test_acc)
    row_ls.append(round(test_acc*100)/100)

    test_loss, test_acc = test_epoch(model, test_superfamily_loader, criterion, device)
    #print(test_acc)
    row_ls.append(round(test_acc*100)/100)

    test_loss, test_acc = test_epoch(model, test_family_loader, criterion, device)
    #print(test_acc)
    row_ls.append(round(test_acc*100)/100)

    pandas_ls.append(row_ls)

    df = pd.DataFrame(data = pandas_ls, columns = ['M','train', 'val', 'fold', 'superfamily', 'family'])
    df.to_csv('mlp_results_attbig.csv')