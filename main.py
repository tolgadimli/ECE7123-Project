import torch
import torch.nn as nn
from models.model import SimpleMLP2
from embedding_methods import *
from utils import get_dataloaders_from_dict, batch_correct, test_epoch

OUT_SIZE = 1195
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

data = get_elmo()
loaders, inp_size = get_dataloaders_from_dict(data, bs = 128)

torch.manual_seed(231)
model = SimpleMLP2(inp_size, 2048*8, 2048*4, OUT_SIZE)

#model = RecurrentModel(input_size = 32, hidden_size = 512, 
                            #num_layers = 3, num_classes = OUT_SIZE, device = device, type = 'GRU')

model.to(device)


criterion = nn.CrossEntropyLoss(reduction = 'sum')
criterion.to(device)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, 
                               # momentum = 0.9, nesterov = True, weight_decay=1e-5)

optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-3, weight_decay=1e-5)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


train_loader = loaders[0]
val_loader = loaders[1]
test_fold_loader = loaders[2]
test_family_loader = loaders[3]
test_superfamily_loader = loaders[4]



best_val_acc = 0

for epoch in range(1,101):

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


    val_loss, val_acc = test_epoch(model, test_fold_loader, criterion, device)

    print(val_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'model_params.pt')

    
print("TEST PHASE ===============================")
model.load_state_dict(torch.load('model_params.pt'))

test_loss, test_acc = test_epoch(model, test_fold_loader, criterion, device)
print(test_acc)
test_loss, test_acc = test_epoch(model, test_superfamily_loader, criterion, device)
print(test_acc)
test_loss, test_acc = test_epoch(model, test_family_loader, criterion, device)
print(test_acc)
#print(test_loss)