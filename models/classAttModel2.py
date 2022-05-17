import torch
import torch.nn as nn

class ClassAttBig(nn.Module):
    def __init__(self, e, rep_len, M, hid1, hid2, out_size):
        super().__init__()
        self.e = e
        self.M = M
        #self.dec_h = dec_h
        self.num_class = out_size
        self.attn = nn.Linear(self.M, 1, bias= False)

        self.w_start = nn.Linear(self.e, self.M)
        self.state_maker = []
        for _ in range(rep_len):
            self.state_maker.append(nn.Linear(M,M))
            # state_maker.append(nn.ReLU())
        self.hid_layers = nn.Sequential(*self.state_maker)

        self.dropout = nn.Dropout(0.3)

        
        decode_ls = [nn.Linear(M, hid1), nn.ReLU(), nn.Linear(hid1, hid2), nn.ReLU(), self.dropout, nn.Linear(hid2, out_size)]
        self.decode_layers = nn.Sequential(*decode_ls)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()



    def forward(self, x):

        prep = self.relu(self.w_start(x))

        forward_list = [prep]
        for layer in self.state_maker:
            hid = self.relu(layer(forward_list[-1]))
            self.dropout(hid)
            forward_list.append(hid) 

        # getting alphas
        alphas = []
        for fid in forward_list:
            a = self.attn(fid)
            alphas.append(a)

        weights = torch.cat(alphas, dim = 1)
        weights = self.softmax(weights)

        #print(weights.shape)
        

        context = 0
        for ind, fid in enumerate(forward_list):
            context += torch.mul(weights[:,ind].unsqueeze(1), fid)

        out = self.decode_layers(context)
        #print(self.w_start.weight.grad)
        return out

        

        # hid1 = torch.cat((P1,P2,P3), axis = 1)
        # last_hid = self.wh(hid1)
        # #print(self.wh.grad)
        # last_hid = self.relu(last_hid) # M x 1

        # # attention
        # alpha1 = torch.diag(torch.matmul(last_hid, P1.T), 0).unsqueeze(1)
        # alpha2 = torch.diag(torch.matmul(last_hid, P2.T), 0).unsqueeze(1)
        # alpha3 = torch.diag(torch.matmul(last_hid, P3.T), 0).unsqueeze(1)


        # alphas = torch.cat((alpha1, alpha2, alpha3), axis = 1)
        # weights = self.softmax(alphas) # B x 3

        # context = torch.mul(weights[:,0].unsqueeze(1), P1) + torch.mul(weights[:,1].unsqueeze(1), P2) + torch.mul(weights[:,2].unsqueeze(1), P3)
        # decode_start = torch.cat((context, last_hid), axis = 1)

        # #print(decode_start.shape)
        # # decoder part
        # out = self.relu(self.wd1(decode_start))
        # out = self.wd2(out) # B x 1195
        # #print(self.wh.weight.grad)

        #return out



        








        







