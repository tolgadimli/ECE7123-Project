import torch
import torch.nn as nn

class ClassAtt(nn.Module):
    def __init__(self, e1, e2, e3, M, dec_h, out_size):
        super().__init__()
        self.e1, self.e2, self.e3 = e1, e2, e3
        self.M = M
        self.dec_h = dec_h
        self.num_class = out_size

        self.w1 = nn.Linear(self.e1, M)
        self.w2 = nn.Linear(self.e2, M)
        self.w3 = nn.Linear(self.e3, M)

        self.wh = nn.Linear(3*self.M, self.M)

        self.wd1 = nn.Linear(2*self.M, self.dec_h)
        self.wd2 = nn.Linear(self.dec_h, self.num_class)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    

    def forward(self, tube):

        x1 = tube[:,0:self.e1]
        x2 = tube[:, self.e1:self.e1+self.e2]
        x3 = tube[:, self.e1+self.e2:self.e1+self.e2+self.e1+self.e3]

        P1 = self.relu(self.w1(x1)) # B x M
        P2 = self.relu(self.w2(x2)) # B x M
        P3 = self.relu(self.w3(x3)) # B x M

        hid1 = torch.cat((P1,P2,P3), axis = 1)
        last_hid = self.wh(hid1)
        #print(self.wh.grad)
        last_hid = self.relu(last_hid) # M x 1

        # attention
        alpha1 = torch.diag(torch.matmul(last_hid, P1.T), 0).unsqueeze(1)
        alpha2 = torch.diag(torch.matmul(last_hid, P2.T), 0).unsqueeze(1)
        alpha3 = torch.diag(torch.matmul(last_hid, P3.T), 0).unsqueeze(1)


        alphas = torch.cat((alpha1, alpha2, alpha3), axis = 1)
        weights = self.softmax(alphas) # B x 3

        context = torch.mul(weights[:,0].unsqueeze(1), P1) + torch.mul(weights[:,1].unsqueeze(1), P2) + torch.mul(weights[:,2].unsqueeze(1), P3)
        decode_start = torch.cat((context, last_hid), axis = 1)

        #print(decode_start.shape)
        # decoder part
        out = self.relu(self.wd1(decode_start))
        out = self.wd2(out) # B x 1195
        #print(self.wh.weight.grad)

        return out



        








        







