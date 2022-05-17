import torch
import torch.nn as nn
from math import sqrt
        
class Attention(nn.Module):
    def __init__(self, key_size, que_size, val_size, proj_size):
        super().__init__()
        self.key_size = key_size
        self.que_size = que_size
        self.val_size = val_size
        self.proj_size = proj_size

        self.key_W = nn.Linear(self.key_size, self.proj_size)
        self.query_W = nn.Linear(self.que_size, self.proj_size)
        self.value_W = nn.Linear(self.val_size, self.proj_size)
        self.softmax = nn.Softmax(dim = 1)

        
    def forward(self, k, q, v): # x : Batch_size x emb_size

        key = self.key_W(k)
        query = self.query_W(q)
        value = self.value_W(v)

        # print(key.shape)
        # print(query.mT.shape)

        k_q_prod = torch.div( torch.bmm( key, query.mT  ), sqrt(self.proj_size) ) #query.view(-1, 1, self.proj_size)
        k_q_prod = self.softmax(k_q_prod)

        out = torch.matmul(k_q_prod, value)

        return out
        


class MultiHeadAttention(nn.Module):
    def __init__(self, head_count, dff, d_model, proj_size, device):
        super().__init__()
        self.head_count = head_count
        self.proj_size = proj_size

        self.dff = dff
        self.linear = nn.Linear(proj_size*head_count, d_model)

        self.attentions = []
        for _ in range(self.head_count):
            a = Attention(d_model, d_model, d_model, proj_size).to(device)
            self.attentions.append(a)

        
    def forward(self, k, q, v): # x : Batch_size x emb_size

        cat_list = [] 
        for att in self.attentions:
            c = att(k, q, v)
            cat_list.append(c)#.unsqueeze(1))

        catted = torch.cat(cat_list, axis = 2) #concat
        out = self.linear(catted)

        return out


class TromerBlock(nn.Module):
    def __init__(self, h, dff, d_model, proj_size, device):
        super().__init__()

        self.att = MultiHeadAttention(h, dff, d_model, proj_size, device)
        self.att.to(device)

        ff_list = [nn.Linear(d_model, dff), nn.ReLU(), nn.Linear(dff, d_model), nn.Dropout(0.1)] # feed forward network from the paper
        self.feed_forward = nn.Sequential(*ff_list) # W_o in the paper
        self.feed_forward.to(device)

    def forward(self, q, k ,v):

        bs = v.shape[0]
        l = v.shape[1]

        att_out = self.att(q, k ,v)
        out = att_out + v

        mean = torch.mean(out.view( bs, -1 ), dim = 1).view(bs, 1, -1)
        std = torch.std(out.view( bs, -1 ) , dim = 1).view(bs, 1, -1)
        out = (out - mean) / (std + 1e-8)

        ## second phase
        ff_out = self.feed_forward(out)

        out = out + ff_out

        mean = torch.mean(out.view( bs, -1 ), dim = 1).view(bs, 1, -1)
        std = torch.std(out.view( bs, -1 ) , dim = 1).view(bs, 1, -1)
        out = (out - mean) / (std + 1e-8)

        return out
        

class Transformer(nn.Module):
    def __init__(self, N,  h, dff, d_model, proj_size, emb_size, out_size, device):
        super().__init__()

        self.emb_size = emb_size
        self.d_model = d_model
        self.N = N

        self.encoders = []
        self.decoders = []

        for _ in range(N):
            self.encoders.append(TromerBlock( h, dff, d_model, proj_size, device))
            self.decoders.append(TromerBlock( h, dff, d_model, proj_size, device))

        # self.encoder = TromerBlock( h, dff, d_model, proj_size, device)
        # self.decoder = TromerBlock( h, dff, d_model, proj_size, device)

        self.classifier = nn.Sequential(* [ nn.Linear(N*emb_size*d_model, 2048), nn.ReLU(), nn.Dropout(0.1), 
                                            nn.Linear(2048, 2048), nn.ReLU(),  nn.Dropout(0.1),  nn.Linear(2048, out_size) ] )

    
    def forward(self, x):

        x_ = x.unsqueeze(2) # BS x EMB => BS x EMB x 1
        
        outs = []
        for i in range(self.N):
            encoder, decoder = self.encoders[i], self.decoders[i]
            enc_out = encoder(x_, x_, x_)

            decode_out = decoder(enc_out, enc_out, x_)
            decode_out = decode_out.view(-1, self.emb_size * self.d_model)
            outs.append(decode_out)
        #print(decode_out.shape)

        outs_concat = torch.cat(outs, dim = 1)
        del outs 
        out = self.classifier(outs_concat)
        
        return out

# if __name__ == '__main__':

    

#     N = 6
#     h = 5
    
#     d_model = 1
#     proj_size = 32
#     dff = proj_size*4
#     emb_size = 1900
#     x = torch.randn([4, emb_size])
#     out_size = 1195
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#     # enc = TromerBlock( h, dff, d_model, proj_size, device)
#     # enc = enc.to(device)
#     # x = x.to(device)
#     # a = enc(x)

#     N = 6
#     trf = Transformer(N, h, dff, d_model, proj_size, emb_size, out_size, device)
#     trf = trf.to(device)
#     x = x.to(device)

#     a = trf(x)







