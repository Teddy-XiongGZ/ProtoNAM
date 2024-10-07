import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiChannelLinear(nn.Module):
    
    def __init__(self, in_dim, out_dim, n_channel=1):
        super(MultiChannelLinear, self).__init__()
        
        #initialize weights
        self.w = torch.nn.Parameter(torch.zeros(n_channel, out_dim, in_dim))
        self.b = torch.nn.Parameter(torch.zeros(1, n_channel, out_dim))
        
        #change weights to kaiming
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b, -bound, bound)
    
    def forward(self, x):
        '''
            args:
                x: input, whose shape can be 
                    batch_size, (channel), in_dim
            return:
                output, whose shape will be
                    batch_size, (channel), out_dim
        '''
        # b, ch, r, c  = x.size()
        # return (( x * self.w).sum(-1 ) + self.b).view(b,ch,1,-1)
        return (self.w * x.unsqueeze(-2)).sum(-1) + self.b

class ExU(torch.nn.Module):

    def __init__(self, in_feat, out_feat):
        super(ExU, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weights = nn.parameter.Parameter(torch.Tensor(in_feat, out_feat))
        self.bias = nn.parameter.Parameter(torch.Tensor(in_feat))

        torch.nn.init.trunc_normal_(self.weights, mean=4.0, std=0.5)
        torch.nn.init.trunc_normal_(self.bias, std=0.5)

    def forward(self, inputs, n = 1):
        output = (inputs - self.bias).matmul(torch.exp(self.weights))

        output = F.relu(output)
        output = torch.clamp(output, 0, n)

        return output

class ProtoNAM(nn.Module):

    def __init__(self, problem, n_feat, h_dim, n_proto, n_layers, n_class, dropout=0.0, dropout_output=0.0, output_penalty=0.0, p=1, n_layers_pred=2, batch_norm=False):
        super(ProtoNAM, self).__init__()

        self.problem = problem
        self.n_feat = n_feat
        self.h_dim = h_dim
        self.n_class = n_class
        self.n_layers = n_layers
        self.n_proto = n_proto
        self.p = p
        if self.p == 1:
            self.mask = nn.parameter.Parameter(F.one_hot(torch.arange(n_feat), num_classes=n_feat), requires_grad=False) # (n_comp, n_feat)
        elif self.p == 2:
            mask = torch.stack([torch.arange(self.n_feat).repeat_interleave(self.n_feat), torch.arange(self.n_feat).repeat(self.n_feat)]).transpose(0,1)
            mask = mask[mask[:,0] < mask[:,1]]
            self.mask = nn.parameter.Parameter(F.one_hot(mask[:,0], num_classes=n_feat) + F.one_hot(mask[:,1], num_classes=n_feat), requires_grad=False)
        elif self.p == -1:
            self.mask = nn.parameter.Parameter(torch.ones(1, self.n_feat), requires_grad=False)
        self.n_comp = len(self.mask)
        self.output_penalty = output_penalty

        assert self.n_layers > 0

        self.dropout = nn.Dropout(p=dropout)
        self.dropout_output = nn.Dropout(p=dropout_output)
        self.activate = nn.ReLU()

        if batch_norm:
            self.norm = nn.BatchNorm1d(self.n_feat)
        else:
            self.norm = nn.LayerNorm(self.h_dim)
        self.lnorm = nn.LayerNorm(self.h_dim)

        self.enc_list = nn.Sequential(
            nn.Sequential(MultiChannelLinear(1, self.h_dim, self.n_feat), self.norm, self.dropout, self.activate),
            *([nn.Sequential(
                MultiChannelLinear(self.h_dim + 1, self.h_dim, self.n_feat), self.norm, self.dropout, self.activate
            ) for _ in range(self.n_layers - 1)])
        )

        self.center = nn.parameter.Parameter(torch.zeros(self.n_layers, self.n_proto, self.n_feat), requires_grad = True)
        self.coeff = nn.parameter.Parameter(torch.ones(self.n_layers, self.n_proto, self.n_feat), requires_grad = True)
        self.bias = nn.parameter.Parameter(torch.zeros(self.n_layers, self.n_proto, self.n_feat), requires_grad = True)
        
        # if self.n_layers == 1:
        #     self.clfs = nn.Sequential(
        #         *([nn.Sequential(
        #             nn.Linear(self.n_feat * self.h_dim, self.n_class), 
        #         ) for _ in range(self.n_layers)])
        #     )
        # else:
        
        # self.clfs = nn.Sequential(
        #     *([nn.Sequential(
        #         nn.Linear(self.n_feat * self.h_dim, self.h_dim), \
        #         # self.lnorm, self.dropout, self.activate, nn.Linear(self.h_dim, self.h_dim), \
        #         self.lnorm, self.dropout, self.activate, nn.Linear(self.h_dim, self.n_class), 
        #     ) for _ in range(self.n_layers)])
        # )

        if n_layers_pred == 1:
            self.clfs = nn.Sequential(
                *([nn.Sequential(
                    nn.Linear(self.n_feat * self.h_dim, self.n_class), 
                ) for _ in range(self.n_layers)])
            )
        else:
            self.clfs = nn.Sequential(
                *([nn.Sequential(
                    nn.Linear(self.n_feat * self.h_dim, self.h_dim), \
                    *([it for _ in range(n_layers_pred - 1) for it in (self.lnorm, self.dropout, self.activate, nn.Linear(self.h_dim, self.n_class))]), 
                ) for _ in range(self.n_layers)])
            )

        self.aggs = nn.Sequential(
            *([nn.Linear(self.n_comp, 1) for _ in range(self.n_layers)])
        )
        if self.problem == "regression":
            self.criterion = nn.MSELoss()
        elif self.n_class == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        nn.init.xavier_normal_(self.center)
        for fc in self.aggs:
            nn.init.xavier_normal_(fc.weight)
        for clf in self.clfs:
            for fc in clf:
                if type(fc) == nn.Linear:
                    nn.init.xavier_normal_(fc.weight)
    
    def initialize(self, X):
        with torch.no_grad():
            percent_q = 100 * np.arange(start = 1.0, stop = self.n_proto + 1) / (self.n_proto + 1)
            self.center[:] = torch.from_numpy(np.percentile(X, percent_q, axis=0)[None])

    def forward(self, x, y, T = 1e-8):
        '''
        Input:
            x: input features (batch_size, n_feat)
            y: target label / value (batch_size)
        '''
        bsz = len(y)

        Z = self.encode(x, T = T) # list of (batch_size, n_feat, h_dim)

        Logits = []
        output_loss = 0
        for i in range(len(Z)):
            res, loss = self.predict(self.clfs[i], self.aggs[i], Z[i])
            output_loss += loss
            Logits.append(res) # (batch_size, n_class) or (batch_size)
        
        for i in range(len(Logits) - 1):
            Logits[i + 1] += Logits[i]
        
        total_loss = 0

        for i in range(len(Logits)):
            total_loss += self.criterion(Logits[i], y)
        
        return total_loss + self.output_penalty * output_loss, Logits[-1]
    
    def encode(self, x, T):
        '''
            Input:
                x: batch of features (batch, n_feat)
                T: temperature (float)
            Other param used:
                self.center: learned prototypes (n_layers, n_proto, n_feat)
                self.coeff: learned prototypes (n_layers, n_proto, n_feat)
                self.bias: learned prototypes (n_layers, n_proto, n_feat)
        '''        
        dists = torch.pow(x[:,None,None] - self.center[None], 2) # (batch_size, n_layers, n_proto, n_feat)
        x_res = (torch.softmax(-dists/T, dim=-2) * (x[:,None,None] * self.coeff[None] + self.bias[None])).sum(-2) # (batch_size, n_layers, n_feat)
        Z = []
        for i, layer in enumerate(self.enc_list):
            if i == 0:
                # z = layer(torch.cat([x_res[:,i].unsqueeze(-1)], dim=-1))
                # z = layer(x.unsqueeze(-1))
                z = layer(x_res[:,i].unsqueeze(-1))
            else:
                z = layer(torch.cat([z, x_res[:,i].unsqueeze(-1)], dim=-1))
            Z.append(z)

        return Z

    def predict(self, clf, agg, z):
        '''
            Input:
                clf: classifier
                    input_size: (n_feat * h_dim)
                    output_size: (n_class)
                agg: aggregator which combines information of different interactions
                    input_size: (n_comp)
                    output_size: (1)
                z: encoded input
                    size: (batch_size, n_feat, h_dim)
            Used parameter:
                mask: mask for high-order interactions (n_comp, n_feat)
        '''
        z_comp = z[:,None] * self.mask[None,:,:,None] # (batch_size, n_comp, n_feat, h_dim)
        res = clf(z_comp.flatten(start_dim=-2)) # (batch_size, n_comp, n_class)
        res = self.dropout_output(res)
        loss = res.pow(2).mean()
        # res = agg(res.transpose(-1,-2)).squeeze(-1)
        res = (res * agg.weight[0,None,:,None]).mean(-2) + agg.bias[None,:]
        return res.squeeze(-1), loss
