import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from maf.maf_layer import MAFLayer
from maf.made import MADE
from maf.batch_norm_layer import BatchNormLayer
from maf.option import parser
from maf.model2 import GCN
from torchvision import transforms
from maf.iafdataset import Dataset as NormDataset
from maf.utility import * 
from os.path import join



def normalize_tensor(tensor, dim=None):

    mean = tensor.mean(dim=dim)
    std = tensor.std(dim=dim)

    # Step 2: Subtract mean
    tensor_centered = tensor - mean

    # Step 3: Divide by standard deviation, handle division by zero
    tensor_normalized = tensor_centered / std.where(std != 0, torch.tensor(1.0))

    return tensor_normalized

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.register_parameter('h', torch.nn.Parameter(torch.zeros(args.h_size)))
        self.register_parameter('dec_log_stdv', torch.nn.Parameter(torch.Tensor([0.])))

        layers = []
        for i in range(args.depth):
            layer = []

            for j in range(args.n_blocks):
                downsample = (i > 0) and (j == 0)
                layer += [MAFLayer(dim=args.h_size, hidden_dims=[256, 256], reverse=downsample)]
                # layer += [BatchNormLayer(dim=args.h_size)]

            layers += [nn.ModuleList(layer)]
 
        self.layers = nn.ModuleList(layers)
        self.first_conv = nn.Conv2d(1, args.h_size, 4, 2, 1)
        self.last_conv = nn.ConvTranspose2d(4, 1, 4, 2, 1)

    def forward(self, input):
        input = input.view(-1, 1, 200, 32)
        x = self.first_conv(input)
        kl, kl_obj = 0., 0.
        h = self.h
        for layer in self.layers:
            for sub_layer in layer: 
                x, log_det = sub_layer(x)
                kl += torch.sum(log_det)

        h = h.expand_as(x)

        for layer in reversed(self.layers):
            for sub_layer in reversed(layer):
                h, curr_kl = sub_layer.backward(h)
                kl += torch.sum(curr_kl)

        x = F.elu(h)
        x = self.last_conv(x)
        x = x.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)

        return x, kl, kl_obj

    def sample(self, n_samples=64):
        h = self.h.view(1, -1, 1, 1)
        h = h.expand((n_samples, *self.hid_shape))

        for layer in reversed(self.layers):
            for sub_layer in reversed(layer):
                h, _, _ = sub_layer.backward(h, sample=True)

        x = F.elu(h)
        x = self.last_conv(x)

        return x.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)

    def cond_sample(self, input):
        input = input.view(-1, 1, 200, 32)

        x = self.first_conv(input)
        kl, kl_obj = 0., 0.

        h = self.h.view(1, -1, 1, 1)

        for layer in self.layers:
            for sub_layer in layer:
                x, log_det = sub_layer(x)
                kl += torch.sum(log_det)

        h = h.expand_as(x)
        outs = []

        current = 0
        for i, layer in enumerate(reversed(self.layers)):
            for j, sub_layer in enumerate(reversed(layer)):
                h, curr_kl, curr_kl_obj = sub_layer.backward(h)
                kl += torch.sum(curr_kl)

                h_copy = h
                again = 0

                for layer_ in reversed(self.layers):
                    for sub_layer_ in reversed(layer_):
                        if again > current:
                            h_copy, _, _ = sub_layer_.backward(h_copy, sample=True)

                        again += 1

                x = F.elu(h_copy)
                x = self.last_conv(x)
                x = x.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)
                outs += [x]

                current += 1

        return outs

import torch.nn.utils.rnn as rnn_utils
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def collate_batch(batch):

    batch = sorted(batch, key=lambda x: x[1], reverse=True)
    sequences, lengths = zip(*batch)
    packed_sequence = pack_padded_sequence(torch.stack(sequences), lengths, batch_first=True)

    return packed_sequence


if __name__ == '__main__':
    device = torch.device("cuda")
    args = parser.parse_args()
    #Defining models segments pretrained using MIL_EDL.py
    gcn_model = GCN(args)
    gcn_model = gcn_model.to(device)
    
    
    # #Normalizing flow model (to be trained in this section)
    nf_model = VAE(args).to(device)
    

    #training parameters for Normalizing Flow model
    opt = torch.optim.Adamax(nf_model.parameters(), lr=args.lr)

    # create datasets / dataloaders
    scale_inv = lambda x : x + 0.5
    ds_transforms = transforms.Compose([transforms.ToTensor(), lambda x : x - 0.5])
    kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}

    # Preparing data for normalizing flow model, uses only Normal Data

    train_loader = torch.utils.data.DataLoader(NormDataset(args, test_mode=False), batch_size=16, num_workers=2, shuffle=True)

    test_loader  = torch.utils.data.DataLoader(NormDataset(args, test_mode=True), batch_size=16, num_workers=2, shuffle=True)
    


    print("Loaders initiated!")

    # spawn writer
    model_name = 'NB{}_D{}_Z{}_H{}_BS{}_FB{}_LR{}_MAF{}'.format(args.n_blocks, args.depth, args.z_size, args.h_size, 
                                                                args.batch_size, args.free_bits, args.lr, args.iaf)

    model_name = 'test' if args.debug else model_name
    log_dir    = join('runs', model_name)
    # sample_dir = "C:/Users/TRISHA/Downloads/OpenVAD-Final/OpenVAD/runs/NB4_D2_Z25_H25_BS128_FB0.1_LR0.001_IAF5 "
    # writer     = SummaryWriter(log_dir=log_dir)
    # maybe_create_dir(sample_dir)
    maybe_create_dir(log_dir)

    print_and_save_args(args, log_dir)
    # print('logging into %s' % log_dir)
    best_test = float('inf')

    ## create psuedo anomaly holder

    print('NF Model training --->')
    print(torch.cuda.is_available())
    for epoch in range(args.max_epoch):
        nf_model.train()
        train_log = reset_log()
        print(f'Epoch {epoch+1}/{args.max_epoch}')
        for batch_idx, (data,_) in enumerate(train_loader):
            # print("GCN Model running!")
            data = data.to(device)
            seq_len = torch.sum(torch.max(torch.abs(data), dim=2)[0]>0, 1)
            data = data[:, :200, :]
            input = gcn_model(data, seq_len)

            input = normalize_tensor(input)

            
            # print("NF Model running!")
            input = input.view(-1, 1, 200, 32)
            input = input.to(device) 
            x, kl, kl_obj = nf_model(input)

            log_pxz = logistic_ll(x, nf_model.dec_log_stdv, sample=input)
            loss = (kl_obj - log_pxz).sum() / x.size(0)
            elbo = (kl     - log_pxz)
            bpd  = elbo / (32 * 32 * 3 * np.log(2.))

            train_log['kl']         += [kl.mean()]
            train_log['bpd']        += [bpd.mean()]
            train_log['elbo']       += [elbo.mean()]
            # train_log['kl obj']     += [kl_obj.mean()]
            train_log['log p(x|z)'] += [log_pxz.mean()]
            # print("Loss calculated")
        
            opt.zero_grad()
            loss.backward()
            opt.step()

        
        nf_model.eval()
        test_log = reset_log()

        with torch.no_grad():
            for batch_idx, (data,_) in enumerate(train_loader):
            # print("GCN Model running!")
                data = data.to(device)
                seq_len = torch.sum(torch.max(torch.abs(data), dim=2)[0]>0, 1)
                data = data[:, :200, :]
                input = gcn_model(data, seq_len)

                input = normalize_tensor(input)

                
                # print("NF Model running!")
                input = input.view(-1, 1, 200, 32)
                input = input.to(device) 
                x, kl, kl_obj = nf_model(input)

                log_pxz = logistic_ll(x, nf_model.dec_log_stdv, sample=input)
                loss = (kl_obj - log_pxz).sum() / x.size(0)
                elbo = (kl     - log_pxz)
                bpd  = elbo / (32 * 32 * 3 * np.log(2.))
                
                test_log['kl']         += [kl.mean()]
                test_log['bpd']        += [bpd.mean()]
                test_log['elbo']       += [elbo.mean()]
                # test_log['kl obj']     += [kl_obj.mean()]
                test_log['log p(x|z)'] += [log_pxz.mean()]
                
        savestr = 'nfmodel'+str(epoch)+'.pth'
        # current_test = sum(test_log['bpd']) / batch_idx
        torch.save(nf_model.state_dict(), join(log_dir, savestr))
            
    
    
    #Final section, train EDL Head
    
   