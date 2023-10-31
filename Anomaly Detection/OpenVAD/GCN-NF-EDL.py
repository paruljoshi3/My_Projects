from maf.option import parser
import torch
import torch.nn as nn
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from os.path import join
# from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from collections import OrderedDict as OD 
from torchvision import transforms
from sklearn.model_selection import train_test_split


from maf.iafdataset import Dataset as IAFDataset
from maf.dataset import Dataset as Dataset 
from maf.model2 import GCN, EDLModel

from VAE import VAE, normalize_tensor

#for EDL Head
from sklearn.metrics import auc, precision_recall_curve, accuracy_score
import torch.optim as optim


device = torch.device('cuda')
args = parser.parse_args()

nf_model = VAE(args).to(device)
gcn_model =GCN(args).to(device)
edl_head = EDLModel().to(device)

if __name__ == '__main__':
    
    
    train_loader = torch.utils.data.DataLoader(Dataset(args, test_mode=False), batch_size=128, num_workers=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(Dataset(args, test_mode=True), batch_size=128, num_workers=2, shuffle=True)
    
   
    
    criterion = torch.nn.BCELoss()
    base_param = edl_head.parameters()

    optimizer = optim.SGD(params=edl_head.classifier.parameters(), lr=0.01)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
    
    train_score = []
    for epoch in range(args.max_epoch):
        print('Epoch: ', epoch + 1, '/', args.max_epoch)
        train_score = []

        with torch.set_grad_enabled(True):
            loss_m = []
            score = []
            
            for batch_idx, (data, label) in enumerate(train_loader):
                edl_head.train()
                pred = torch.zeros(0).to(device) #creates an empty tensor 
                
                # print("Batch:", batch_idx)
                data, label = data.to(device), label.to(device)
                seq_len = torch.sum(torch.max(torch.abs(data), dim=2)[0]>0, 1)
                data = data[:, :200, :]
                input = gcn_model(data, seq_len)
                
                input = normalize_tensor(input)
                try:
                    mis = torch.zeros(0).to(device)
                    for i in range(input.shape[0]):
                        if label[i] == 0.0:
                            x, _, _ = nf_model(input[i])
                            x = x.view(-1, 200, 32)
                            mis = torch.cat((x, mis))
                    input = torch.cat((mis, input))
                    label = torch.cat((torch.ones(mis.shape[0]).to(device), label))
                except:
                    pass                    
                output = edl_head(input)
                
                temp = torch.squeeze(output)
                sig = torch.sigmoid(temp)
                sig = torch.mean(sig, 1)
                pred = torch.cat((pred, sig))
                
                loss = criterion(pred.to(torch.float32), label.to(torch.float32))
                
                precision, recall, th = precision_recall_curve(label.cpu().detach().numpy(), pred.detach().cpu().numpy())
                pr_auc = auc(recall, precision)
                score.append(pr_auc)
                loss_m.append(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            scheduler.step()
            # train_score.append(sum(metrics)/len(metrics))
            print("Epoch train score: ", sum(score)/len(score))

           #save model performance and scores 
            with open("train_final2.txt", "a") as file:
                line = [epoch+1, sum(loss_m)/len(loss_m), sum(score)/len(score)]
                file.write(str(line)+"\n")


            if epoch % 2 == 0 and not epoch == 0:
                torch.save(edl_head.state_dict(), './ckpt/' + args.model_name + '{}.pkl'.format(epoch)) # save every alternate weight file

        
        with torch.no_grad():
            val_loss = []
            validate = []
            
            for batch_idx, (data, label) in enumerate(test_loader):
                edl_head.train()
                pred = torch.zeros(0).to(device) #creates an empty tensor 
                
                # print("Batch:", batch_idx)
                data, label = data.to(device), label.to(device)
                seq_len = torch.sum(torch.max(torch.abs(data), dim=2)[0]>0, 1)
                data = data[:, :200, :]
                input = gcn_model(data, seq_len)
                
                input = normalize_tensor(input)
                try:
                    mis = torch.zeros(0).to(device)
                    for i in range(input.shape[0]):
                        if label[i] == 0.0:
                            x, _, _ = nf_model(input[i])
                            x = x.view(-1, 200, 32)
                            mis = torch.cat((x, mis))
                    input = torch.cat((mis, input))
                    label = torch.cat((torch.ones(mis.shape[0]).to(device), label))
                                    
                except:
                    pass                    
                output = edl_head(input)
                
                temp = torch.squeeze(output)
                sig = torch.sigmoid(temp)
                sig = torch.mean(sig, 1)
                pred = torch.cat((pred, sig))
                
                loss = criterion(pred.to(torch.float32), label.to(torch.float32))
                val_loss.append(loss)
                
                precision, recall, th = precision_recall_curve(label.cpu().detach().numpy(), pred.detach().cpu().numpy())
                pr_auc = auc(recall, precision)
                validate.append(pr_auc)

            avg_loss = sum(val_loss)/len(val_loss)
            score = sum(validate)/len(validate)

            print("Epoch validation score: ", score)
            
            with open("val_final2.txt", "a") as file:
                line = [epoch+1, avg_loss, score]
                file.write(str(line)+"\n")