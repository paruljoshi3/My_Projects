from torch.utils.data import DataLoader

import torch.optim as optim
import torch
import time
import numpy as np
import random
import os
from maf.model2 import GCN, EDLModel
from maf.dataset import Dataset
from maf.train2 import train, TripletLoss, CLAS, sample_triplets
from maf.option import parser

import torch.multiprocessing as mp
import numpy as np

from sklearn.metrics import auc, precision_recall_curve, accuracy_score
import torch
from sklearn.model_selection import train_test_split
import warnings




# Ignore all UserWarning messagespip


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
args = parser.parse_args()
device = torch.device("cuda")
train_loader = torch.utils.data.DataLoader(Dataset(args, test_mode=False), batch_size=128, num_workers=2, shuffle=True)
test_loader = torch.utils.data.DataLoader(Dataset(args, test_mode=True), batch_size=128, num_workers=2, shuffle=True)

gcn = GCN(args).to(device)
edl_head = EDLModel().to(device)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # setup_seed(2333)
    

    gcn_parameters = list(gcn.parameters())
    edl_head_parameters = list(edl_head.parameters())

    # Combine the parameters into a single list
    all_parameters = gcn_parameters + edl_head_parameters

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')
    optimizer = optim.SGD(params= all_parameters, lr=args.lr, weight_decay=0.005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
    schedule2 = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = torch.nn.BCELoss()

    is_topk = True

    triplet_loss = TripletLoss(margin = 0.3)

    for epoch in range(args.max_epoch): #iterative training 
        print('Epoch: ', epoch + 1, '/', args.max_epoch)
        st = time.time()
        score, loss = train(train_loader, gcn, edl_head, optimizer, criterion, device, is_topk)

        #save model performance and scores 
        with open("train.txt", "a") as file:
            line = [epoch+1, loss, score]
            file.write(str(line)+"\n")

        if epoch % 2 == 0 and not epoch == 0:
            torch.save(gcn.state_dict(), './ckpt/' + args.model_name + '{}.pkl'.format(epoch)) # save every alternate weight file

        loss_m = []
        score = []
        with torch.no_grad():
            gcn.eval()
            edl_head.eval()

            for idx, (input, label) in enumerate(test_loader):
                pred = torch.zeros(0).to(device)
                seq_len = torch.sum(torch.max(torch.abs(input), dim=2)[0]>0, 1)
                input = input[:, :torch.max(seq_len), :]
                input, label = input.float().to(device), label.float().to(device)
                logits = gcn(input, seq_len)
                logits = logits.to(device)
                
                anchor_indices, positive_indices, negative_indices = sample_triplets(test_loader)

            # Extract instances based on sampled indices
                anchor = input[anchor_indices]
                positive = input[positive_indices]
                negative = input[negative_indices]
                
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                loss = triplet_loss(anchor, positive, negative)
                output = edl_head(logits)
                clsloss = CLAS(output, label, seq_len, criterion, device, is_topk)
    
                temp = torch.squeeze(output)
                sig = torch.sigmoid(temp)
                sig = torch.mean(sig, 1)
                pred = torch.cat((pred, sig))
                pred = list(pred.cpu().detach().numpy())
                total_loss = clsloss + loss

                label1 = list(label.cpu().detach().numpy())
                precision, recall, th = precision_recall_curve(label1, pred)
                pr_auc = auc(recall, precision)
                score.append(pr_auc)
                loss_m.append(total_loss.item())


            with open("validate.txt", "a") as file:
                line = [epoch+1, sum(loss_m)/len(loss_m), sum(score)/len(score)]
                file.write(str(line)+"\n")

            scheduler.step()


        # mean_val = sum(validate_auc) / len(validate_auc)
        # print('Validation: offline pr_auc:{2:.4}\n'.format(epoch, args.max_epoch, mean_val))


    torch.save(gcn.state_dict(), './ckpt/' + args.model_name + '.pkl') 