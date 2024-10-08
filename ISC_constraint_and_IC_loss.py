import torch
import torch.nn as nn


class ISC_constraint_and_IC_loss(nn.Module):
    def __init__(self, n_classes, device):
        super().__init__()
        self.n_classes = n_classes
        self.device = device

    def forward(self, predicts, labels, sim_all, epoch, T, mu, eta):  
        """
        Args:
            predicts:  the outputs, shape (B, C).
            labels: the ground-truth, shape (B).
            sim_all: Semantic similarity between classes.
            epoch: Current epoch.
            T: The temperature scaling parameter.
            mu: Hyper-parameter.
            eta: Hyper-parameter.
        Returns:
            LCDLoss and sim_batch.
        """
        eps = 1e-6
        ##### 1. softmax_t
        predicts = torch.softmax(predicts / T, dim=1)    
        n_classes = predicts.size(1)
        onehot_targets = self.to_onehot(labels, n_classes)
        predlist = []
        onehot_targets_list = []
        
        ##### 2. compute similiarity per class of a batch
        sorted_pred, indices = torch.sort(predicts, dim=1, descending=True)
        sim_batch = torch.zeros((predicts.size(1), predicts.size(1))).to(self.device)  ### sim_batch: [C, C]
        for ind in range(predicts.size(0)):
            if(labels[ind] == indices[ind][0] and self.n_classes > 1):     ## labels[ind]: the ground-truth, indices[ind][0]: the predicted class
                j = indices[ind][1]        ## labels[ind][1]: the second highest class in the predicts
                sim_batch[labels[ind]][j] += 1 
        
        
        ##### 3. the efficient batch-based strategy
        pred_ = predicts.chunk(predicts.size(1),1)
        for i in range(len(pred_)):
            predlist.append(pred_[i].repeat(1,predicts.size(0))) 
        pred = torch.stack(predlist, dim=0)     
        pred_T = pred.permute(0,2,1)
        
        onehot_ = onehot_targets.chunk(onehot_targets.size(1), 1)
        for i in range(len(onehot_)):
            onehot_targets_list.append(onehot_[i].repeat(1, onehot_targets.size(0)))
        onehots = torch.stack(onehot_targets_list, dim=0)
        onehots_T = onehots.permute(0, 2, 1)
        inter_parm = torch.triu(onehots!=onehots_T, 1).int()
        
        labels = labels.unsqueeze(-1).repeat(1,predicts.size(0))
        labels_t = labels.T
        
        
        ##### 4. compute sim_b
        sim_b = torch.zeros((labels.size(0), labels.size(0))).to(self.device) 
        sim_b[:] = sim_all[labels, labels_t]
        sim_b_diag = torch.triu(sim_b)      
        if epoch == 0 or n_classes == 1:
            sim_b_diag = torch.triu(torch.ones((labels.size(0), labels.size(0)))).to(self.device)  
        

        ##### 5. compute ISC_constraint and IC_loss  
        loss = torch.sum(torch.triu(pred * torch.log(pred/(pred_T+eps)+eps), 1), axis=0)
        IC_loss= torch.sum(torch.abs(loss*(labels==labels.T)))
        ISC_constraint = torch.sum(torch.abs(torch.sum(torch.triu(pred * torch.log(pred/(pred_T+eps)+eps)*inter_parm, 1), axis=0)*sim_b_diag))

        same_count = torch.sum(torch.abs(torch.triu(labels==labels.T, 1)))
        diff_count= torch.sum(torch.abs(torch.triu(labels!=labels.T, 1)))
        
        if same_count != 0:
            IC_loss /= same_count
        if diff_count!= 0:
            ISC_constraint /= diff_count
        
        if ISC_constraint != 0:
            ISC_constraint = 1 / (ISC_constraint + eps) * mu
        IC_loss = IC_loss * eta      
        
        return IC_loss + ISC_constraint, sim_batch
    

    def to_onehot(targets, n_classes):
        onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
        onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
        return onehot