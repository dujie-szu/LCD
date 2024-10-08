import torch

def to_onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
    return onehot

###### inter_intra_loss  矩阵运算  after 230928  不同类只用拉远对应的两类  sim
def _intra_inter_loss(self, outputs, labels, sim_all, epoch, T=2, diff_scale=0.001, same_scale=1.0):  
    eps = 1e-6
    # Softmax_t
    predicts = outputs["logits"]
    predicts = torch.softmax(predicts / T, dim=1)    
    n_classes = predicts.size(1)
    onehot_targets = to_onehot(labels, n_classes)
    predlist = []
    onehot_targets_list = []
    
    ##### compute similiarity per class, batch
    sorted_pred, indices = torch.sort(predicts, dim=1, descending=True)
    sim_batch = torch.zeros((predicts.size(1), predicts.size(1))).to(self._device)  ### sim_batch: [C, C]
    for ind in range(predicts.size(0)):
        if(labels[ind] == indices[ind][0] and self._n_classes > 1):     ## labels[ind]即真实类别，indices[ind][0]即预测的类别
            j = indices[ind][1]        ## labels[ind][1]即预测中第二高的类
            sim_batch[labels[ind]][j] += 1 
    #####
    
    pred_ = predicts.chunk(predicts.size(1),1)
    for i in range(len(pred_)):
        predlist.append(pred_[i].repeat(1,predicts.size(0))) 
    pred = torch.stack(predlist, dim=0)     # 变成三维
    pred_T = pred.permute(0,2,1)
    
    
    onehot_ = onehot_targets.chunk(onehot_targets.size(1), 1)
    for i in range(len(onehot_)):
        onehot_targets_list.append(onehot_[i].repeat(1, onehot_targets.size(0)))
    onehots = torch.stack(onehot_targets_list, dim=0)
    onehots_T = onehots.permute(0, 2, 1)
    inter_parm = torch.triu(onehots!=onehots_T, 1).int()
    
    
    labels = labels.unsqueeze(-1).repeat(1,predicts.size(0))
    labels_t = labels.T
    
    
    #### sim_b
    sim_b = torch.zeros((labels.size(0), labels.size(0))).to(self._device) 
    sim_b[:] = sim_all[labels, labels_t]
    sim_b_diag = torch.triu(sim_b)      ## new sim_b_diag: 只取上三角
    if epoch == 0 or self._n_classes == 1:
        sim_b_diag = torch.triu(torch.ones((labels.size(0), labels.size(0)))).to(self._device)  ## new sim_b_diag: 只取上三角 
    
    ####
        
    loss = torch.sum(torch.triu(pred * torch.log(pred/(pred_T+eps)+eps), 1), axis=0)
    kl_same_loss= torch.sum(torch.abs(loss*(labels==labels.T)))
    kl_diff_loss = torch.sum(torch.abs(torch.sum(torch.triu(pred * torch.log(pred/(pred_T+eps)+eps)*inter_parm, 1), axis=0)*sim_b_diag))

    kl_same_count = torch.sum(torch.abs(torch.triu(labels==labels.T, 1)))
    kl_diff_count = torch.sum(torch.abs(torch.triu(labels!=labels.T, 1)))
    
    
    if kl_same_count != 0:
        kl_same_loss /= kl_same_count
    if kl_diff_count != 0:
        kl_diff_loss /= kl_diff_count 
    

    if kl_diff_loss != 0:
        kl_diff_loss = 1 / (kl_diff_loss+eps) * diff_scale
    kl_same_loss = kl_same_loss*same_scale      ###231024
    
    return kl_same_loss + kl_diff_loss, sim_batch