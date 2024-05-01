import os
import time 
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
import config
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from transformers import get_linear_schedule_with_warmup,AdamW
from dataset import Multimodal_Data
from losses import SupConLoss, SupConLoss_LABEL
from itertools import islice
import math
from sklearn.metrics import matthews_corrcoef

def bce_for_loss(logits,labels):
    loss=nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss*=labels.size(1)
    return loss

def compute_auc_score(logits,label):
    bz=logits.shape[0]
    logits=logits.cpu().numpy()
    label=label.cpu().numpy()
    auc=roc_auc_score(label,logits,average='weighted')*bz
    return auc

def compute_score(logits,labels):
    #print (logits,logits.shape)
    logits=torch.max(logits,1)[1]
    #print (logits)
    one_hot=torch.zeros(*labels.size()).cuda()
    one_hot.scatter_(1,logits.view(-1,1),1)
    score=one_hot * labels
    return score.sum().float()

def f1_score(y_pred, y_true, threshold=0.5):
    
    y_pred = (y_pred >= threshold).float()
    tp = torch.sum(y_true * y_pred, axis=0)
    fp = torch.sum((1 - y_true) * y_pred, axis=0)
    fn = torch.sum(y_true * (1 - y_pred), axis=0)
    precision = tp / (tp + fp + 1e-16)
    recall = tp / (tp + fn + 1e-16)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    return f1.mean()


def compute_scaler_score(logits,labels):
    #print (logits,logits.shape)
    logits=torch.max(logits,1)[1]
    labels=labels.squeeze(-1)
    score=(logits==labels).int()
    #print (score.sum(),labels,logits)
    return score.sum().float()


def log_hyperpara(logger,opt):
    dic = vars(opt)
    for k,v in dic.items():
        logger.write(k + ' : ' + str(v))
        
def train_for_epoch(opt,model,train_loader,test_loader):
    #initialization of saving path
    if opt.SAVE:
        model_path=os.path.join('../models',
                          '_'.join([opt.MODEL,opt.DATASET]))
        if os.path.exists(model_path)==False:
            os.mkdir(model_path)
    #multi-qeury configuration
    if opt.MULTI_QUERY:
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    #initialization of logger
    log_path=os.path.join(opt.DATASET)
    if os.path.exists(log_path)==False:
        os.mkdir(log_path)
    logger=utils.Logger(os.path.join(log_path,str(opt.SAVE_NUM)+'.txt'))  
    log_hyperpara(logger,opt)
    logger.write('Length of training set: %d, length of testing set: %d' %
                 (len(train_loader.dataset),len(test_loader.dataset)))
    
    if opt.MODEL=='pbm':
        #initialization of optimizer
        params = {}
        for n, p in model.named_parameters():
            if opt.FIX_LAYERS > 0:
                if 'encoder.layer' in n:
                    try:
                        layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                    except:
                        print(n)
                        raise Exception("")
                    if layer_num >= opt.FIX_LAYERS:
                        print('yes', n)
                        params[n] = p
                    else:
                        # p.require_grad = False
                        print('no ', n)
                elif 'embeddings' in n:
                    # p.require_grad = False
                    print('no ', n)
                else:
                    print('yes', n)
                    params[n] = p
            else:
                params[n] = p
        # params
        params2 = {}
        params3 = {}
        for n, p in model.named_parameters():
            if 'encoder.layer' in n:
                # low_lr_layer
                print('low_yes', n)
                params2[n] = p

            elif 'embeddings' in n:
                # p.require_grad = False
                print('low_yes ', n)
                params2[n] = p
            elif 'lm_head' in n:
                print('low_yes ', n)
                params2[n] = p
            else:
                print('normal_yes', n)
                params3[n] = p
        no_decay = ["bias", "LayerNorm.weight"]
        pretrain_layer = ["encoder"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                "weight_decay": opt.WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer_grouped_parameters2 = [
            {
                "params": [p for n, p in params2.items()],
                "weight_decay": 0.01,
                "lr": opt.PLR_RATE
            },
            {
                "params": [p for n, p in params3.items()],
                "weight_decay": 0.01,
                "lr": opt.LR_RATE
            },
        ]
    
        optim = AdamW(
            optimizer_grouped_parameters2,
            # lr=opt.LR_RATE,
            eps=opt.EPS,
        )
        # CL OPTIM
        cl_optim = AdamW(
            optimizer_grouped_parameters,
            lr=opt.CL_LR_RATE,
            eps=opt.CL_EPS,
        )
    else:
        params=model.parameters()
        optim=AdamW(params,
                    lr=1e-5,
                    eps=1e-8
                   )
    num_training_steps=len(train_loader) * opt.EPOCHS
    scheduler=get_linear_schedule_with_warmup(optim,
                                    num_warmup_steps=0,
                                    num_training_steps=num_training_steps
                                   )
    #strat training
    record_auc=[]
    record_acc=[]
    record_macro_f1=[]
    record_micro_f1=[]
    for epoch in range(opt.EPOCHS):
        model.train(True)
        total_loss=0.0
        scores=0.0
        for i,batch in enumerate(train_loader):
            #break
            cap=batch['cap_tokens'].long().cuda()
            label=batch['label'].float().cuda().view(-1,1)
            mask=batch['mask'].cuda()
            target=batch['target'].cuda()
            feat=None
            mask_pos=batch['mask_pos'].cuda()
            label_0_pos=batch['label_0_pos'].cuda()
            label_1_pos=batch['label_1_pos'].cuda()
            # CL PROCESS
            # features = model(cap,mask,mask_pos,feat,cl_forward=True) # bsz, 1024
            # features = features.unsqueeze(1) # bsz,1,1024
            # cl_loss = SupConLoss()(features, label, None)
            # cl_optim.zero_grad()
            # cl_loss.backward()
            # cl_optim.step()
            # cl_optim.zero_grad()
            # print('Epoch:', epoch, 'Iteration:', i, cl_loss.item())
            # normal, label_feature for contrast mask and label
            logits, features, label_feature = model(cap,mask,mask_pos,feat,label_0_pos,label_1_pos)
            features = features.unsqueeze(1)  # bsz,1,1024
            cl_loss = SupConLoss(margin=opt.MARGIN)(features, label, None)
            label_loss = SupConLoss_LABEL(margin=opt.MARGIN)(features,label, None, label_feature)

            loss=bce_for_loss(logits,target)
            if math.isnan(float(label_loss)):
                loss = loss + opt.CL_RATE * cl_loss
            else:
                loss = loss + opt.CL_RATE * cl_loss + opt.CL_LABEL_RATE*label_loss
            #print (logits,target)
            batch_score=compute_score(logits,target)
            scores+=batch_score
            
            print ('Epoch:',epoch,'Iteration:', i, loss.item(),batch_score)
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
            
            total_loss+=loss
            

        model.train(False)
        scores/=len(train_loader.dataset)

        eval_acc, eval_auc, eval_macro_f1, eval_micro_f1= eval_model(opt, model, tokenizer)
        

        record_auc.append(eval_auc)
        record_acc.append(eval_acc)
        record_macro_f1.append(eval_macro_f1)
        record_micro_f1.append(eval_micro_f1)

        logger.write('Epoch %d' % (epoch))
        logger.write('\ttrain_loss: %.2f, accuracy: %.2f' % (total_loss,
                                                                scores * 100.0))
        logger.write('\tevaluation auc: %.2f, accuracy: %.2f, macro_f1: %.2f, micro_f1: %.2f' % (eval_auc,
                                                                    eval_acc,
                                                                    eval_macro_f1,eval_micro_f1))

    max_idx = sorted(range(len(record_auc)),
                        key=lambda k: record_macro_f1[k] + record_acc[k],
                        reverse=True)[0]
    logger.write('Maximum epoch: %d' % (max_idx))
    logger.write('\tevaluation auc: %.2f, accuracy: %.2f, macro_f1: %.2f, micro_f1: %.2f' % (record_auc[max_idx],
                                                                record_acc[max_idx],
                                                                record_macro_f1[max_idx], 
                                                                record_micro_f1[max_idx]))
        
def eval_model(opt,model,tokenizer):
    scores=0.0
    auc=0.0
    test_set = Multimodal_Data(opt, tokenizer, opt.DATASET, 'test')
    test_loader = DataLoader(test_set,
                             opt.BATCH_SIZE,
                             shuffle=False,
                             num_workers=1)
    len_data = len(test_loader.dataset)
    print ('Length of test set:',len_data)
    total_logits=[]
    total_labels=[]
    for i,batch in enumerate(test_loader):
        with torch.no_grad():
            cap=batch['cap_tokens'].long().cuda()
            label=batch['label'].float().cuda().view(-1,1)
            mask=batch['mask'].cuda()
            target=batch['target'].cuda()
            feat=None
            if opt.MODEL=='pbm':
                mask_pos=batch['mask_pos'].cuda()
                label_0_pos = batch['label_0_pos'].cuda()
                label_1_pos = batch['label_1_pos'].cuda()
                logits,_,_=model(cap,mask,mask_pos,label_0_pos=label_0_pos,label_1_pos=label_1_pos)
                if opt.FINE_GRIND:
                    #attack=batch['attack'].cuda()#B,6
                    #logits=logits*attack
                    logits[:,1]=torch.sum(logits[:,1:],dim=1)
                    logits=logits[:,:2]
                
            elif opt.MODEL=='roberta':
                if opt.UNIMODAL==False:
                    feat=batch['feat'].cuda()
                logits=model(cap,mask,feat)
            
            batch_score=compute_score(logits,target)
            scores+=batch_score
            norm_logits=F.softmax(logits,dim=-1)[:,1].unsqueeze(-1)
            
            total_logits.append(norm_logits)
            total_labels.append(label)
    total_logits=torch.cat(total_logits,dim=0)
    total_labels=torch.cat(total_labels,dim=0)
    print (total_logits.shape,total_labels.shape)
    auc=compute_auc_score(total_logits,total_labels)
    macro_pre, macro_recall, macro_f1, _ = precision_recall_fscore_support(total_labels.cpu(), (total_logits >= 0.5).int().cpu(), average='macro')
    micro_pre, micro_recall, micro_f1, _ = precision_recall_fscore_support(total_labels.cpu(), (total_logits >= 0.5).int().cpu(), average='micro')

    #print (auc)
    return scores*100.0/len_data, auc*100.0/len_data, macro_f1*100, micro_f1*100
