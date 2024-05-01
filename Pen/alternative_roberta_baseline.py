import torch
import torch.nn as nn
from transformers import RobertaModel

from classifier import SingleClassifier, SimpleClassifier
from rela_encoder import Rela_Module 

class RobertaBaseModel(nn.Module):
    def __init__(self,roberta,classifier,attention,proj_v):
        super(RobertaBaseModel, self).__init__()
        self.text_encoder=roberta
        self.classifier=classifier
        self.attention=attention
        self.proj_v=proj_v

    def forward(self,tokens,attention_mask,feat=None):
        output=self.text_encoder(tokens,
                                 attention_mask=attention_mask)[0]
        if feat==None:
            joint_repre=output[:,0]
        else:
            #print ('Multimodal')
            text_repre=output[:,0]
            vis=self.proj_v(feat)
            att_vis=self.attention(vis,output)
            joint_repre=torch.cat((att_vis,text_repre),dim=1)
        logits=self.classifier(joint_repre)
        return logits
        
    
def build_baseline(opt):  
    final_dim=2
    times=2-int(opt.UNIMODAL)
    """
    text_encoder=RobertaForSequenceClassification.from_pretrained(
        'roberta-large',
        num_labels=final_dim,
        output_attentions=False,
        output_hidden_states=True
    )
    """
    text_encoder=RobertaModel.from_pretrained('roberta-large')
    attention=Rela_Module(opt.ROBERTA_DIM,
                          opt.ROBERTA_DIM,opt.NUM_HEAD,opt.MID_DIM,
                          opt.TRANS_LAYER,
                          opt.FC_DROPOUT)
    classifier=SimpleClassifier(opt.ROBERTA_DIM*times,
                                opt.MID_DIM,final_dim,opt.FC_DROPOUT)
    proj_v=SingleClassifier(opt.FEAT_DIM,opt.ROBERTA_DIM,opt.FC_DROPOUT)
    return RobertaBaseModel(text_encoder,classifier,attention,proj_v)

    