import os
import json
import pickle as pkl
import numpy as np
import torch
import utils
from tqdm import tqdm
import config
import random

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data
    
def read_hdf5(path):
    data=h5py.File(path,'rb')
    return data

def read_csv(path):
    data=pd.read_csv(path)
    return data

def read_csv_sep(path):
    data=pd.read_csv(path,sep='\t')
    return data
    
def dump_pkl(path,info):
    pkl.dump(info,open(path,'wb'))  
    
def read_json(path):
    utils.assert_exits(path)
    data=json.load(open(path,'rb'))
    '''in anet-qa returns a list'''
    return data

def pd_pkl(path):
    data=pd.read_pickle(path)
    return data

def read_jsonl(path):
    total_info=[]
    with open(path,'rb')as f:
        d=f.readlines()
    for i,info in enumerate(d):
        data=json.loads(info)
        total_info.append(data)
    return total_info

class Roberta_Data():
    #mem, off, harm
    def __init__(self,opt,tokenizer,dataset,mode='train',few_shot_index=0):
        super(Roberta_Data,self).__init__()
        self.opt=opt
        self.tokenizer = tokenizer
        self.mode=mode
        self.dataset=dataset
        self.num_ans=self.opt.NUM_LABELS
        self.unimodal=self.opt.UNIMODAL
        if self.opt.FEW_SHOT:
            self.few_shot_index=str(few_shot_index)
            self.num_shots=self.opt.NUM_SHOTS
            print ('Few shot learning setting for Iteration:',self.few_shot_index)
            print ('Number of shots:',self.num_shots)
        
        self.length=self.opt.LENGTH
        self.pad_id=self.tokenizer.pad_token_id
        self.add_ent=self.opt.ADD_ENT
        self.add_dem=self.opt.ADD_DEM
        
        self.entries=self.load_entries(mode)
        if self.opt.DEBUG:
            self.entries=self.entries[:128]
        self.unimodal=self.opt.UNIMODAL
        print ('The length of the dataset for:',mode,'is:',len(self.entries))

    def load_entries(self,mode):
        #print ('Loading data from:',self.dataset)
        #only in training mode, in few-shot setting the loading will be different
        if self.opt.FEW_SHOT and mode=='train':
            path=os.path.join(self.opt.DATA,
                              'domain_splits',
                              self.opt.DATASET+'_'+str(self.num_shots)+'_'+self.few_shot_index+'.json')
        else:
            path=os.path.join(self.opt.DATA,
                              'domain_splits',
                              self.opt.DATASET+'_'+mode+'.json')
        data=read_json(path)
        cap_path=os.path.join(self.opt.CAPTION_PATH,
                              self.opt.DATASET+'_'+self.opt.PRETRAIN_DATA,
                              self.opt.IMG_VERSION+'_captions.pkl')
        captions=load_pkl(cap_path)
        entries=[]
        for k,row in enumerate(data):
            label=row['label']
            img=row['img']
            cap=captions[img.split('.')[0]][:-1]#remove the punctuation in the end
            sent=row['clean_sent']
            #remember the punctuations at the end of each sentence
            if self.unimodal:
                cap=cap+' . '+sent+' . '
            else:
                cap=cap+' . '+sent
            #whether using external knowledge
            if self.add_ent:
                cap=cap+' . '+row['entity']+' . '
            if self.add_dem:
                cap=cap+' . '+row['race']+' . '
            entry={
                'cap':cap.strip(),
                'label':label,
                'img':img
            }
            entries.append(entry)
        return entries
    
    def enc(self,text):
        return self.tokenizer.encode(text)
   
    def process_tokens(self,sent):
        tokens=self.enc(sent)
        mask=[]
        mask+= [1 for i in range(len(tokens))]
        while len(tokens) < self.length:
            tokens.append(self.pad_id)
            mask.append(0)
        if len(tokens) > self.length:
            tokens = tokens[:self.length]
            mask = mask[:self.length]
        return mask,tokens
            
    def __getitem__(self,index):
        #query item
        entry=self.entries[index]
        vid=entry['img']
        label=torch.tensor(entry['label'])
        target=torch.from_numpy(np.zeros((self.num_ans),dtype=np.float32))
        target[label]=1.0
        
        mask,tokens=self.process_tokens(entry['cap'])       
        mask=torch.Tensor(mask)
        tokens=torch.Tensor(tokens)
        
        batch={
            'mask':mask,
            'img':vid,
            'target':target,
            'cap_tokens':tokens,
            'label':label
        }
        if self.unimodal==False:
            if self.dataset=='mem':
                info=np.load(os.path.join(self.opt.DATA,
                                          'multimodal-hate',
                                          'mem',
                                          'faster_hatefulmem_clean_36',
                                          vid.split('.')[0]+'.npy'),
                             allow_pickle=True).item()
            else:
                info=np.load(os.path.join(self.opt.DATA,
                                          'multimodal-hate',
                                          'harm',
                                          'clean_features',
                                          vid.split('.')[0]+'.npy'),
                             allow_pickle=True).item()
            feat=info['features']
            feat=torch.from_numpy(feat)
            batch['feat']=feat
        return batch
        
    def __len__(self):
        return len(self.entries)
    
