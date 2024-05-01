import torch
import torch.nn as nn
from transformers import RobertaForMaskedLM

class LSTMModel(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda()
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class Cross_LMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, hidden_size=1024):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.activation = nn.GELU()
        self.decoder = nn.Linear(hidden_size, 50265)
        self.bias = nn.Parameter(torch.zeros(50265))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)
        return x

class Cross_mixture(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, hidden_size=1024):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense_1 = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, mask_features, label_feature):
        x = self.dense1(mask_features.unsqueeze(-2))
        y = self.dense2(label_feature.unsqueeze(-2))
        score = self.activation(self.dense_1(x+y))
        cross_mixture = (torch.matmul(score,x) + torch.matmul(1-score,y)).squeeze(-2)

        return cross_mixture

class RobertaPromptModel(nn.Module):
    def __init__(self,label_list):
        super(RobertaPromptModel, self).__init__()
        self.label_word_list=label_list
        self.roberta = RobertaForMaskedLM.from_pretrained('roberta-large')
        self.bad_network = nn.Linear(2048, 1024)
        self.good_network = nn.Linear(2048, 1024)
        self.cross = Cross_mixture()
        self.query = LSTMModel()
        self.support1 = LSTMModel()
        self.support2 = LSTMModel()
        # self.cross_lmhead = Cross_LMHead()
        # self.Softmax = nn.Softmax(dim=-1)


    def forward(self,tokens,attention_mask,mask_pos,feat=None,label_0_pos=None,label_1_pos=None):
        batch_size = tokens.size(0)
        #the position of word for prediction
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()
        if label_0_pos is not None:
            label_0_pos = label_0_pos.squeeze()
        if label_1_pos is not None:
            label_1_pos = label_1_pos.squeeze()
            # label_loss_v1
        # label_embedding = self.roberta.roberta.embeddings(torch.tensor(self.label_word_list).unsqueeze(0).cuda()).detach()
        # label_embedding = label_embedding.squeeze(0)
        out = self.roberta.roberta(tokens,
                           attention_mask)
        sequence_output = out[0]
        sequence_mask_output = sequence_output[torch.arange(batch_size), mask_pos]

        # label_cross_mix
        sequence_label_0_output = sequence_output[torch.arange(batch_size), label_0_pos]
        sequence_label_1_output = sequence_output[torch.arange(batch_size), label_1_pos]
        # label_embedding = torch.cat((sequence_label_0_output.unsqueeze(-2),sequence_label_1_output.unsqueeze(-2)),dim=-2)
        query_info = self.query(sequence_output[:,1:mask_pos[0]-2])
        support_label_0 = self.support1(sequence_output[:,label_1_pos[0]+4:label_0_pos[0]-2])
        support_label_1 = self.support2(sequence_output[:,mask_pos[0]+4:label_1_pos[0]-2])

        label_embedding = torch.cat((sequence_label_0_output.unsqueeze(-2), sequence_label_1_output.unsqueeze(-2)),
                                    dim=-2)


        sequence_label_0_output = sequence_label_0_output + support_label_0
        sequence_label_1_output = sequence_label_1_output + support_label_1



        # v1
        # sequence_mix_mask_output = torch.cat((sequence_mask_output,sequence_label_0_output,sequence_label_1_output),dim=-1)
        # mask_mixture = self.fc(sequence_mix_mask_output)
        # mask_mixture_all = self.cross(sequence_mask_output, mask_mixture)

        # v2
        # mask_label0_mixture = self.cross(sequence_mask_output, sequence_label_0_output)
        # mask_label1_mixture = self.cross(sequence_mask_output, sequence_label_1_output)
        #
        # mask_label_mixture = self.cross2(mask_label0_mixture, mask_label1_mixture)
        #
        # mask_mixture_all = sequence_mask_output + mask_label_mixture

        # v3
        query_mask = sequence_mask_output + query_info
        sequence_mix_mask_label0 = torch.cat((query_mask, sequence_label_0_output), dim=-1)
        sequence_mix_mask_label1 = torch.cat((query_mask, sequence_label_1_output), dim=-1)

        label0_prompt = self.bad_network(sequence_mix_mask_label0)
        label1_prompt = self.good_network(sequence_mix_mask_label1)

        istruction = self.cross(label0_prompt, label1_prompt)
        mask_mixture_all = istruction + sequence_mask_output


        # v4 double decode


        # prediction_scores =self.roberta.lm_head(sequence_output)
        prediction_scores = self.roberta.lm_head(mask_mixture_all)
        prediction_mask_scores = prediction_scores

        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:,
                                                 self.label_word_list[label_id]
                                                ].unsqueeze(-1))
            #print(prediction_mask_scores[:, self.label_word_list[label_id]].shape)
        logits = torch.cat(logits, -1)

        # v4 append and v5 double_lmhead
        prediction_scores_label0 = self.roberta.lm_head(label0_prompt+sequence_mask_output)
        prediction_mask_scores_label0 = prediction_scores_label0
        mix_logits_label0 = []
        for label_id in range(len(self.label_word_list)):
            mix_logits_label0.append(prediction_mask_scores_label0[:,
                                                 self.label_word_list[label_id]
                                                ].unsqueeze(-1))
        mix_logits_label0 = torch.cat(mix_logits_label0, -1)

        # v6 label0 and lable1 prompt
        prediction_scores_label1 = self.roberta.lm_head(label1_prompt+sequence_mask_output)
        prediction_mask_scores_label1 = prediction_scores_label1
        mix_logits_label1 = []
        for label_id in range(len(self.label_word_list)):
            mix_logits_label1.append(prediction_mask_scores_label1[:,
                              self.label_word_list[label_id]
                              ].unsqueeze(-1))
        mix_logits_label1 = torch.cat(mix_logits_label1, -1)

        # v7 infomation
        prediction_scores_query_info = self.roberta.lm_head(query_mask)
        prediction_mask_scores_query = prediction_scores_query_info
        mix_logits_query = []
        for label_id in range(len(self.label_word_list)):
            mix_logits_query.append(prediction_mask_scores_query[:,
                                     self.label_word_list[label_id]
                                     ].unsqueeze(-1))
        mix_logits_query = torch.cat(mix_logits_query, -1)

        logits_max_value, logits_max_idx = torch.topk(logits, 1, dim=1)
        label_0_max_value, label_0_max_idx = torch.topk(mix_logits_label0, 1, dim=1)
        label_1_max_value, label_1_max_idx = torch.topk(mix_logits_label1, 1, dim=1)
        query_max_value, query_max_idx = torch.topk(mix_logits_query, 1, dim=1)

        output_tensor_logits = torch.zeros_like(logits)
        output_tensor_logits.scatter_(1, logits_max_idx, 1)

        output_tensor_label0 = torch.zeros_like(mix_logits_label0)
        output_tensor_label0.scatter_(1, label_0_max_idx, 1)

        output_tensor_label1 = torch.zeros_like(mix_logits_label1)
        output_tensor_label1.scatter_(1, label_1_max_idx, 1)

        output_tensor_query = torch.zeros_like(mix_logits_query)
        output_tensor_query.scatter_(1, query_max_idx, 2)

        # logits = output_tensor_logits + output_tensor_label0 + output_tensor_label1 + mix_logits_query
        logits = logits + mix_logits_label0 + mix_logits_label1 + mix_logits_query
        #print(logits.shape)
        return logits, sequence_mask_output, label_embedding


        
    
def build_baseline(opt,label_list):  
    print (label_list)
    return RobertaPromptModel(label_list)

    
