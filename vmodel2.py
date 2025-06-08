import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.data import Batch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

class GCN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)
    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.gcn2(x, edge_index)
        return x, F.log_softmax(x, dim=1)

class VanillaCoTMAGDi(nn.Module):
    def __init__(self, model_name, gcn_in_channels, gcn_hidden_channels, gcn_out_channels, alpha=1.0, beta=1.0, gamma=0.1):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.gcn = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, pos, neg, graph):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']

        pos_input_ids = pos['input_ids']
        pos_attention_mask = pos['attention_mask']
        pos_labels = pos['labels']

        neg_input_ids = neg['input_ids']
        neg_attention_mask = neg['attention_mask']

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss_main = outputs.loss

        pos_outputs = self.model(input_ids=pos_input_ids, attention_mask=pos_attention_mask, labels=pos_labels)
        loss_pos = pos_outputs.loss

        neg_outputs = self.model(input_ids=neg_input_ids, attention_mask=neg_attention_mask)
        neg_logits = neg_outputs.logits

        pos_emb = pos_outputs.hidden_states[-1][:,0,:]
        neg_emb = neg_outputs.hidden_states[-1][:,0,:]

        margin_loss = nn.MarginRankingLoss(margin=1.0)(pos_emb, neg_emb, torch.ones(pos_emb.size(0), device=pos_emb.device))

        graph_batch = graph
        gcn_out, logits = self.gcn(graph_batch.x, graph_batch.edge_index)
        ce_loss = nn.CrossEntropyLoss()(logits, graph_batch.y.to(logits.device))

        total_loss = self.alpha * (loss_main + loss_pos) + self.beta * ce_loss + self.gamma * margin_loss
        return total_loss, loss_main, loss_pos, ce_loss, margin_loss

class VanillaCoTMAGDiCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, batch):
        def pad_tensors(samples):
            input_ids = [torch.tensor(s['input_ids']) for s in samples]
            attention_mask = [torch.tensor(s['attention_mask']) for s in samples]
            labels = [torch.tensor(s['labels']) for s in samples] if 'labels' in samples[0] else None
            input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
            if labels is not None:
                labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
            else:
                labels_padded = None
            return {
                'input_ids': input_ids_padded,
                'attention_mask': attention_mask_padded,
                'labels': labels_padded
            }
        inputs = [item['inputs'] for item in batch]
        pos_samples = [item['pos'] for item in batch]
        neg_samples = [item['neg'] for item in batch]
        graphs = [item['graph'] for item in batch]
        inputs_batch = pad_tensors(inputs)
        pos_batch = pad_tensors(pos_samples)
        neg_batch = pad_tensors(neg_samples)
        graph_batch = Batch.from_data_list(graphs)
        return inputs_batch, pos_batch, neg_batch, graph_batch
