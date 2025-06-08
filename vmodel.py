import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear
from transformers import AutoModelForCausalLM

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.conv2(x, edge_index)
        return x, F.log_softmax(x, dim=1)

class VanillaCoTMAGDi(nn.Module):
    def __init__(self, model_name, gcn_in_channels, gcn_hidden_channels, gcn_out_channels,
                 alpha=1.0, beta=1.0, gamma=0.1):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(model_name)
        self.gcn = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels)
        self.alpha = alpha  # LM loss weight
        self.beta = beta    # GCN node classification loss weight
        self.gamma = gamma  # Contrastive loss weight
        
        self.mlp_pos_neg = nn.Sequential(
            Linear(gcn_out_channels, gcn_out_channels),
            nn.ReLU(),
            Linear(gcn_out_channels, 1),
            nn.Tanh()
        )

    def forward(self, inputs, pos_samples, neg_samples, graph_batch):
        # Language modeling loss
        lm_outputs = self.lm(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs.get("labels", None),
            output_hidden_states=True
        )
        lm_loss = lm_outputs.loss
        
        # Positive sample embeddings
        pos_outputs = self.lm(
            input_ids=pos_samples["input_ids"],
            attention_mask=pos_samples["attention_mask"],
            labels=pos_samples.get("labels", None),
            output_hidden_states=True
        )
        pos_emb = pos_outputs.hidden_states[-1].mean(dim=1)
        
        # Negative sample embeddings
        neg_outputs = self.lm(
            input_ids=neg_samples["input_ids"],
            attention_mask=neg_samples["attention_mask"],
            output_hidden_states=True
        )
        neg_emb = neg_outputs.hidden_states[-1].mean(dim=1)
        
        # Contrastive scoring
        pos_score = self.mlp_pos_neg(pos_emb)
        neg_score = self.mlp_pos_neg(neg_emb)
        
        margin_loss_fn = nn.MarginRankingLoss(margin=1.0)
        target = torch.ones_like(pos_score).to(pos_score.device)
        contrastive_loss = margin_loss_fn(pos_score, neg_score, target)
        
        # GCN forward & node classification loss
        gcn_out, logits = self.gcn(graph_batch.x, graph_batch.edge_index)
        graph_batch.y = graph_batch.y.to(logits.device)
        node_loss_fn = nn.CrossEntropyLoss()
        node_loss = node_loss_fn(logits, graph_batch.y)
        
        total_loss = self.alpha * lm_loss + self.beta * node_loss + self.gamma * contrastive_loss
        
        return total_loss, (lm_loss, node_loss, contrastive_loss)

class VanillaCoTMAGDiDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        pos_input_ids = torch.stack([item["pos_input_ids"] for item in batch])
        pos_attention_mask = torch.stack([item["pos_attention_mask"] for item in batch])
        pos_labels = torch.stack([item["pos_labels"] for item in batch])

        neg_input_ids = torch.stack([item["neg_input_ids"] for item in batch])
        neg_attention_mask = torch.stack([item["neg_attention_mask"] for item in batch])
        neg_labels = torch.stack([item["neg_labels"] for item in batch])

        graphs = [item["graph"] for item in batch]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pos_input_ids": pos_input_ids,
            "pos_attention_mask": pos_attention_mask,
            "pos_labels": pos_labels,
            "neg_input_ids": neg_input_ids,
            "neg_attention_mask": neg_attention_mask,
            "neg_labels": neg_labels,
            "graph": graphs
        }
