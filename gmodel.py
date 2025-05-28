import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear
from transformers import AutoModelForCausalLM

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

class GraphOfThoughtComponent(nn.Module):
    """Graph of Thought reasoning component."""

    def __init__(self, model_name, hidden_size=None, n_paths=3):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.hidden_size = hidden_size or self.model.config.hidden_size
        self.projection = nn.Linear(self.model.config.hidden_size, self.hidden_size)
        self.n_paths = n_paths

    def forward(self, input_ids, attention_mask, labels=None):
        paths = []
        for _ in range(self.n_paths):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True
            )
            last_hidden = outputs.hidden_states[-1]
            weights = attention_mask.unsqueeze(-1).float()
            weighted_hidden = last_hidden * weights
            pooled = weighted_hidden.sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)
            projected = self.projection(pooled)
            paths.append((outputs.loss, projected))

        loss = torch.stack([p[0] for p in paths]).mean()
        embedding = torch.stack([p[1] for p in paths]).mean(dim=0)

        return loss, embedding

class GraphOfThoughtMAGDi(nn.Module):
    """
    Multi-Agent Graph Distillation model with Graph of Thought reasoning.
    """

    def __init__(self, model_name, gcn_in_channels, gcn_hidden_channels,
                 gcn_out_channels, alpha=1.0, beta=1.0, gamma=0.1, delta=0.5):
        super().__init__()

        self.got_component = GraphOfThoughtComponent(model_name)

        self.gcn = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels)

        self.mlp1 = Linear(self.got_component.hidden_size, self.got_component.hidden_size)
        self.mlp2 = Linear(self.got_component.hidden_size, 1)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def forward(self,
                input_ids, attention_mask, labels,
                pos_input_ids, pos_attention_mask, pos_labels,
                neg_input_ids, neg_attention_mask, neg_labels,
                graph):

        # Process main input (replaces decomposer and solver calls)
        main_loss, main_emb = self.got_component(input_ids, attention_mask, labels)

        # Process positive examples
        pos_loss, pos_emb = self.got_component(pos_input_ids, pos_attention_mask, pos_labels)

        # Process negative examples (no labels)
        _, neg_emb = self.got_component(neg_input_ids, neg_attention_mask, None)

        # Filter out padding in negative examples
        row_sums = neg_attention_mask.sum(dim=1)
        neg_mask = row_sums > 5  # Ignore negative padding
        if neg_mask.any():
            neg_mask = neg_mask.to(pos_emb.device)
            pos_emb = pos_emb[neg_mask]
            neg_emb = neg_emb[neg_mask]

        # Compute contrastive scores
        pos_h = torch.relu(self.mlp1(pos_emb))
        pos_score = torch.tanh(self.mlp2(pos_h))

        neg_h = torch.relu(self.mlp1(neg_emb))
        neg_score = torch.tanh(self.mlp2(neg_h))

        # Visualization of losses and scores during training
        if self.training:
            print("--- Visualization: Loss Components ---")
            print(f"Main Loss:      {main_loss.item():.4f}")
            print(f"Positive Loss:  {pos_loss.item():.4f}")
            print("Contrastive Scores (GoT pos vs neg):")
            for i in range(min(5, pos_score.size(0))):
                print(f"Sample {i}: pos_score = {pos_score[i].item():.4f}, neg_score = {neg_score[i].item():.4f}")

        mr_cri = nn.MarginRankingLoss(1.0, reduction='mean').to(pos_score.device)
        mr_loss = mr_cri(pos_score, neg_score, torch.ones_like(pos_score).to(pos_score.device))

        from torch_geometric.loader import DataLoader
        graph_loader = DataLoader(graph, batch_size=len(graph), shuffle=False, pin_memory=False, num_workers=0)
        graph_batch = next(iter(graph_loader))

        gcn_output, logits = self.gcn(graph_batch.x, graph_batch.edge_index)
        graph_batch.y = graph_batch.y.to(logits.device)
        ce_cri = nn.CrossEntropyLoss()
        node_loss = ce_cri(logits, graph_batch.y)

        # Alignment loss: here between main embedding and pos embedding
        alignment_loss = F.mse_loss(main_emb, pos_emb)

        # More detailed visualization of other losses
        if self.training:
            print(f"Node Classification Loss: {node_loss.item():.4f}")
            print(f"Margin Ranking Loss:      {mr_loss.item():.4f}")
            print(f"Alignment Loss:           {alignment_loss.item():.4f}")

        return (
            self.alpha * (main_loss + pos_loss),
            self.beta * node_loss,
            self.gamma * mr_loss,
            self.delta * alignment_loss
        )

class GraphOfThoughtMAGDiDataCollator:
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
