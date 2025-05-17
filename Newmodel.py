import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear
from transformers import AutoModelForCausalLM, AutoTokenizer

class GCN(torch.nn.Module):
    """Graph Convolutional Network for processing multi-agent interaction graphs."""
    
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

class SocraticDecomposer(nn.Module):
    """Problem decomposer component of the Socratic model."""
    
    def __init__(self, model_name, hidden_size=None):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.hidden_size = hidden_size or self.model.config.hidden_size
        self.projection = nn.Linear(self.model.config.hidden_size, self.hidden_size)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        # Get the last hidden state
        last_hidden = outputs.hidden_states[-1]
        
        # Create weighted representation based on attention mask
        weights = attention_mask.unsqueeze(-1).float()
        weighted_hidden = last_hidden * weights
        pooled = weighted_hidden.sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)
        
        # Project to desired dimension
        projected = self.projection(pooled)
        
        return outputs.loss, projected

class SocraticSolver(nn.Module):
    """Subproblem solver component of the Socratic model."""
    
    def __init__(self, model_name, hidden_size=None):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.hidden_size = hidden_size or self.model.config.hidden_size
        self.projection = nn.Linear(self.model.config.hidden_size, self.hidden_size)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        # Get the last hidden state
        last_hidden = outputs.hidden_states[-1]
        
        # Create weighted representation based on attention mask
        weights = attention_mask.unsqueeze(-1).float()
        weighted_hidden = last_hidden * weights
        pooled = weighted_hidden.sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)
        
        # Project to desired dimension
        projected = self.projection(pooled)
        
        return outputs.loss, projected

class SocraticMAGDi(nn.Module):
    """
    Multi-Agent Graph Distillation model with Socratic questioning components.
    Combines a decomposer and solver with graph-based knowledge distillation.
    """
    
    def __init__(self, decomposer_name, solver_name, gcn_in_channels, gcn_hidden_channels, 
                 gcn_out_channels, alpha=1.0, beta=1.0, gamma=0.1, delta=0.5):
        super().__init__()
        
        # Socratic components
        self.decomposer = SocraticDecomposer(decomposer_name)
        self.solver = SocraticSolver(solver_name)
        
        # Graph component
        self.gcn = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels)
        
        # Projection layers
        self.mlp1 = Linear(self.decomposer.hidden_size, self.decomposer.hidden_size)
        self.mlp2 = Linear(self.decomposer.hidden_size, 1)
        
        # Loss weights
        self.alpha = alpha  # Weight for language modeling loss
        self.beta = beta    # Weight for node classification loss
        self.gamma = gamma  # Weight for contrastive loss
        self.delta = delta  # Weight for decomposer-solver alignment loss
    
    def forward(self, decomposer_input_ids, decomposer_attention_mask, decomposer_labels,
                solver_input_ids, solver_attention_mask, solver_labels,
                pos_input_ids, pos_attention_mask, pos_labels,
                neg_input_ids, neg_attention_mask, neg_labels, graph):
        
        # Process decomposer inputs
        decomposer_loss, decomposer_emb = self.decomposer(
            decomposer_input_ids, decomposer_attention_mask, decomposer_labels
        )
        
        # Process solver inputs
        solver_loss, solver_emb = self.solver(
            solver_input_ids, solver_attention_mask, solver_labels
        )
        
        # Process positive examples
        pos_loss, pos_emb = self.solver(
            pos_input_ids, pos_attention_mask, pos_labels
        )
        
        # Process negative examples
        _, neg_emb = self.solver(
            neg_input_ids, neg_attention_mask, None
        )
        
        # Filter out padding in negative examples
        row_sums = neg_attention_mask.sum(dim=1)
        neg_mask = row_sums > 5  # Ignore negative padding
        
        if neg_mask.any():
            neg_mask = neg_mask.to(pos_emb.device)
            pos_emb = pos_emb[neg_mask]
            neg_emb = neg_emb[neg_mask]
        
        # Calculate contrastive scores
        pos_h = torch.relu(self.mlp1(pos_emb))
        pos_score = torch.tanh(self.mlp2(pos_h))
        
        neg_h = torch.relu(self.mlp1(neg_emb))
        neg_score = torch.tanh(self.mlp2(neg_h))
        
        # Margin ranking loss
        mr_cri = torch.nn.MarginRankingLoss(1.0, reduction='mean').to(pos_score.device)
        mr_loss = mr_cri(pos_score, neg_score, torch.ones_like(pos_score).to(pos_score.device))
        
        # Process graph data
        from torch_geometric.loader import DataLoader
        graph_loader = DataLoader(graph, batch_size=len(graph), shuffle=False, pin_memory=False, num_workers=0)
        graph_batch = next(iter(graph_loader))
        
        # Graph node classification loss
        gcn_output, logits = self.gcn(graph_batch.x, graph_batch.edge_index)
        graph_batch.y = graph_batch.y.to(logits.device)
        ce_cri = torch.nn.CrossEntropyLoss()
        node_loss = ce_cri(logits, graph_batch.y)
        
        # Decomposer-solver alignment loss
        alignment_loss = F.mse_loss(decomposer_emb, solver_emb)
        
        # Return individual loss components
        return (
            self.alpha * (decomposer_loss + solver_loss + pos_loss), 
            self.beta * node_loss, 
            self.gamma * mr_loss,
            self.delta * alignment_loss
        )

class SocraticMAGDiDataCollator:
    """Data collator for SocraticMAGDi model training."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        """
        Process a batch of examples for training.
        Each item in batch should be a dictionary with graph and examples.
        """
        decomposer_input_ids = torch.stack([item["decomposer_input_ids"] for item in batch])
        decomposer_attention_mask = torch.stack([item["decomposer_attention_mask"] for item in batch])
        decomposer_labels = torch.stack([item["decomposer_labels"] for item in batch])
        
        solver_input_ids = torch.stack([item["solver_input_ids"] for item in batch])
        solver_attention_mask = torch.stack([item["solver_attention_mask"] for item in batch])
        solver_labels = torch.stack([item["solver_labels"] for item in batch])
        
        pos_input_ids = torch.stack([item["pos_input_ids"] for item in batch])
        pos_attention_mask = torch.stack([item["pos_attention_mask"] for item in batch])
        pos_labels = torch.stack([item["pos_labels"] for item in batch])
        
        neg_input_ids = torch.stack([item["neg_input_ids"] for item in batch])
        neg_attention_mask = torch.stack([item["neg_attention_mask"] for item in batch])
        neg_labels = torch.stack([item["neg_labels"] for item in batch])
        
        graphs = [item["graph"] for item in batch]
        
        return {
            "decomposer_input_ids": decomposer_input_ids,
            "decomposer_attention_mask": decomposer_attention_mask,
            "decomposer_labels": decomposer_labels,
            "solver_input_ids": solver_input_ids,
            "solver_attention_mask": solver_attention_mask,
            "solver_labels": solver_labels,
            "pos_input_ids": pos_input_ids,
            "pos_attention_mask": pos_attention_mask,
            "pos_labels": pos_labels,
            "neg_input_ids": neg_input_ids,
            "neg_attention_mask": neg_attention_mask,
            "neg_labels": neg_labels,
            "graph": graphs
        }
