import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

# In GCN class in smodel.py [1]
class GCN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)
        # Add a linear projection for the residual connection if dimensions differ
        self.residual_proj = None
        if dim_in != dim_out:
            self.residual_proj = Linear(dim_in, dim_out)

    def forward(self, x, edge_index):
        residual = x
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.gcn2(x, edge_index)

        # Add residual connection
        if self.residual_proj:
            residual = self.residual_proj(residual)
        x += residual

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
    def forward(self, decomposer, solver, pos, neg, graph):
        """Forward pass with properly batched tensors."""
        
        # Access tensors directly from the dictionaries
        decomposer_input_ids = decomposer['input_ids']
        decomposer_attention_mask = decomposer['attention_mask']
        decomposer_labels = decomposer['labels']
    
        solver_input_ids = solver['input_ids']
        solver_attention_mask = solver['attention_mask']
        solver_labels = solver['labels']
    
        pos_input_ids = pos['input_ids']
        pos_attention_mask = pos['attention_mask']
        pos_labels = pos['labels']
    
        neg_input_ids = neg['input_ids']
        neg_attention_mask = neg['attention_mask']
        neg_labels = neg['labels']
    
        # Component computations
        decomposer_loss, decomposer_emb = self.decomposer(
            decomposer_input_ids, decomposer_attention_mask, decomposer_labels
        )
        solver_loss, solver_emb = self.solver(
            solver_input_ids, solver_attention_mask, solver_labels
        )
        
        # Dummy value masking
        pos_mask = (pos_input_ids.sum(dim=1) != 0).float().unsqueeze(-1)
        neg_mask = (neg_input_ids.sum(dim=1) != 0).float().unsqueeze(-1)
    
        # Contrastive learning with mask
        pos_loss, pos_emb = self.solver(pos_input_ids, pos_attention_mask, pos_labels)
        _, neg_emb = self.solver(neg_input_ids, neg_attention_mask, neg_labels)
        
        pos_h = torch.relu(self.mlp1(pos_emb)) * pos_mask
        pos_score = torch.tanh(self.mlp2(pos_h))
        neg_h = torch.relu(self.mlp1(neg_emb)) * neg_mask
        neg_score = torch.tanh(self.mlp2(neg_h))
        
        mr_cri = torch.nn.MarginRankingLoss(1.0, reduction='mean')
        mr_loss = mr_cri(pos_score, neg_score, torch.ones_like(pos_score))
        
        # Node classification with valid node masking
        valid_nodes = (graph.x.sum(dim=1) != 0).float()
        gcn_output, logits = self.gcn(graph.x, graph.edge_index)
        node_loss = (F.cross_entropy(logits, graph.y, reduction='none') * valid_nodes).mean()
    
        # Alignment loss with valid example masking
        valid_decomposer = (decomposer_input_ids.sum(dim=1) != 0).float().unsqueeze(-1)
        valid_solver = (solver_input_ids.sum(dim=1) != 0).float().unsqueeze(-1)
        alignment_loss = F.mse_loss(
            decomposer_emb * valid_decomposer, 
            solver_emb * valid_solver
        )
    
        # Final loss calculation
        lm_combined = self.alpha * (decomposer_loss + solver_loss + pos_loss)
        node_weighted = self.beta * node_loss
        mr_weighted = self.gamma * mr_loss
        alignment_weighted = self.delta * alignment_loss
    
        total_loss = lm_combined + node_weighted + mr_weighted + alignment_weighted
    
        return (lm_combined, node_weighted, mr_weighted, alignment_weighted)
from transformers import PreTrainedTokenizerBase
from torch_geometric.data import Batch
class SocraticMAGDiDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _collate_component(self, batch, component, input_keys):
        texts = []
        labels = []
        for item in batch:
            ex = item[component][0]
            # For decomposer/solver, use completion as both input and label (causal LM)
            if component in ["decomposer", "solver"]:
                # If prompt and completion are both needed, concatenate them here
                # Otherwise, just use completion
                input_ids = ex[input_keys[2]]
                texts.append(self.tokenizer.decode(input_ids) if isinstance(input_ids, (list, tuple)) else self.tokenizer.decode(input_ids.tolist()))
                labels.append(self.tokenizer.decode(input_ids) if isinstance(input_ids, (list, tuple)) else self.tokenizer.decode(input_ids.tolist()))
            else:
                # For pos/neg, use input_ids and labels as is
                input_ids = ex[input_keys[0]]
                label_ids = ex[input_keys[2]]
                texts.append(self.tokenizer.decode(input_ids) if isinstance(input_ids, (list, tuple)) else self.tokenizer.decode(input_ids.tolist()))
                labels.append(self.tokenizer.decode(label_ids) if isinstance(label_ids, (list, tuple)) else self.tokenizer.decode(label_ids.tolist()))
        # Tokenize and pad
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Tokenize and pad labels separately to ensure same length
        tokenized_labels = self.tokenizer(
            labels,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Set labels to the tokenized label ids
        tokenized["labels"] = tokenized_labels["input_ids"]
        return tokenized

    def __call__(self, batch):
        processed_batch = {}
        # Collate each component
        processed_batch["decomposer"] = self._collate_component(
            batch, "decomposer", ["prompt_input_ids", "prompt_attention_mask", "completion_input_ids"]
        )
        processed_batch["solver"] = self._collate_component(
            batch, "solver", ["prompt_input_ids", "prompt_attention_mask", "completion_input_ids"]
        )
        processed_batch["pos"] = self._collate_component(
            batch, "pos", ["input_ids", "attention_mask", "labels"]
        )
        processed_batch["neg"] = self._collate_component(
            batch, "neg", ["input_ids", "attention_mask", "labels"]
        )
        # Collate graphs
        processed_batch["graph"] = Batch.from_data_list([item["graph"] for item in batch])
        return processed_batch
