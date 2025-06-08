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
    
    def forward(self, decomposer, solver, pos, neg, graph):
        """Forward pass without memory handling for high-memory GPU."""
        
        try:
            # Quick validation
            batch_size = len(decomposer)
            
            # Process tensors
            decomposer_input_ids = torch.stack([item["prompt_input_ids"] for item in decomposer])
            decomposer_attention_mask = torch.stack([item["prompt_attention_mask"] for item in decomposer])
            decomposer_labels = torch.stack([item["completion_input_ids"] for item in decomposer])
            
            solver_input_ids = torch.stack([item["prompt_input_ids"] for item in solver])
            solver_attention_mask = torch.stack([item["prompt_attention_mask"] for item in solver])
            solver_labels = torch.stack([item["completion_input_ids"] for item in solver])
            
            pos_input_ids = torch.stack([item["input_ids"] for item in pos])
            pos_attention_mask = torch.stack([item["attention_mask"] for item in pos])
            pos_labels = torch.stack([item["labels"] for item in pos])
            
            neg_input_ids = torch.stack([item["input_ids"] for item in neg])
            neg_attention_mask = torch.stack([item["attention_mask"] for item in neg])
            neg_labels = torch.stack([item["labels"] for item in neg])
                        
            # Component computations
            decomposer_loss, decomposer_emb = self.decomposer(
                decomposer_input_ids, decomposer_attention_mask, decomposer_labels
            )
            
            solver_loss, solver_emb = self.solver(
                solver_input_ids, solver_attention_mask, solver_labels
            )
            
            pos_loss, pos_emb = self.solver(pos_input_ids, pos_attention_mask, pos_labels)
            
            _, neg_emb = self.solver(neg_input_ids, neg_attention_mask, None)
            
            # Contrastive computation
            pos_h = torch.relu(self.mlp1(pos_emb))
            pos_score = torch.tanh(self.mlp2(pos_h))
            neg_h = torch.relu(self.mlp1(neg_emb))
            neg_score = torch.tanh(self.mlp2(neg_h))
            
            mr_cri = torch.nn.MarginRankingLoss(1.0, reduction='mean').to(pos_score.device)
            mr_loss = mr_cri(pos_score, neg_score, torch.ones_like(pos_score).to(pos_score.device))
            
            from torch_geometric.loader import DataLoader
            graph_loader = DataLoader(graph, batch_size=len(graph), shuffle=False, pin_memory=False, num_workers=0)
            graph_batch = next(iter(graph_loader))
            
            gcn_output, logits = self.gcn(graph_batch.x, graph_batch.edge_index)
            graph_batch.y = graph_batch.y.to(logits.device)
            ce_cri = torch.nn.CrossEntropyLoss()
            node_loss = ce_cri(logits, graph_batch.y)
            
            alignment_loss = F.mse_loss(decomposer_emb, solver_emb)
            
            # Calculate final losses
            lm_combined = self.alpha * (decomposer_loss + solver_loss + pos_loss)
            node_weighted = self.beta * node_loss
            mr_weighted = self.gamma * mr_loss
            alignment_weighted = self.delta * alignment_loss
            
            total_loss = lm_combined + node_weighted + mr_weighted + alignment_weighted
                        
            return (lm_combined, node_weighted, mr_weighted, alignment_weighted)
            
        except Exception as e:
            print(f"🚨 CRASH in forward pass: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            device = next(self.parameters()).device
            dummy = torch.tensor(0.1, device=device, requires_grad=True)
            return dummy, dummy, dummy, dummy
        
        


class SocraticMAGDiDataCollator:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch):        
        # Handle variable-length lists by taking first item from each list
        # (or implement more sophisticated aggregation if needed)
        
        processed_batch = {
            "decomposer": [],
            "solver": [], 
            "pos": [],
            "neg": [],
            "graph": []
        }
        
        for item in batch:
            # Get first decomposer example if exists, else create dummy
            if item["decomposer"]:
                decomposer_item = item["decomposer"][0]
            else:
                # Create dummy tensors if no decomposer examples
                decomposer_item = {
                    "prompt_input_ids": torch.zeros(512, dtype=torch.long),
                    "prompt_attention_mask": torch.zeros(512, dtype=torch.long),
                    "completion_input_ids": torch.zeros(512, dtype=torch.long),
                    "completion_attention_mask": torch.zeros(512, dtype=torch.long)
                }
            processed_batch["decomposer"].append(decomposer_item)
            
            # Get first solver example if exists, else create dummy  
            if item["solver"]:
                solver_item = item["solver"][0]
            else:
                solver_item = {
                    "prompt_input_ids": torch.zeros(512, dtype=torch.long),
                    "prompt_attention_mask": torch.zeros(512, dtype=torch.long),
                    "completion_input_ids": torch.zeros(512, dtype=torch.long),
                    "completion_attention_mask": torch.zeros(512, dtype=torch.long)
                }
            processed_batch["solver"].append(solver_item)
            
            # Get first positive example if exists, else create dummy
            if item["pos"]:
                pos_item = item["pos"][0]
            else:
                pos_item = {
                    "input_ids": torch.zeros(512, dtype=torch.long),
                    "attention_mask": torch.zeros(512, dtype=torch.long),
                    "labels": torch.zeros(512, dtype=torch.long)
                }
            processed_batch["pos"].append(pos_item)
            
            # Get first negative example if exists, else create dummy
            if item["neg"]:
                neg_item = item["neg"][0] 
            else:
                neg_item = {
                    "input_ids": torch.zeros(512, dtype=torch.long),
                    "attention_mask": torch.zeros(512, dtype=torch.long),
                    "labels": torch.zeros(512, dtype=torch.long)
                }
            processed_batch["neg"].append(neg_item)
            
            # Add graph
            processed_batch["graph"].append(item["graph"])
        
        return processed_batch
