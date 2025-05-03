class TwinNodeStructuredDistillation(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_agents):
        super(TwinNodeStructuredDistillation, self).__init__()
        self.num_agents = num_agents
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)
        self.twin_linear = nn.Linear(out_channels * 2, out_channels) 


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
      
        dense_adj = to_dense_adj(edge_index)[0]
        twin_node_representation = []

        for i in range(self.num_agents):
        
          neighbors = torch.nonzero(dense_adj[i,:])[:,0] 
          if len(neighbors) > 0:
            neighbor_features = x[neighbors]
            agent_features = x[i].repeat(len(neighbors),1)

 
            combined_features = torch.cat([agent_features, neighbor_features], dim=1)
            twin_feature = self.twin_linear(combined_features)
            twin_node_representation.append(twin_feature)
          else:
            twin_node_representation.append(x[i].unsqueeze(0)) 

        twin_node_representation = torch.cat(twin_node_representation)

        return twin_node_representation


num_nodes = 5
num_features = 3
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                           [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
x = torch.randn(num_nodes, num_features)
data = Data(x=x, edge_index=edge_index)

model = TwinNodeStructuredDistillation(in_channels=num_features, hidden_channels=64, out_channels=32, num_agents=num_nodes)
twin_node_representation = model(data)
twin_node_representation.shape
