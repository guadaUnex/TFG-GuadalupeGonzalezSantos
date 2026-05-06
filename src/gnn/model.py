class GATNetwork(nn.Module):
    def init(self, in_channels, out_channels, network_config, star_topology, concat=False):
        """Initalize object
        
        Args:
            in_channels (int): Size of each input sample
                Note: Number of node features.
            
            out_channels (int): Size of each output sample
                Note: Should be 3 + num_joints for the intended use case.
            
            network_config (dict): Dictionary containing the netwoek creation attributes 
                Note: Loaded from the .yaml file or the .pth checkpoint.
        """
        
        super().init()
        
        self.in_channels     = in_channels
        self.out_channels    = out_channels
        self.star_topology   = star_topology
        # self.model_pth       = network_config['model_pth']
        if not 'model_type' in network_config:
            self.model_type  = "GATNetwork"
        else:
            self.model_type  = network_config['model_type']

        self.hidden_channels = network_config['hidden_channels']
        self.heads           = network_config['heads']
        self.concat          = False #network_config['concat']

        self.num_hidden_layers = len(self.hidden_channels)
        self.num_layers = self.num_hidden_layers + 1
        
        self.layers = self.make_layers()
        self.model = Sequential('x, edge_index', self.layers)
        
    def make_layers(self):
        layers = []
        
        # Input layer
        layers.append((GATConv(self.in_channels, self.hidden_channels[0], self.heads[0], self.concat), 
                       'x, edge_index -> x'))
        layers.append(ReLU(inplace=True))
            
        # Hidden layers
        for idx in range(len(self.hidden_channels) - 1):
            # If previous layer concatenated, actual output size is hidden * heads
            input_dim = self.hidden_channels[idx] * (self.heads[idx] if self.concat else 1)
            output_dim = self.hidden_channels[idx + 1]
            heads = self.heads[idx + 1]
            layers.append((GATConv(input_dim, output_dim, heads, self.concat),
                           'x, edge_index -> x'))
            layers.append(LeakyReLU(negative_slope=0.1))
            
        # Output layer — always 1 head, no concat, to get clean [N, out_channels] output
        # Input dim comes from the last hidden layer (heads[-1] is the last hidden layer's heads)
        input_dim = self.hidden_channels[-1] * (self.heads[-1] if self.concat else 1)
        layers.append((GATConv(input_dim, self.out_channels, heads=1, concat=False), 
                       'x, edge_index -> x'))
        
        return layers
        
    def forward(self, batch_data):
        """
        Args:
            batch_data: A PyG Batch object with fields:
                - batch_data.x:          [total_nodes, in_channels]
                - batch_data.edge_index: [2, total_edges]
                - batch_data.ptr:        [batch_size + 1] — start index of each graph

        Returns:
            Tensor of shape [batch_size, out_channels]
        """
        # Run message passing over ALL nodes (neighborhood aggregation)
        out = self.model(batch_data.x, batch_data.edge_index)  # [total_nodes, out_channels]
        
        if self.star_topology:
            # Select only the root node (node 0) of each graph in the batch
            root_idx = batch_data.ptr[:-1] # [batch_size]
        
            return out[root_idx]*100  # [batch_size, out_channels]
        else:
            # Calculate the average out value
            out_pooled = global_mean_pool(out, batch_data.batch) # [batch_size, out_channels]
            return out_pooled*100