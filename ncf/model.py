import torch
import torch.nn as nn


class NCF(nn.Module):
    """
    NCF combines:
    - GMF (Generalized Matrix Factorization) branch: element-wise product of user/item embeddings
    - MLP branch: concatenation of user/item embeddings fed through MLP layers
    Final output: logits of GMF and MLP outputs.
    """
    def __init__(self, num_users, num_items, embed_dim, mlp_layers):
        super().__init__()
        # GMF embeddings
        self.user_emb_gmf = nn.Embedding(num_users, embed_dim)
        self.item_emb_gmf = nn.Embedding(num_items, embed_dim)

        # MLP embeddings - concat(mlp_layers[0] // 2) = mlp_layers[0]
        self.user_emb_mlp = nn.Embedding(num_users, mlp_layers[0]//2)
        self.item_emb_mlp = nn.Embedding(num_items, mlp_layers[0]//2)

        # Build MLP from several layers of Linear -> ReLU -> Dropout
        mlp_modules = []
        input_size = mlp_layers[0]
        for layer_size in mlp_layers[1:]:
            mlp_modules.append(nn.Linear(input_size, layer_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=0.5))
            input_size = layer_size
        self.mlp = nn.Sequential(*mlp_modules)

        # Final layer: concat GMF and MLP output
        self.predict_layer = nn.Linear(embed_dim + mlp_layers[-1], 1)
        # self.sigmoid = nn.Sigmoid()

        # Initializes layers with specific distributions
        self._init_weights()

    def _init_weights(self):

        nn.init.normal_(self.user_emb_gmf.weight, std=0.01)
        nn.init.normal_(self.item_emb_gmf.weight, std=0.01)
        nn.init.normal_(self.user_emb_mlp.weight, std=0.01)
        nn.init.normal_(self.item_emb_mlp.weight, std=0.01)

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.predict_layer.weight)
        if self.predict_layer.bias is not None:
            nn.init.zeros_(self.predict_layer.bias)

    def forward(self, user, item):

        # GMF path
        gmf_u = self.user_emb_gmf(user).squeeze(1) # size = (batch, embed_dim)
        gmf_i = self.item_emb_gmf(item).squeeze(1)
        gmf_output = gmf_u * gmf_i # element-wise product

        # MLP path
        mlp_u = self.user_emb_mlp(user).squeeze(1) # size = (batch, mlp_layers[0]/2)
        mlp_i = self.item_emb_mlp(item).squeeze(1)
        mlp_input = torch.cat([mlp_u, mlp_i], dim=1) # size = (batch, mlp_layers[0])
        mlp_output = self.mlp(mlp_input) # size = (batch, mlp_layers[-1])

        # Concatenate GMF and MLP
        concat = torch.cat([gmf_output, mlp_output], dim=1) # size = (batch, embed_dim + mlp_layers[-1])
        logits = self.predict_layer(concat)
        return logits
