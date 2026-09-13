import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self,img_size,patch_size, embed_dim, in_channels = 3):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels= in_channels,
            out_channels= embed_dim,
            kernel_size= patch_size,
            stride= patch_size,
        )

    def forward(self,x):
        x = self.proj(x) # (batch , embed_dim , sqrt(n_patches) , sqrt(n_patches))
        x = x.flatten(2) # (batch , embed_dim , n_patches)
        x = x.transpose(1,2) # (batch, n_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim = 768, num_heads = 12, attn_p = 0 , proj_p = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim,embed_dim)
        self.key = nn.Linear(embed_dim,embed_dim)
        self.value = nn.Linear(embed_dim,embed_dim)
        self.proj = nn.Linear(embed_dim,embed_dim)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self,x):
        batch_size, seq_len,  embed_dim = x.shape
        q = self.query(x).reshape(batch_size,seq_len,self.num_heads, self.head_dim).transpose(1,2)
        k = self.key(x).reshape(batch_size,seq_len,self.num_heads, self.head_dim).transpose(1,2)
        v = self.value(x).reshape(batch_size,seq_len,self.num_heads, self.head_dim).transpose(1,2)

        attn_scores = q @ k.transpose(-1,-2) / (self.head_dim ** 0.5)
        attn_weights = attn_scores.softmax(dim = -1)
        attn_weights = self.attn_drop(attn_weights)
        context_vector = (attn_weights @ v ).transpose(1,2).reshape(batch_size,seq_len,embed_dim)
        context_vector = self.proj(context_vector)
        context_vector = self.proj_drop(context_vector)
        return context_vector

class MLP(nn.Module):
    def __init__(self,embed_dim,mlp_ratio = 4,mlp_p = 0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffnn = nn.Sequential(
            nn.Linear(embed_dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_p),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(mlp_p)
        )
    
    def forward(self,x):
        return self.ffnn(x)
    

class TransformerBlock(nn.Module):
    def __init__(self,embed_dim = 768 , num_heads = 12, mlp_ratio = 4.0 , attn_p = 0 , proj_p = 0, mlp_p = 0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.attn = MultiHeadAttention(embed_dim=embed_dim,num_heads=num_heads,attn_p=attn_p,proj_p=proj_p)
        self.norm2 = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.mlp = MLP(embed_dim= embed_dim , mlp_ratio= mlp_ratio, mlp_p= mlp_p)
    
    def forward(self,x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    

class VisionTransformer(nn.Module):
    def __init__(self,img_size =  224, patch_size = 16 , in_channels = 3, n_classes = 2,
                 embed_dim = 768 , depth = 12, num_heads = 12, mlp_ratio = 4.0 , 
                 pos_drop = 0, attn_p = 0 , proj_p = 0, mlp_p = 0 ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size=img_size,patch_size=patch_size,embed_dim=embed_dim,in_channels=in_channels)
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_emb = nn.Parameter((torch.zeros(1,1 + self.patch_embedding.num_patches,embed_dim)))
        self.pos_drop = nn.Dropout(pos_drop)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,attn_p=attn_p,proj_p=proj_p,mlp_p=mlp_p)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self,x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x) # (batch_size,num_patches,embed_dim)
        cls_token = self.cls_token.expand(batch_size,-1,-1) #(batch_size,1,embed_dim)
        x = torch.cat((cls_token,x),dim = 1) #(batch_size , num_patches + 1, embed_dim)
        x = x + self.pos_emb
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        print(x.shape)
        cls_token_final = x[:,0]
        print(cls_token_final.shape)
        x = self.head(cls_token_final)
        return x , cls_token_final


rand = torch.rand(1,3,224,224)
vit = VisionTransformer()
print(vit(rand))
        

