import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    embedding_dimension: int = 512
    num_attention_heads: int = 8
    attention_dropout_p: float = 0.0
    hidden_dropout_p: float = 0.0
    mlp_ratio: int = 4
    encoder_depth: int = 6
    decoder_depth: int = 6

    src_vocab_size: int = 30522
    tgt_vocab_size: int = 32000

    max_src_len:int = 512
    max_tgt_len: int = 512
    learn_pos_embed: bool = False

class PositionalEncoding(nn.Module):
    def __init__(self, max_len,embed_dim,required_grad = False):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.requires_grad = required_grad
        self.encodings = self._build_positional_encodings()
    
    def _build_positional_encodings(self):
        encoding = torch.zeros(self.max_len, self.embed_dim,dtype = torch.float32)
        position_idx = torch.arange(0,self.max_len, dtype = torch.float32).reshape(-1,1)
        embed_dim_skip = torch.arange(0,self.embed_dim,step = 2,dtype = torch.float32)

        encoding[:,0::2] = torch.sin(position_idx/(10000** (embed_dim_skip/self.embed_dim)))
        encoding[:,1::2] = torch.cos(position_idx/(10000** (self.embed_dim/self.embed_dim)))

        encoding = nn.Parameter(encoding,requires_grad=self.requires_grad)
        return encoding
    
    def forward(self,x):
        seq_len = x.shape[1]
        encodings = self.encodings[:seq_len]
        x = x + encodings
        return x
    
class Embeddings(nn.Module):
    def __init__(self,config : TransformerConfig):
        super().__init__()
        self.src_embeddings = nn.Embedding(config.src_vocab_size, config.embedding_dimension)
        self.tgt_embeddings = nn.Embedding(config.tgt_vocab_size, config.embedding_dimension)
        self.src_positional_encodings = PositionalEncoding(
            config.max_src_len,
            config.embedding_dimension,
            config.learn_pos_embed
        )
        self.tgt_posiitonal_encodigns = PositionalEncoding(
            config.max_tgt_len,
            config.embedding_dimension,
            config.learn_pos_embed
        )

    def forward_src(self,input_ids):
        embeddings = self.src_embeddings(input_ids)
        embeddings = self.src_positional_encodings(embeddings)
        return embeddings
    
    def forward_tgt(self,input_ids):
        embeddings = self.tgt_embeddings(input_ids)
        embeddings = self.tgt_posiitonal_encodigns(embeddings)
        return embeddings
    
class Attention(nn.Module):
    def __init__(self,config: TransformerConfig):
        super().__init__()
        self.config = config
        assert  config.embedding_dimension % config.num_attention_heads ==0 , "Embedding Dimension must be divisible by number of heads"
        self.head_dim = config.embedding_dimension // config.num_attention_heads
        self.w_q = nn.Linear(config.embedding_dimension,config.embedding_dimension)
        self.w_k = nn.Linear(config.embedding_dimension,config.embedding_dimension)
        self.w_v = nn.Linear(config.embedding_dimension,config.embedding_dimension)
        self.w_o = nn.Linear(config.embedding_dimension, config.embedding_dimension)

    def forward(self,src,tgt = None, attention_mask = None, causal = False):
        batch, src_len ,d_model = src.shape
        if tgt is None:
            q = self.w_q(src).reshape(batch,src_len,self.config.num_attention_heads,self.head_dim).transpose(1,2).contiguous()
            k = self.w_k(src).reshape(batch,src_len,self.config.num_attention_heads,self.head_dim).transpose(1,2).contiguous()
            v = self.w_v(src).reshape(batch,src_len,self.config.num_attention_heads,self.head_dim).transpose(1,2).contiguous()

            if attention_mask is not None:
                attention_mask = attention_mask.bool()
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,src_len,1)
            
            attention_out = F.scaled_dot_product_attention(q,k,v,
                                                           attn_mask=attention_mask,
                                                           dropout_p=self.config.attention_dropout_p if self.training else 0,
                                                           is_causal=causal)
        else:
            tgt_len = tgt.shape[1]
            print(tgt.shape)
            q = self.w_q(tgt).reshape(batch,tgt_len,self.config.num_attention_heads,self.head_dim).transpose(1,2).contiguous()
            k = self.w_k(tgt).reshape(batch,tgt_len,self.config.num_attention_heads,self.head_dim).transpose(1,2).contiguous()
            v = self.w_v(tgt).reshape(batch,tgt_len,self.config.num_attention_heads,self.head_dim).transpose(1,2).contiguous()

            if attention_mask is not None:
                attention_mask = attention_mask.bool()
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,tgt_len,1)

            attention_out = F.scaled_dot_product_attention(q,k,v,
                                                           attn_mask=attention_mask,
                                                            dropout_p=self.config.attention_dropout_p if self.training else 0.0,
                                                            is_causal= False )

        attention_out = attention_out.transpose(1,2).flatten(2)
        attention_out = self.w_o(attention_out)
        return attention_out
    
class FeedForward(nn.Module):
    def __init__(self,config:TransformerConfig):
        super().__init__()
        hidden_size = config.embedding_dimension * config.mlp_ratio
        self.intermediate_dense = nn.Linear(config.embedding_dimension , hidden_size)
        self.activation = nn.GELU()
        self.intermediate_dropout = nn.Dropout(config.hidden_dropout_p)
        self.output_dense = nn.Linear(hidden_size,config.embedding_dimension)
        self.output_dropout = nn.Dropout(config.hidden_dropout_p)
    
    def forward(self,x):
        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.intermediate_dropout(x)
        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x
    

class TransformerEncodeLayer(nn.Module):
    def __init__(self,config:TransformerConfig):
        super().__init__()
        self.enc_attention = Attention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_p)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension)
        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dimension)
    
    def forward(self,x,attention_mask = None):
        x =  x + self.dropout(self.enc_attention(x,attention_mask = attention_mask))
        x = self.layer_norm(x)

        x = x + self.feed_forward(x)
        x = self.final_layer_norm(x)
        return x


class TransformerDecoderlayer(nn.Module):
    def __init__(self,config: TransformerConfig):
        super().__init__()
        self.dec_attn = Attention(config)
        self.dec_attn_dropout  = nn.Dropout(config.hidden_dropout_p)
        self.dec_attn_layernorm = nn.LayerNorm(config.embedding_dimension)

        self.cross_attention = Attention(config)
        self.cross_attention_dropout = nn.Dropout(config.hidden_dropout_p)
        self.cross_attention_layernorm = nn.LayerNorm(config.embedding_dimension)

        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dimension)

    def forward(self,src,tgt,src_mask = None, tgt_mask = None):
        tgt = tgt + self.dec_attn_dropout(self.dec_attn(tgt,attention_mask = tgt_mask,causal = True))
        tgt = self.dec_attn_layernorm(tgt)

        tgt = tgt + self.cross_attention_dropout(self.cross_attention(src,tgt,attention_mask = src_mask))
        tgt = self.cross_attention_layernorm(tgt)

        tgt = tgt + self.feed_forward(tgt)
        tgt = self.final_layer_norm(tgt)

        return tgt
    
class Transformer(nn.Module):
    def __init__(self,config: TransformerConfig):
        super().__init__()

        self.config = config
        self.encodings = Embeddings(config)

        self.encoder = nn.ModuleList(
            [TransformerEncodeLayer(config) for _ in range(config.encoder_depth)]
        )

        self.decoder = nn.ModuleList(
            [TransformerDecoderlayer(config) for _ in range(config.decoder_depth)]
        )

        self.head = nn.Linear(config.embedding_dimension,config.tgt_vocab_size)
        self.apply(_init_weights_)

    def forward(self,src_ids,tgt_ids,src_attention_mask = None, tgt_attention_mask = None):
        src_embeddings = self.encodings.forward_src(src_ids)
        tgt_embeddings = self.encodings.forward_tgt(tgt_ids)

        for layer in self.encoder:
            src_embeddings = layer(src_embeddings,src_attention_mask)

        for layer in self.decoder:
            tgt_embeddings = layer(src_embeddings,tgt_embeddings,src_attention_mask,tgt_attention_mask)
        
        pred = self.head(tgt_embeddings)

        return pred
    
    @torch.no_grad()
    def inference(self,src_ids,tgt_start_id = 2 , tgt_end_id = 3, max_len = 512):
        tgt_ids = torch.tensor([tgt_start_id], device = src_ids.device).reshape(1,1)
        src_embeddings = self.encodings.forward_src(src_ids)
        for layer in self.encoder:
            src_embeddings = layer(src_embeddings)
        
        for i in range(max_len):
            tgt_embeddings = self.encodings.forward_tgt(tgt_ids)
            for layer in self.decoder:
                tgt_embeddings = layer(src_embeddings,tgt_embeddings)
            tgt_embeddings = tgt_embeddings[:,-1]
            pred = self.head(tgt_embeddings)
            pred = pred.argmax(axis = - 1).unsqueeze(0)
            tgt_ids = torch.cat([tgt_ids,pred],axis = -1)

            if torch.all(pred==tgt_end_id):
                break
        return tgt_ids.squeeze().cpu().tolist()
        

def _init_weights_(module):
    if isinstance(module,nn.Linear):
        module.weight.data.normal_(mean = 0.0, std = 0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module,nn.Embedding):
        module.weight.data.normal_(mean = 0.0, std = 0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module,nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

if __name__ == "__main__":
    src = torch.randint(0,1000,(1,128)).to("cuda")
    config = TransformerConfig()
    model = Transformer(config)
    model = model.to("cuda")
    model.inference(src)
                              