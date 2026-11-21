import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    d_model: int = 768 #hidden_size
    vocab_size: int = 262144
    max_seq_len: int = 4096
    n_layers: int = 19 # num_hidden_layers
    # Attention
    num_attention_heads: int = 64 
    num_key_value_heads: int =4
    head_dim: int = d_model // num_attention_heads
    qkv_bias: bool = False
    qk_norm: bool = True
    attn_dropout: float = 0.0
    # Mixture of Experts
    num_experts: int = 128
    num_experts_per_token: int = 6
    n_groups: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.5
    num_shared_experts: int = 1
    intermediate_size: int = 8192
    # RoPE
    base: float = 10000.0
    # RMS
    rms_norm_eps: float = 1e-06
    



class RMSNorm(nn.Module):
    def __init__(self,d_model, epsilon = 1e-20):
        super().__init__()
        self.epsilon = epsilon
        self.weights = nn.Parameter(torch.ones(d_model))

    def forward(self,x):
        variance = x.pow(2).mean(dim= - 1, keepdims =True)
        rms = x * torch.rsqrt(variance + self.epsilon)
        return x * rms * self.weights


class SarvamRotaryPositionalEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.base = config.base

        inv_freq = 1.0 / (
            self.base ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                / self.head_dim
            )
        )
        positions = torch.arange(config.max_seq_len)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("positions", positions, persistent=False)

        
    def forward(self,x):
        _ , seq_len , _ = x.shape
        freqs = self.positions[:seq_len,None].float() * self.inv_freq[None,:]
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)

        return sin, cos


def rotate_half(x):
    x1 = x[...,:x.shape[-1]//2]
    x2 = x[...,x.shape[-1]//2:]
    return torch.cat((-x2,x1),dim=-1)

def apply_rope_embedding(q,k,cos,sin):
    cos = cos[None,None,:,:] #(seq_len,d_model) -> #(1,seq_len,1,d_model)
    sin = sin[None,None,:,:]

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin

    return q_embed , k_embed


def repeat_kv(x,num_kv_groups):
    batch_size, num_heads , seq_len , head_dim = x.shape
    x = x[:,:,None,:,:].expand(-1,-1,num_kv_groups,-1,-1)
    return x.reshape(batch_size,num_heads * num_kv_groups,seq_len,head_dim)

class SarvamGroupedQueryAttention(nn.Module):
    def __init__(self,config: TransformerConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.key_value_heads = config.num_key_value_heads

        assert config.d_model % self.num_heads == 0 
        self.head_dim = config.d_model // self.num_heads
        self.num_kv_groups = self.num_heads // self.key_value_heads

        if self.config.qk_norm:
            self.query_rms_norm = RMSNorm(config.d_model,config.rms_norm_eps)
            self.key_rms_norm = RMSNorm(config.d_model,config.rms_norm_eps)
            
        self.rope = SarvamRotaryPositionalEmbedding(config)

        self.query_key_value = nn.Linear(
            config.d_model, (self.num_heads + 2 * self.key_value_heads) * self.head_dim , bias = config.qkv_bias
        )
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.dense = nn.Linear(config.d_model,config.d_model,bias= False)



    def forward(self,x,position_embeddings,attn_mask):
        batch_size, seq_len, d_model = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.view(batch_size,seq_len,self.num_heads + 2 * self.key_value_heads,self.head_dim)
        query,key,value = qkv.split([self.num_heads,self.key_value_heads,self.key_value_heads],dim = 2)
        query = query.transpose(1,2).contiguous() # (batch Size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len,head_dim)
        key = key.transpose(1,2).contiguous()
        value = value.transpose(1,2).contiguous()
        if self.config.qk_norm:
            query = self.query_rms_norm(query)
            key = self.key_rms_norm(key)
        sin, cos = position_embeddings
        query,key = apply_rope_embedding(query,key,cos,sin)
        key = repeat_kv(key,self.num_kv_groups)
        value = repeat_kv(value,self.num_kv_groups)

        attn_scores = query @ key.transpose(-1,-2)  / math.sqrt(self.head_dim)
        attn_scores = attn_scores + attn_mask 
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        context_vector = (attn_weights @ value).transpose(1,2).contiguous() # (batch Size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads,head_dim)
        context_vector = context_vector.view(batch_size,seq_len,-1) # (batch_size, seq_len, num_heads,head_dim) -> (batch_size, seq_len, d_model)
        attn_output = self.dense(context_vector)
        return attn_output


# MLP - FFNN
class Expert(nn.Module):
    def __init__(self,config: TransformerConfig):
        super().__init__()
        self.gate = nn.Linear(config.d_model,config.intermediate_size,bias = False)
        self.up_proj = nn.Linear(config.d_model,config.intermediate_size,bias= False)
        self.down_proj = nn.Linear(config.intermediate_size,config.d_model,bias= False)
        self.act = nn.SiLU()

    def forward(self,x):
        return self.down_proj(self.act(self.gate(x)) * self.up_proj(x))


class SarvamRouter(nn.Module):
    def __init__(self,config: TransformerConfig):
        super().__init__()
        self.config = TransformerConfig()
        self.num_experts = config.num_experts
        self.n_groups = config.n_groups
        self.topk_group = config.topk_group
        self.topk = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor

        self.weight = nn.Parameter(torch.empty(config.num_experts,config.d_model))
        self.expert_bias = nn.Parameter(torch.zeros((config.num_experts)),requires_grad = False)

    def reset_parameters(self):
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def group_limited_topk(self,scores):
        num_tokens , _  = scores.shape
        #(num_tokens, num_experts) -> (num_tokens, n_groups, num_experts)
        group_scores = scores.view(num_tokens, self.n_groups,-1).topk(2,dim=-1)[0].sum(dim=-1) 
        topk_group_idx = torch.topk(group_scores,k = self.topk_group,dim=-1)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1,topk_group_idx,1) 
        score_mask = group_mask.unsqueeze(-1).expand(num_tokens,self.n_groups,self.num_experts//self.n_groups).reshape(num_tokens,self.num_experts)
        masked_scores = score_mask.masked_fill(~score_mask.bool(), float('-inf'))
        probs , top_idx = torch.topk(masked_scores , k = self.topk, dim=-1)
        return probs, top_idx
        
    def forward(self,x):
        x = x.view(-1,x.shape[-1]) #(batch_size, seq_len,d_model) -> (batch_size * seq_len , d_model)
        logits = F.linear(x.to(torch.float32), self.weight.to(torch.float32))
        scores = torch.sigmoid(logits.float()).type_as(logits)
        routing_scores = scores + self.expert_bias
        probs,  topk_idx = self.group_limited_topk(routing_scores)
        scores = torch.gather(routing_scores,dim=-1, index = topk_idx).type_as(logits)
        topk_weight = scores / (scores.sum(dim=-1,keepdim = True) + 1e-20) if self.topk > 1 else scores
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight , logits

class SarvamMixtureofExperts(nn.Module):
    def __init__(self,config: TransformerConfig):
        super().__init__()
        self.config = config
        self.SarvamRouter = SarvamRouter(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        self.shared_experts = nn.ModuleList([Expert(config) for _ in range(config.num_shared_experts)])

    def forward(self,x):
        batch, seq_len, d_model = x.shape
        topk_idx, topk_weight ,  logits = self.SarvamRouter(x)

        shared_out = sum(expert(x) for expert in self.shared_experts)
        
        flat_x = x.view(-1,x.shape[-1])
        routed_out = torch.zeros_like(flat_x)
        for i,expert in enumerate(self.experts):
            mask = (topk_idx==i)
            row_idx , which_k = torch.where(mask)
            if row_idx.numel()==0:
                continue
            expert_in = flat_x[row_idx]
            expert_out = expert(expert_in)
            w = topk_weight[row_idx,i].unsqueeze(-1)
            routed_out.index_add_(0,row_idx,w * expert_out)
        routed_out = routed_out.view(batch, seq_len, d_model)
        return shared_out + routed_out



class SarvamTransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = SarvamGroupedQueryAttention(config=config)
        self.post_attention_layernorm = RMSNorm(config.d_model,config.rms_norm_eps)
        self.moe = SarvamMixtureofExperts(config=config)

    def forward(self,x,pos,attn):
        residual = x
        x = self.input_layernorm(x)
        x = self.attn(x,pos,attn)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.moe(x)
        x = x + residual
        return x




class SarvamCausalLM(nn.Module):
    def __init__(self,config: TransformerConfig):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size,config.d_model)
        self.rope = SarvamRotaryPositionalEmbedding(config)
        self.blocks = nn.ModuleList(
                    [SarvamTransformerBlock(config) for _ in range(19)]
                )
        self.norm = RMSNorm(config.d_model,config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model,config.vocab_size,bias = False)

    def causal_mask(self,S: int,device):
        mask = torch.triu(torch.ones(S,S,device = device,dtype = torch.bool),diagonal=1)
        return torch.zeros(S,S,device=device).masked_fill(mask,float('-inf'))


    def forward(self,x):
        B,S,D = x.shape
        x = self.embed(x)
        position_embedding = self.rope(x)
        print(x.shape)

        mask = self.causal_mask(S,x.device)
        for block in self.blocks:
            x = block(x,position_embedding,mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        print(logits.shape)


        return logits






        
        


if __name__ == "__main__":

    x = torch.randn(2,10,768)
    # B,S,D = x.shape
    # mask = torch.triu(torch.ones(S,S,device = x.device,dtype = torch.bool),diagonal=1)
    # attn_mask = torch.zeros(S,S,device= x.device).masked_fill(mask,float('-inf'))
    # gqa = SarvamGroupedQueryAttention(TransformerConfig)
    # rope = SarvamRotaryPositionalEmbedding(TransformerConfig)
    # pos = rope(x)
    # output =gqa(x,pos,attn_mask[:,:,None,None])
    # print(output.shape)
    # moe = SarvamTransformerBlock(TransformerConfig)
    # out = moe(x)
    # print(out.shape)
    # rotated = rotate_half(x)
    # print(rotated.shape)
    # transformer = SarvamTransformerBlock(TransformerConfig)
    # print(transformer(x,pos,mask).shape)
    lm = SarvamCausalLM(TransformerConfig)
    device = torch.cuda.is_available()
    
    lm = lm.to("cuda")
    total_params = 0

    for param in lm.parameters():
        total_params += param.numel()

    print(f"Total parameters: {total_params:,}")
    
    