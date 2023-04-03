import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


class Head(nn.Module):

    def __init__(self, head_size, num_embedding, block_size):
        """

        :param head_size: num_embeddings/num of heads
        :param num_embedding: the channels
        :param block_size: the number of timestamps to be taken into account
        """
        super().__init__()
        head_size = int(head_size)
        print("number embedding ",num_embedding,"head size",head_size)
        self.key = nn.Linear(num_embedding, head_size, bias=False)
        self.query = nn.Linear(num_embedding, head_size, bias=False)
        self.value = nn.Linear(num_embedding, head_size, bias=False)

        # Transform x(B,T,C) -> (B,T,head_size) through (K x X) with no bias

        """If you have parameters in your model, which should be saved and restored in the state_dict, 
        but not trained by the optimizer, you should register them as buffers. Buffers won’t be returned in 
        model.parameters(), so that the optimizer won’t have a change to update them."""

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)  # B,T,heads_dim
        q = self.query(x)  # B,T,heads_dim

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B,T,h) *(B,h,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        # so that it doesn't communicate with the past
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)  # (B,T,C) -> (B,T,head_dim)
        out = wei @ v  # (B,T,T) *(B,T,heads_dim) -> (B,T,heads_dim)

        return out


class MultiHeadAttention(nn.Module):
    """ Running multiple attention heads in parallel"""

    def __init__(self, num_heads, head_size, num_embeddings, block_size):
        super().__init__()
        head_size = int(head_size)
        print("num heads ",num_heads)
        self.heads = nn.ModuleList(
            [Head(head_size=head_size, num_embedding=num_embeddings, block_size=block_size) for _ in range(num_heads)]).to(
            device=device)

        self.projection = nn.Linear(head_size * num_heads, num_embeddings).to(device=device)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        out = self.projection(out)

        return out


class FeedForward(nn.Module):

    def __init__(self, num_embed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_embed, 4 * num_embed),
                                 nn.ReLU(),
                                 nn.Linear(4 * num_embed, num_embed)).to(device=device)

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, num_embeddding, num_head, block_size):
        super().__init__()
        head_size = int(num_embeddding // num_head)
        self.sa = MultiHeadAttention(num_head, head_size, num_embeddings=num_embeddding,
                                     block_size=block_size).to(device=device)

        self.feed_forward = FeedForward(num_embed=num_embeddding).to(device=device)

        self.ln1 = nn.LayerNorm(num_embeddding).to(device=device)
        self.ln2 = nn.LayerNorm(num_embeddding).to(device=device)

    def forward(self, x):
        x = x + self.sa((self.ln1(x)))
        x = x + self.feed_forward(self.ln2(x))

        return x


class BigramModel(nn.Module):

    def __init__(self, vocab_size, embed_size=256, block_size=8, num_heads = 8, device=device,n_layers=6):
        """

        :param vocab_size:
        :param embed_size:
        :param block_size: Number of timestamps to look after (8 timesteps)
        """
        super().__init__()
        self.block_size = block_size
        self.num_heads = num_heads
        head_size = embed_size/num_heads
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size).to(device)
        # it's a lookup table , for every value in vocab_size we will have a corresponding vector of size num_embedding

        self.position_embedding_table = nn.Embedding(block_size, embed_size).to(device)  # positional embedding

        # self.sa_head = Head(embed_size, embed_size, block_size).to(device)  # keeping it num_embedding just for now\
        # print("the type is ",type(embed_size//head_size))
        self.sa_head = MultiHeadAttention(int(embed_size // head_size), head_size, num_embeddings=embed_size,
                                          block_size=block_size)
        self.lm_head = nn.Linear(embed_size, vocab_size).to(device).to(device)
        # to create the logits the same way , we need a linear layer to map out embed_size->vocab_size

        self.feed_forward = FeedForward(num_embed=embed_size)

        self.blocks = nn.Sequential(
            *[Block(num_embeddding=embed_size, num_head=int(embed_size // head_size), block_size=block_size)],
            nn.LayerNorm(embed_size)).to(device=device)
        self.ln_f = nn.LayerNorm(embed_size).to(device=device)
    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embedding = self.token_embedding_table(idx)  # (B,T,C)
        pos_embedding = self.position_embedding_table(torch.arange(T, device=device))  # (B,T)

        x = token_embedding + pos_embedding  # (B,T,C)

        # x = self.blocks(x)  # blocks employ multiple(self attention +feed forward)
        # x = self.sa_head(x)  # (B,T,heads_dim)
        # #
        # # # the idea is the logits didn't get the time to interact with each other and get to a good conclusion,
        # # # so that's why we have added feed forward network
        # x = self.feed_forward(x)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,head_dim) ->(B,T,C)

        if targets is None:
            loss = None
        else:

            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            # crop idx to the last block size of tokens
            idx_cond = idx[:, -self.block_size:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]  # take the last row , to predict from (B,C)
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)

            idx = torch.cat([idx, idx_next], dim=1)

        return idx
