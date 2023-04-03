import nltk
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import yaml
from yaml.loader import SafeLoader
from blocks import BigramModel, Head

with open('./config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)
device = "cuda" if torch.cuda.is_available() else "cpu"

# print(text[:1000])

# sentence _vectorize
# sentences = nltk.sent_tokenize(text)
#
# words = nltk.word_tokenize(text)

block_size = 256  # what is the maximum context length for the predictions
batch_size = 16  # how many independent sequence will we process in parallel
num_embeddings = 32
learning_rate = 3e-4
num_heads = 6
n_layers = 6  # how many layers of blocks will we have
max_iter = 5000

# so if we see , the whole text is converted to a list of words, which we can use to actually
# taking set
class GetCorpus:

    def __init__(self, path, encoding="utf-8"):
        with open(path, encoding=encoding, errors="backslashreplace") as f:
            self.text = f.read()

        # self.words = nltk.word_tokenize(self.text)
        self.words = self.text.split()
        # print("the self words are", self.words)

        self.distinct_words = sorted(list(set(self.words)))
        self.distinct_words.extend(["<PAD>", "<UNK>", " ", "|", "<SOS>", "<EOS>"])

        self.stoi = {ch: i for i, ch in enumerate(self.distinct_words)}
        self.itos = {i: ch for i, ch in enumerate(self.distinct_words)}

    def create_dictionary(self):
        return self.distinct_words, self.stoi, self.itos

    def corpus_tensor(self):
        return torch.tensor(self.encode(self.text), dtype=torch.long).to(device)

    def encode(self, input_string):
        encoded_vector = [self.stoi[word] for word in input_string.split()]
        return encoded_vector

    def decode(self, input_vector):
        # its coming list of list so , we will convert it into list
        # print("the input vector is ",input_vector[0])
        # input_vector = input_vector[0]
        print("the input vector ",input_vector)
        decoded_text = [self.itos[vec] for vec in input_vector]
        return decoded_text


get_corus = GetCorpus(path="draculabr00stokuoft_djvu.txt")
distinct_words, stoi, itos = get_corus.create_dictionary()


def create_train_test_split(corpus, percentage: int = 90):
    n = int((percentage / 100) * len(corpus))
    # print(" the corpus is ", corpus)
    train_data = corpus[:n]
    val_data = corpus[n:]

    return train_data, val_data


# train_data, val_data = create_train_test_split(get_corus.corpus_tensor(), 95)


# print(train_data[:block_size + 1])


def get_batch(split, batch_size):
    # generate a small batch of inputs x and targets y
    train_data, val_data = create_train_test_split(get_corus.corpus_tensor(), 99)
    data = train_data if split == "train" else val_data
    # print("len of data is ", len(data))
    ix = torch.randint(len(data) - block_size, (batch_size,)).to(device)
    # print("what is ix exactly", ix)
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)

    return x, y


x_batch, y_batch = get_batch("train", batch_size=batch_size)


class BiGramLanguageModel(nn.Module):

    def __init__(self, vocab_size, embed_size=500, device=device):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, embed_size).to(device)

        # so in the bigram model , to create the logits the same way , we need a linear layer

        self.lm_head = nn.Linear(embed_size, vocab_size).to(device)

        # introducing the positional embedding
        # Each position from timestep 0-8 will get its own positional encoding
        self.position_embedding = nn.Embedding(block_size, num_embeddings)

    def forward(self, idx, targets=None):
        # idx: (B,T) target: (B,T)
        B, T = idx.shape
        # Based on the token Embedding at the index x , we can know a lot more of what will
        # come next , just based on what this token actually is
        token_embeddings = self.token_embedding_table(idx)  # (B,T,embed_size)

        pos_embeddings = self.position_embedding(torch.arange(T))

        x = token_embeddings + pos_embeddings
        logits = self.lm_head(x)  # (B,T,C) C-> vocab size

        if targets is None:
            loss = None
        else:
            # since cross entrpopy ( FROM TORCH EMBEDDINGS SOURCE) expects B,C,T
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # squish the batches * length into 1D sequence ( so finally making it 2D)
            targets = targets.view(B * T).to(device)

            # calculate the loss
            loss = F.cross_entropy(logits, targets)
            # Quality of logits , How well are we measuring logits , given we already know the target Ideally in the
            # lookup table , the loss corresponding to the target should be really low and other things should be high

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(idx)

            # logits dimension: is target is None:(B,T,C) batch,timestep,channels
            # take the last timestep

            logits = logits[:, -1, :]  # Logits: B,C taking the last step !
            probs = F.softmax(logits, dim=1)

            idx_next = torch.multinomial(probs, num_samples=1)  # Batch ,1

            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx


# model = BiGramLanguageModel(vocab_size=len(distinct_words), device=device)
# print(f"dict:{distinct_words} ,embed_size:{num_embeddings},block_size:{block_size}")
model = BigramModel(vocab_size=len(distinct_words),
                    embed_size=num_embeddings, block_size=block_size)
logits, loss = (model(x_batch, y_batch))  # shape is B,T,C = (4,8,500)

print(logits.shape)
print(f"loss shape is {loss}")

idx = torch.zeros((1, 1), dtype=torch.long).to(device)  # idx with Batch =1, timestep = 1 holding value=0
print("generated sequence")
generated_text = model.generate(idx, max_new_tokens=100)

decoded_text = get_corus.decode(generated_text[0].tolist())
print("Before training ...\n",decoded_text)
# let's optimize this bad boii

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for steps in tqdm(range(max_iter)):
    x_batch, y_batch = get_batch("train", batch_size=batch_size)

    logits, loss = model(x_batch, y_batch)
    loss.backward()
    optimizer.step()

# saving the model checkpoint

print(loss.item())
generated_text = model.generate(idx, max_new_tokens=100)
decoded_text = get_corus.decode(generated_text[0].tolist())

print("After training ...\n",decoded_text)