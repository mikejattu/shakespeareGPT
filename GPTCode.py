import torch
import torch.nn as nn
from torch.nn import functional as F
import selfattention as sa
def load_text():
    """
    Description: This function loads the text file and returns the data
    Parameters: None
    Returns: data (str): the text data
    """
    with open('input.txt') as f:
        data = f.read()
    #getting all the tokens available in the data
    return data

def sorted_data(data):
    """
    Description: This function returns a list of all the unique characters in the data
    Parameters: data (str): the text data
    Returns: sorted_data (list): a list of all the unique characters in the data
    """
    #sorting the data
    return sorted(list(set(data)))

def encode_string(string,sti):
    """
    Description: This function encodes a string of characters to a list of integers
    Parameters: string (str): the string of characters to be encoded
                sti (dict): a dictionary that maps characters to integers
    Returns: encoded_string (list): a list of integers
    """
    return [sti[char] for char in string]

def decode_int(encoded_string,its):
    """
    Description: This function decodes a list of integers to a string of characters
    Parameters: encoded_string (list): a list of integers
                its (dict): a dictionary that maps integers to characters
    Returns: decoded_string (str): a string of characters
    """
    return ''.join([its[i] for i in encoded_string])

def get_batches(data, batch_size, block_size):
    """
    Description: This function returns a list of input and target sequences
    Parameters: data (list): a list of integers
                batch_size (int): the number of sequences that the model is trained on
                block_size (int): the length of the sequence that the model is trained on
    Returns:    inputs (tensor): a tensor of input sequences
                targets (tensor): a tensor of target sequences
    """
    # generating random starting points for the sequences
    start_point = torch.randint(len(data) - block_size, (batch_size,))
    # creating a list of sequences
    inputs = torch.stack([data[start:start+block_size] for start in start_point])
    targets = torch.stack([data[start+1:start+block_size+1] for start in start_point])
    return inputs,targets

class BigramLanguageModel(nn.Module):
    """
    Description: This class defines the GPT model
    """
    def __init__(self,vocab_size,n_embd,block_size,device,n_layer,n_heads,dropout) -> None:
        self.device = device
        self.block_size = block_size
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # the embedding layer that takes in the tokens and returns the embeddings
        self.postion_embedding_table = nn.Embedding(block_size, n_embd) # the embedding layer that takes in the positions and returns the embeddings
        self.blocks = nn.Sequential(*[sa.Block(n_embd,block_size,n_heads,dropout) for _ in range(n_layer)])
        # adding the normalization layer
        self.normFF = nn.LayerNorm(n_embd)
        self.lmhead = nn.Linear(n_embd, vocab_size) # the output layer that takes in the embeddings and returns the logits
        
    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        tokenEmbeddings = self.token_embedding_table(idx) # (B,T,C)
        positionEmbeddings = self.postion_embedding_table(torch.arange(idx.shape[1],device = self.device)) # (T,C)
        sumEmbeddings = tokenEmbeddings + positionEmbeddings # (B,T,C)
        sumEmbeddings = self.blocks(sumEmbeddings)
        sumEmbeddings = self.normFF(sumEmbeddings)
        logits = self.lmhead(sumEmbeddings) # (B,T,V) where V is the vocab size
        if targets is None:
            loss = None 
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """
        Description: This function generates new tokens given the current context and the number of tokens to generate
        Parameters: idx (tensor): the current context
                    max_new_tokens (int): the number of tokens to generate
        Returns: idx (tensor): the new context
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the idx to the block size
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

