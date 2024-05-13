import GPTCode as cd
import torch
# loading the text file
data = cd.load_text()
# getting all the unique characters from the text
chars = cd.sorted_data(data)
# creating vocabulary size
vocab_size = len(chars)
# creating a dictionary that maps characters to integers
sti = {char:i for i,char in enumerate(chars)}
# creating a dictionary that maps integers to characters
its = {i:char for i,char in enumerate(chars)}

# creating a tensor of integers from the text
encoded_data = torch.tensor(cd.encode_string(data,sti), dtype=torch.long)

# splitting the data into training and validation sets
train_data = encoded_data[:int(0.9*len(encoded_data))]
val_data = encoded_data[int(0.9*len(encoded_data)):]

#-----------------------------------------------------------
# defining the hyperparameters:

# defining the batch size
batch_size = 64 # the number of sequences that the model is trained on

# defining the block size
block_size = 256 # the length of the sequence that the model is trained on

# defining the number of iterations
iterations = 5000 # the number of iterations that the model is trained for
# defining the interval
interval = 500 # the interval at which the model is evaluated
# defining the learning rate
lr = 3e-4 # the rate at which the model learns

# defining the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps' if torch.backends.mps.is_available() else device

# defining the number of embeddings
n_embd = 384
n_heads = 6
n_layer = 6
dropout = 0.2

#-----------------------------------------------------------

# getting the input and target sequences
inputs,targets = cd.get_batches(train_data,batch_size,block_size)
# sending the input and target sequences to the device
inputs,targets = inputs.to(device),targets.to(device)
# defining the model
model = cd.BigramLanguageModel(vocab_size,n_embd,block_size,device, n_layer, n_heads, dropout)
# moving the model to the device
model.to(device)

@torch.no_grad() # this is to prevent PyTorch from building the computation graph
def estimateLoss():
    """
    Description: This function estimates the loss of the model by averging the loss over multiple batches
    Parameters: iterations (int): the number of iterations to estimate the loss
    Returns: loss (dictionary): the estimated loss of training and validation sets
    """
    loss_dict = {'train':0,'val':0}
    model.eval()
    for data in ["train","val"]:
        losses = torch.zeros(iterations)
        for i in range(iterations):
            Fdata = train_data if data == 'train' else val_data
            inputs,targets = cd.get_batches(Fdata,batch_size,block_size)
            inputs,targets = inputs.to(device),targets.to(device)
            logits,loss = model(inputs,targets)
            losses[i] = loss.item()
        loss_dict[data] = losses.mean()
    return loss_dict

# defining the optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

# training the model
for _ in range(iterations):
    # print the loss every once in a while
    if _ % interval        == 0:
        loss = estimateLoss()
        print(f'iteration: {_}, train loss: {loss["train"]}, val loss: {loss["val"]}')
    # getting the input and target sequences
    inputs,targets = cd.get_batches(train_data,batch_size,block_size)   
    # sending the input and target sequences to the device
    inputs,targets = inputs.to(device),targets.to(device)
    # getting the logits and loss
    logits,loss = model(inputs,targets)
    # zeroing the gradients from the previous iteration
    optimizer.zero_grad(set_to_none=True)
    # calculating the gradients
    loss.backward()
    # updating the weights using the gradients
    optimizer.step()

# generating text

value = model.generate(idx = torch.zeros((1,1), device=device, dtype=torch.long),max_new_tokens = 500)[0].tolist()
# writng the decoded text to a file
with open('output.txt','w') as f:
    f.write(cd.decode_int(value,its))
