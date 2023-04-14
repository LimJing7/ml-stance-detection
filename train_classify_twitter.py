import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.utils.data import RandomSampler, SequentialSampler
import transformers

device = torch.device('cuda')

tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('microsoft/infoxlm-base')
model = transformers.XLMRobertaModel.from_pretrained('microsoft/infoxlm-base').to(device)

search_df = pd.read_csv('./twitter_wiki_cont_samples_manual_helpful_comb.csv')

x = search_df['text']
y = search_df['useful']

y = torch.tensor([0 if i == 'n' else 1 for i in y]).double().to(device)

n_examples = len(x)
batch_size = 16
x_enc = []
nEpochs = 100

for i in range(0, n_examples, batch_size):
    text = x[i:i+batch_size]
    enc = tokenizer.batch_encode_plus(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    enc = {i: j.to(device) for i,j in enc.items()}
    output = model(**enc).pooler_output
    x_enc.append(output.detach().cpu())

x_enc = torch.cat(x_enc, dim=0)

dataset = TensorDataset(x_enc, y)
train_sp = RandomSampler(dataset)
dataloader = DataLoader(dataset, sampler=train_sp, batch_size=batch_size)

my_useful_model = torch.nn.Sequential(torch.nn.Linear(768,768), torch.nn.ReLU(), torch.nn.Linear(768,1)).to(device)
params = my_useful_model.parameters()

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params)


for epoch in range(nEpochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = my_useful_model(inputs.to(device))

        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        # print statistics
        print(f'{epoch = }')
        print(loss.item())

print('Finished Training')

torch.save(my_useful_model.state_dict(), './useful_model')