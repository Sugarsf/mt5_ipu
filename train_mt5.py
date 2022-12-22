from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader,Dataset


tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
max_source_length = 256
max_target_length = 50
batch_size = 1

class MyDataloader(Dataset):
    def __init__(self,data_set):
        self.data_set = data_set
        self.length = len(data_set)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        D = self.data_set[index]

        encode1 = tokenizer.encode_plus(D['review_body'],
                              None, # pad to the longest sequence in the batch
                              max_length=max_source_length,
                              add_special_tokens=True,
                              pad_to_max_length=True,
                              truncation=True
                              )
        encode2 = tokenizer.encode_plus(D['review_title'],
                              None, # pad to the longest sequence in the batch
                              max_length=max_target_length,
                              add_special_tokens=True,
                              pad_to_max_length=True,
                              truncation=True
                              )
        input_ids = encode1['input_ids']
        sql_ids = encode2['input_ids']
        return {'input_ids':input_ids,'labels':sql_ids}

        
def collate_fn(data):   
    text = []
    label = []
    mask = []
    for unit in data:
        text.append(unit['input_ids'])
        label.append(unit['labels'])
    return {'text':text,'label':label}

    
class mT5(torch.nn.Module):
    '''
    A basic mT5 model 
    '''
    def __init__(self):
        super(mT5, self).__init__()
        self.mt5 = MT5ForConditionalGeneration.from_pretrained('mt5-small')
        
    def forward(self, input_ids, labels):
        out = self.mt5(input_ids=input_ids, labels=labels)
        
        return out
    
    def generate(self, input_ids):
        result = self.mt5.generate(input_ids=input_ids)
        
        return result


model = mT5()

hg_dataset = load_dataset("parquet", data_files={'train': 'mt5-small-finetuned-amazon-en-es_books_dataset/data/train-00000-of-00001.parquet'})
dataset = MyDataloader(hg_dataset['train'])          

train_loader = DataLoader(dataset,batch_size,shuffle=True,collate_fn=collate_fn)
datum = next(iter(train_loader))
#torch.onnx.export(model,(torch.tensor(datum['text']),torch.tensor(datum['label'])),"mt5.onnx",opset_version=12)

optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
device = torch.device('cpu')
model.train()
#print (model)

model = model.to(device)

for epoch in range(100):
    for i,batch in enumerate(train_loader):
        optim.zero_grad()
        input_ids = torch.tensor(batch['text'])
        sql_ids = torch.tensor(batch['label'])
        loss = model(input_ids=input_ids,labels=sql_ids).loss
        loss.backward()
        optim.step()
        if epoch % 10 == 0 and i % 10 == 0:
            print("Epoch: ", epoch, " , step: ", i)
            print("training loss: ", loss.item())


