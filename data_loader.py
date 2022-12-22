from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from torch.utils.data import Dataset


max_source_length = 256
max_target_length = 50
tokenizer_path = 'dansum-mt5-base-v1'

class MyDataloader(Dataset):
    def __init__(self,data_set):
        self.data_set = data_set
        self.length = len(data_set)
        self.tokenizer = tokenizer = MT5Tokenizer.from_pretrained(tokenizer_path)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        D = self.data_set[index]

        encode1 = self.tokenizer.encode_plus(D['review_body'],
                              None, # pad to the longest sequence in the batch
                              max_length=max_source_length,
                              add_special_tokens=True,
                              pad_to_max_length=True,
                              truncation=True
                              )
        encode2 = self.tokenizer.encode_plus(D['review_title'],
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
    for unit in data:
        text.append(unit['input_ids'])
        label.append(unit['labels'])
    return {'text':text,'label':label}
