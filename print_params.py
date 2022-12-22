from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch

tokenizer = MT5Tokenizer.from_pretrained('mt5-small')
model1 = MT5ForConditionalGeneration.from_pretrained("mt5-small")
model2 = MT5ForConditionalGeneration.from_pretrained("t5-base")

for key,value in model1.state_dict().items():
    print (key)
    print (value.shape)
print (model1.state_dict()['shared.weight'].shape)
#print (model1.state_dict()['encoder.embed_tokens.weight'].shape)
#print (model2.state_dict()['shared.weight'].shape)

print ("model2.................")
for key,value in model2.state_dict().items():
    print (key)
    print (value.shape)
