from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch
import pdb

#pdb.set_trace()
model = MT5ForConditionalGeneration.from_pretrained("./mt5-small")
tokenizer = MT5Tokenizer.from_pretrained("./mt5-small")

max_source_length = 128
max_target_length = 128

input_sequence1 = "Welcome to Shenzhen"
output_sequence1 = "欢迎来到深圳"


input_sequence2 = "HuggingFace is a company"
output_sequence2 = "抱脸是一家公司"

task_prefix = "translate English to Chinese: "

input_tokens1 = tokenizer.tokenize(task_prefix + input_sequence1)


print ("int_token1 = ",input_tokens1)
output_tokens1 = tokenizer.tokenize(output_sequence1)
print ("out_token1 = ",output_tokens1)

input_tokens2 = tokenizer.tokenize(task_prefix + input_sequence2)
output_tokens2 = tokenizer.tokenize(output_sequence2)

input_sequence = [input_sequence1,input_sequence2]

encoding = tokenizer(
    [task_prefix + sequence for sequence in input_sequence],
    padding = "longest",
    max_length = max_source_length,
    truncation = True,
    return_tensors = "pt",
)

test_sent = "translate: The sailor was happy."
test_tokenized = tokenizer(test_sent, return_tensors="pt")
test_input_ids = test_tokenized["input_ids"].type(torch.int32)
test_attention_mask = test_tokenized["attention_mask"].type(torch.int32)

test_result = "水手很高兴"
test_result_tokenized = tokenizer(test_result, return_tensors="pt")
test_result_ids = test_result_tokenized["input_ids"].type(torch.int32)


print (test_input_ids.shape)
print (test_attention_mask.shape)
print (test_result_ids.shape)
exit()
torch.onnx.export(model,
                 (test_input_ids,test_attention_mask,test_result_ids),                 
                 "mt5.onnx",
                 opset_version=10                
)

exit()

model.eval()
beam_outputs = model.generate(
    input_ids=test_input_ids,attention_mask=test_attention_mask,
)


sent = tokenizer.decode(beam_outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
print (sent)
