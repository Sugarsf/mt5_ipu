from transformers import MT5ForConditionalGeneration, MT5Tokenizer 
import torch
import poptorch

class ipu_mt5(MT5ForConditionalGeneration):
    def parallelize_ipu4(self, log):

            log.logger.info("---------- Device Allocation -----------")
            log.logger.info("embedding  --> IPU 0")
            self.shared = poptorch.BeginBlock(self.shared,"Embedding...",ipu_id=0)

            #layer_ipu = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
            layer_ipu = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]
            for i,layer in enumerate(self.encoder.block):
                ipu = layer_ipu[i]
                self.encoder.block[i] = poptorch.BeginBlock(layer, f"Encoder{i}", ipu_id=ipu)
                log.logger.info(f"Encoder {i:<2} --> IPU {ipu}")
            self.encoder.final_layer_norm = poptorch.BeginBlock(
            self.encoder.final_layer_norm, "Encoder Stack Final LayerNorm", ipu_id=ipu)
        
            shift = len(self.encoder.block)    
            for i,layer in enumerate(self.decoder.block):
                ipu = layer_ipu[i + shift]
                self.decoder.block[i] = poptorch.BeginBlock(layer, f"Decoder{i}", ipu_id=ipu)
                log.logger.info(f"Decoder {i:<2} --> IPU {ipu}")        

            self.decoder.final_layer_norm = poptorch.BeginBlock(
                self.decoder.final_layer_norm, "Decoder Stack Final LayerNorm", ipu_id=ipu)    
        

            log.logger.info("Lm  --> IPU 0")
            self.lm_head = poptorch.BeginBlock(self.lm_head,"Lmhead...",ipu_id=0)
            log.logger.info("-----------------------------------------------------------")
            
            return self


class mT5(torch.nn.Module):
    '''
    A basic mT5 model 
    '''
    def __init__(self,log):
        super(mT5, self).__init__()
        self.mt5 = ipu_mt5.from_pretrained('mt5-da-small').parallelize_ipu4(log)
        # new_embeddings = torch.nn.Embedding(32128,512)
        # self.mt5.encoder.set_input_embeddings(new_embeddings)
        #self.mt5.encoder.embed_tokens.load_state_dict(model_embedding.encoder.embed_tokens.state_dict())
        
    def forward(self, input_ids, labels):
        out = self.mt5(input_ids=input_ids, labels=labels)
        
        return out

    def generate(self, input_ids):
        result = self.mt5.generate(input_ids=input_ids)

        return result


