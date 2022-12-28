from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from datasets import load_dataset
import torch
import poptorch
from log import Logger
import datetime
from ipu_options  import get_options
from data_loader import MyDataloader, collate_fn
from args import parse_args
from optimization import get_lr_scheduler, get_optimizer
import os
import time
from mt5_model_ipu import ipu_mt5, mT5

tokenizer = MT5Tokenizer.from_pretrained('mt5-da-small')
batch_size = 1

if __name__ == "__main__":
    
    config = parse_args()


    log = Logger("./output/"+datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')+'.log', level='info')
    opts = get_options(config)
  
    model = mT5(log).train().half()

    optimizer = get_optimizer(config, model)

    hg_dataset = load_dataset("parquet", data_files={'train': 'mt5-small-finetuned-amazon-en-es_books_dataset/data/train-00000-of-00001.parquet'})
    dataset = MyDataloader(hg_dataset['train'])     
                                
    training_data = poptorch.DataLoader(options=opts,
                                        dataset=dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        collate_fn=collate_fn
                                        )

    #optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    steps_per_epoch = len(training_data)

    start_epoch = 0
    epochs = config.epochs
    factor = config.gradient_accumulation * config.device_iterations
    training_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * config.warmup_epochs

    scheduler = get_lr_scheduler(optimizer, config.lr_schedule,
                                 warmup_steps, training_steps)
    poptorch_model = poptorch.trainingModel(model,
                                            options=opts,
                                            optimizer=optimizer)
    # Compile model
    log.logger.info("---------- Compilation Started ---------")                                            
    start_compile = time.perf_counter()
    datum = next(iter(training_data))
    poptorch_model.compile(torch.tensor(datum['text']),torch.tensor(datum['label']))

    duration_compilation = time.perf_counter() - start_compile
    log.logger.info(f"Compiled model in {duration_compilation} secs")
    log.logger.info("---------------------------------------")

    # Track approx. IPU compute time
    total_compute_time = 0


    for epoch in range(start_epoch, epochs):
        start_step = time.perf_counter()        
        for step,batch in enumerate(training_data):
            current_step = step + epoch * steps_per_epoch      
            input_ids = torch.tensor(batch['text'])
            sql_ids = torch.tensor(batch['label'])
            losses = poptorch_model(input_ids=input_ids, labels=sql_ids).loss
            scheduler.step() 
            poptorch_model.setOptimizer(optimizer)
            step_length = time.perf_counter() - start_step
            step_throughput = config.samples_per_step / step_length

            if step > 0 or epoch > 0:
                total_compute_time += step_length

            log.logger.info("Epoch: {:.2f}/{} Step: {}/{} Lr: {:.6f} loss: {:.3f} throughput: {:.2f} samples/sec"
                            .format(epoch, epochs, current_step, training_steps, scheduler.get_last_lr()[0], losses.mean(), step_throughput))
            start_step = time.perf_counter()


  
    