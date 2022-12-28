#  install Dependencies
pip install -r  requirements.txt  

#  download model and dataset
git lfs install  
git clone https://huggingface.co/sarakolding/mt5-da-small  
git clone https://huggingface.co/Datasets/Amba/mt5-small-finetuned-amazon-en-es_books_dataset  


#  Run code
python train_mt5_ipu.py  
