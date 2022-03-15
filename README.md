# Introduction
This repository was used in our paper:  
  
**Have my arguments been replied to? Argument Pair Extraction as Machine Reading Comprehension**
<br>
Jianzhu Bao, Jingyi Sun, Qinglin Zhu, Ruifeng Xu<sup>\*</sup>. *Proceedings of ACL 2022*
  
Please cite our paper and kindly give a star for this repository if you use this code.

# Prerequisites
python 3.7  
pytorch 1.8.0
cuda 10.2  
transformers 2.11.0  
# Descriptions

**data_v2** - contains dataset about rr-submission-v2 and rr-passage.

**longformer-base**: put the download Pytorch longformer model here (config.json, pytorch_model.bin, vocab.json,merges.txt) (https://huggingface.co/allenai/longformer-base-4096/tree/main). 

**saved_models_n** - filefold to contain saved models, training logs and results.  

**preprocessing** - transfer the raw data into train samples  
* ```prepare_data.py```: parameter setting. 
* ```to_bioes.py```: transfer the labels to bioes mode
* ```make_train_samples.py``` - match the questions and the corresponding contexts to make train samples

**utils** - utils code.  
* ```config.py```: parameter setting. 
* ```decode.py```: the span extraction functions.
* ```classifier.py``` - the classifer for span prediction.  
* ```models.py```: the main model.

# Usage
```
python prepare_data.py 
python to_bioes.py 
python make_train_samples.py 
python run.py 
```
