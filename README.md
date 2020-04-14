# Face Anti-Spoofing Project
This is PyTorch code implementation for Chalearn Multi-modal Cross-ethnicity Face anti-spoofing Recognition Challenge@CVPR2020.
Containing both multi-modal and single-modal(RGB) competitation source. 

# Prerequisites

We use Anaconda3 with python > 3.6 , dependencies as below :

opencv-python 3.4.2

pytorch==1.2.0

imutils==0.5.3

scipy==1.2.1

numpy==1.18.1

tqdm==4.36.1

imgaug==0.2.6

### Change CASIA-CeFA dataset ROOT PATH in code:

in line 5 of  <-PROJECT ROOT->/data_helper.py file:

Replace <...> content in  "DATA_ROOT = r'/<-root directory to your dataset->/CASIA-CeFA/'"

###  Dataset in Below structure:


+-- CASIA-CeFA

    +-- phase1

        +-- train

        +-- dev

    +-- phase1

        +-- test


<-PROJECT ROOT->/dataset/* contains splitted and shuffled file lists for train/val.  
The tool for this work is under ./tools/train_filelist.ipynb


# Train 

CUDA_VISIVLE_DEVICES=0 python main.py --mode=train --dataset_name=4@1 
CUDA_VISIVLE_DEVICES=1 python main.py --mode=train --dataset_name=4@2 
CUDA_VISIVLE_DEVICES=2 python main.py --mode=train --dataset_name=4@3 

# Test

python main.py --image_mode=fusion --mode=dev

python main.py --image_mode=fusion --mode=test


# submission

python main.py --image_mode=fusion --mode=submit


---------------------------------------------------------------
Update on 2020.3.1 

Make the train/val system more automated and less commands and parameters;

Clean whole project source code;

Enrich information in README.md file.

-----------------------------------------------------------
Any question, pls contact email: charles.q.yang@gmail.com or wechat: kim_young  .
