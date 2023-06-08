# PipeNet-PyTorch : RGB-D Face Anti-Spoofing Project
This is PyTorch code implementation for CVPR2020 workshop [paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w39/Yang_PipeNet_Selective_Modal_Pipeline_of_Fusion_Network_for_Multi-Modal_Face_CVPRW_2020_paper.pdf) **"PipeNet: Selective Modal Pipeline of Fusion Network for Multi-Modal Face Anti-Spoofing"**.
* This approach won **Global 3rd place** in **@CVPR2020** Chalearn Multi-modal Cross-ethnicity Face anti-spoofing Recognition Challenge  (Multi_modal track).

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
```bash
### python main.py --mode=train --dataset_name=4@1 
### python main.py --mode=train --dataset_name=4@2 
### python main.py --mode=train --dataset_name=4@3 
```

# Test
```bash
### python main.py --image_mode=fusion --mode=dev
### python main.py --image_mode=fusion --mode=test
```


# Citation

```
@inproceedings{pipenet,
  title={PipeNet: Selective Modal Pipeline of Fusion Network for Multi-Modal Face Anti-Spoofing},
  author={Yang, Qing and Zhu, Xia and Fwu, Jong-Kae and Ye, Yun and You, Ganmei and Zhu, Yuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={644--645},
  year={2020}
}
```

-----------------------------------------------------------
Any question, pls contact email: charles.q.yang@gmail.com or wechat: kim_young  .
