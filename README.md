# Self-Distilled Hierarchical Network

**Self-Distilled Hierarchical Network for Unsupervised Deformable Image Registration**

IEEE Transactions on Medical Imaging (TMI)

Shenglong Zhou, Bo Hu, Zhiwei Xiong and Feng Wu

University of Science and Technology of China (USTC)

## Introduction

![Framework](https://user-images.githubusercontent.com/26156941/201927630-23340d83-52a0-45b6-a007-19c7fb603ea9.png)

We present a novel unsupervised learning approach named Self-Distilled Hierarchical Network (SDHNet).  
By decomposing the registration procedure into several iterations, SDHNet generates hierarchical deformation fields (HDFs) simultaneously in each iteration and connects different iterations utilizing the learned hidden state.
Hierarchical features are extracted to generate HDFs through several parallel GRUs, and HDFs are then fused adaptively conditioned on themselves as well as contextual features from the input image.
Furthermore, different from common unsupervised methods that only apply similarity loss and regularization loss, SDHNet introduces a novel self-deformation distillation scheme. 
This scheme distills the final deformation field as the teacher guidance, which adds constraints for intermediate deformation fields.



## Requirements
The packages and their corresponding version we used in this repository are listed in below.
- Python 3
- Pytorch 1.1
- Numpy
- SimpleITK

## Training
After configuring the environment, please use this command to train the model.
```python
python -m torch.distributed.launch --nproc_per_node=4 train.py  --name=SDHNet  --iters=6 --dataset=brain  --data_path=/xx/xx/  --base_path=/xx/xx/

```
## Testing
Use this command to obtain the testing results.
```python
python eval.py  --name=SDHNet  --model=SDHNet_lpba --dataset=brain --dataset_test=lpba  --iters=6 --local_rank=0 --data_path=/xx/xx/  --base_path=/xx/xx/
```

## Datasets and Pre-trained Models (Based on Cascade VTN)
We follow Cascade VTN to prepare the training and testing datasets, please refer to [Cascade VTN](https://github.com/microsoft/Recursive-Cascaded-Networks) for details.

The related [pretrained models](https://drive.google.com/drive/folders/1BpxkIzL_SrPuKdqC_buiINawNZVMqoWc?usp=share_link) are available, please refer to the testing command for evaluating.


## Note
Due to our further exploration of the self-distillation, the current codebase does not involve the related part temporarily.

Please be free to contact us by e-mail if you are interested in this part or have any problems.


## Acknowledgment
We follow the functional implementation in [Cascade VTN](https://github.com/microsoft/Recursive-Cascaded-Networks), and the overall code framework is adapted from [RAFT](https://github.com/princeton-vl/RAFT).  

Thanks a lot for their great contribution!
