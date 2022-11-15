# Self-Distilled Hierarchical Network

**Self-Distilled Hierarchical Network for Unsupervised Deformable Image Registration**

Shenglong Zhou, Bo Hu, Zhiwei Xiong and Feng Wu

University of Science and Technology of China (USTC)

## Requirements
The packages and their corresponding version we used in this repository are listed in below.
- Python 3
- Pytorch 1.1
- Numpy
- SimpleITK

## Training
After configuring the environment, please use this command to train the model.
```python
python -m torch.distributed.launch --nproc_per_node=4 train.py  --name=SDHNet  --iters=6 --dataset=brain  --dataset_val=lpba_val   --data_path=/xx/xx/  --base_path=/xx/xx/

```
## Testing
Use this command to obtain the testing results.
```python
python eval.py  --name=SDHNet  --restore_step=x --iters=6 --dataset=brain  --dataset_val=lpba_val   --data_path=/xx/xx/  --base_path=/xx/xx/
```

## Acknowledgment
We follow the functional implementation in [Cascade VTN](https://github.com/microsoft/Recursive-Cascaded-Networks), and the overall code framework is adapted from [RAFT](https://github.com/princeton-vl/RAFT).

Thanks a lot for their great contribution!
