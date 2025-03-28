# UGNA-VPR: A Novel Training Paradigm for Visual Place Recognition Based on Uncertainty-Guided NeRF Augmentation


## Citation
If you use our code, please cite the corresponding paper:
```
@ARTICLE{shen2025ral,
  author={Shen, Yehui and Zhang, Lei and Li, Qingqiu and Zhao, Xiongwei and Wang, Yue and Lu, Huimin and Chen, Xieyuanli},
  journal={IEEE Robotics and Automation Letters}, 
  title={UGNA-VPR: A Novel Training Paradigm for Visual Place Recognition Based on Uncertainty-Guided NeRF Augmentation}, 
  year={2025},
  volume={1},
  number={1},
  pages={1-8},
  doi={10.1109/LRA.2025.3554105}
}
```

## Datasets

- The Cambridge dataset can be referenced at [DFNet](https://github.com/activevisionlab/dfnet).
- The LIB and CON datasets were recorded using a smartphone. You can download the original videos of the LIB dataset at [LIB](https://kaggle.com/datasets/9e09bf4ac09c14f13af80e55fd273114351bba1e1a8a49554de7fe7b58142d99), and the original videos of the CON dataset at [CON](https://kaggle.com/datasets/cacfaab9b6c68e3cc0ad0ed2b425302f47d302a789d9dc8c559255a67baa817d).
- Dataset for Place recognition should put in UGNA-VPR/Cambridge/xxxx
If you need datasets processed with Nerfsutido for NeRF training or for place recognition tasks, feel free to contact me.

Training Date for UE at [dataset for UE](https://kaggle.com/datasets/1b2bb5b6d07c5d562bd2662c8dc88f72e41f37b4f943cf45936f7cd6b052a114). It should be put in data/, such as UGNA-VPR/data/dataset/Cambridge.
  
## Pre-trained model
- Our pre-trained NeRF-h model at [nerfh](https://kaggle.com/datasets/c89a647ea14d5a753e7f2169e1b639e72f14cd3a09d0fbf707483b28eae47a90), it should be put in logs/, such as UGNA-VPR/logs/nerfh/shop1_4
- Our pre-trained VPR model at [VPR-model](https://kaggle.com/datasets/3b7d611f65b241de5a4573c26591e58a3db7751ca969de03873ae8596330e778), it should be put in logs/, such as UGNA-VPR/logs/MixVPR/Cambridge/ckpt_best.pth.tar
- Our pre-trained UE model at [UE-model](https://kaggle.com/datasets/680047c599214408c1d35559f0757acf7b487634c4c56f894a0af620df8f5f66), it should be put in logs/, such as UGNA-VPR/logs/Cam_MixVPR_NBP
- our results and its pre-trained model at [ours](https://kaggle.com/datasets/f3e5f27facd26691b415844b1d165f458b44a1aee7e9b1107f4c2d55b9984d00), it should be put in logs/, such as as UGNA-VPR/logs/tri_train_tea_1210_115010

## Train VPR models
In training mode
```shell
self.parser.add_argument('---split', type=str, default='val', help='Split to use', choices=['val', 'test'])
```
```shell
# train the teacher net
python main.py --phase=train_tea
```
## Train UE models
```shell
python train_npn.py -M [name of model] --setup_cfg_path config/Cambridge_training_setup.yaml
```

## Test results 
In test mode
```shell
self.parser.add_argument('---split', type=str, default='val', help='Split to use', choices=['val', 'test'])
```
 needs to be changed to 
```shell
self.parser.add_argument('---split', type=str, default='test', help='Split to use', choices=['val', 'test'])
```
```shell
python main.py --phase=test_tea	 --resume=logs/teacher_triplet/ckpt.pth.tar
```
 **The teacher_triplet/ckpt.pth.tar in the code needs to be changed to the appropriate name**.
