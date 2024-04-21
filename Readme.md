# MAUN
The code of MAUN: Memory-Augmented Deep Unfolding Network for Hyperspectral Image Reconstruction

Published in IEEE/CAA Journal of Automatica Sinica 24

## Overview :<br>
Deep unfolding-based methods have dominated spectral compressive imaging in recent years. However, their performance is still limited by the weak information transmission between iteration stages. To address this issue, we propose a Memory-Augmented Deep Unfolding Network, termed MAUN, for explainable and accurate HSI reconstruction. Specifically, MAUN implements a novel CNN scheme to facilitate a better extrapolation step of the fast iterative shrinkage-thresholding algorithm, introducing an extra momentum incorporation step for each iteration to alleviate information loss. Moreover, to exploit the high correlation of intermediate images from neighboring iterations, we customize a cross-stage transformer (CSFormer) as the deep denoiser to simultaneously capture self-similarity from both in-stage and cross-stage features, which is the first attempt to model the long-distance dependencies between iteration stages. Extensive experiments demonstrate that the proposed MAUN is superior to other state-of-the-art methods both visually and metrically.

## Comparison with other SOTA methods :<br>
![image](https://github.com/HuQ1an/MAUN/assets/86952915/381df459-d776-4277-8ea0-4319273b34c3)

![image](https://github.com/HuQ1an/MAUN/assets/86952915/4827666e-df9b-4552-828d-c73da91484a6)

## Training :<br>
- [ ] Run "python train.py --template MAUN --outf ./exp/MAUN/ --method MAUN_24stg 

## Testing :<br>
- [ ] Run "python test.py --template MAUN--outf ./exp/Visual/ --method MAUN --pretrained_model_path ./Pretrained_Model/MAUN.pth" to reconstruct the 3D HSI from compressive measurement.
- [ ] The test results will be saved in ./exp/Visual.
- [ ] Then run  Cal_quality_assessment.m  to caculate the PSNR and SSIM of output HSIs.

## Recommended Environment:<br>

 - [ ] python = 3.9.16
 - [ ] torch = 1.10.0
 - [ ] numpy = 1.23.5
 - [ ] scipy = 1.9.1
 - [ ] scikit-image = 0.17.2

## Experiment results :<br>
 [Model zoom](https://pan.baidu.com/s/1911G9IRRKDIYXYgDs2Ey2g?pwd=3kch)
 
 [Simulation results](https://pan.baidu.com/s/13PCmWgXiWHiH8wVgHOYa4A?pwd=kp5t)
 
 [Real results](https://pan.baidu.com/s/1mApjGHPJcR4hsgVE9gcbNg?pwd=th9g)

## Citation :<br>

## Contact :<br>
If you have any questions, please feel free to contact me at huq1an@whu.edu.cn

## Acknowledgements :<br>
This work's implementation is based on "A Toolbox for Spectral Compressive Imaging".
For experiment setting, dataset and visualization, you can also refer to https://github.com/caiyuanhao1998/MST.
```
@inproceedings{mst,
  title={Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction},
  author={Yuanhao Cai and Jing Lin and Xiaowan Hu and Haoqian Wang and Xin Yuan and Yulun Zhang and Radu Timofte and Luc Van Gool},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

## Citation :<br>
```
@article{hu2024maun,
  title={MAUN: Memory-Augmented Deep Unfolding Network for Hyperspectral Image Reconstruction},
  author={Hu, Qian and Ma, Jiayi and Gao, Yuan and Jiang, Junjun and Yuan, Yixuan},
  journal={IEEE/CAA Journal of Automatica Sinica},
  volume={11},
  number={5},
  pages={1139--1150},
  year={2024},
  publisher={IEEE}
}
```
