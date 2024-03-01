# MAUN
The code of MAUN: Memory-Augmented Deep Unfolding Network for Hyperspectral Image Reconstruction

Published in IEEE/CAA Journal of Automatica Sinica 24

#### Recommended Environment:<br>

 - [ ] python = 3.9.16
 - [ ] torch = 1.10.0
 - [ ] numpy = 1.23.5
 - [ ] scipy = 1.9.1
 - [ ] scikit-image = 0.17.2

#### Training :<br>
- [ ] Run "python train.py --template MAUN --outf ./exp/MAUN/ --method MAUN_24stg 

#### Testing :<br>
- [ ] Run "python test.py --template MAUN--outf ./exp/Visual/ --method MAUN --pretrained_model_path ./Pretrained_Model/MAUN.pth" to reconstruct the 3D HSI from compressive measurement.
- [ ] The test results will be saved in ./exp/Visual.
- [ ] Then run  Cal_quality_assessment.m  to caculate the PSNR and SSIM of output HSIs.

#### Acknowledgements :<br>
This work's implementation is based on "A Toolbox for Spectral Compressive Imaging"  
For experiment setting, dataset and visualization, you can also refer to https://github.com/caiyuanhao1998/MST
```
@inproceedings{mst,
  title={Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction},
  author={Yuanhao Cai and Jing Lin and Xiaowan Hu and Haoqian Wang and Xin Yuan and Yulun Zhang and Radu Timofte and Luc Van Gool},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
