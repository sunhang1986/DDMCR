Dual Discriminators and Multi-level Channel Recalibration for Unpaired Outdoor Image Defogging Network
==
Created by Shuailing Fang, [Hang Sun](https://github.com/sunhang1986), Zhiping Dan from Department of Computer and Information, China Three Gorges University.

Introduction
--
 Our paper proposes an unpaired outdoor image defogging network based on dual discriminators and multi-channel recalibration. The proposed network includes a dual discriminator mechanism and a multi-channel attention module.

Prerequisites
--
+ Pytorch 1.7.1
+ Python 3.6.12
+ CUDA 8.0
+ Ubuntu 18.04

Test
--
[Download](https://sites.google.com/view/reside-dehaze-datasets) RESIDE dataset . [Download](https://labsites.rochester.edu/gsharma/research/computer-vision/hazerd/) Hazerd dataset . [Download](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/) Ohaze dataset . 
Test the model on *RESIDE*:

` python   test.py   --cuda --gpus 0,1 --test --test_ori --test_path test_imgpath --Gx1_model_path premodel/epoch_ots_Gx1.pth` 
Test the model on *Hazerd*:

` python   test.py   --cuda --gpus 0,1 --test --test_ori --test_path test_imgpath --Gx1_model_path premodel/epoch_hazerd_Gx1.pth` 
Test the model on *Ohaze*:

` python   test.py   --cuda --gpus 0,1 --test --test_ori --test_path test_imgpath --Gx1_model_path premodel/epoch_ohaze_Gx1.pth` 


