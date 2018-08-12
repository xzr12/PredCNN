# PredCNN
This is a TensorFlow implementation of [PredCNN](https://www.ijcai.org/proceedings/2018/0408.pdf), an entire convolutional model for video prediction as described in the following paper: 

**PredCNN: Predictive Learning with Cascade Convolutions**, by Ziru Xu, Yunbo Wang, Mingsheng Long and Jianmin Wang.

## Setup
Required python libraries: tensorflow (>=1.0) + opencv + numpy.
Tested in ubuntu/centOS + nvidia titan X (Pascal) with cuda (>=8.0) and cudnn (>=5.0).

## Datasets
We conduct experiments on one video dataset, [Moving Mnist](https://1drv.ms/f/s!AuK5cwCfU3__fGzXjcOlzTQw158), and two crowd flow dataset, [TaxiBJ and BikeNYC](https://github.com/lucktroy/DeepST/tree/master/data).
For other video format datasets, please extract frames from original video clips and move them to the `data/` folder.

## Training
Use the `scripts/train.sh` script to train the model. To train the default model on Moving MNIST simply use:
```shell
cd scripts;sh train.sh;
```
You might want to change the `--train_data_paths`, `--valid_data_paths` and `--save_dir` which point to paths on your system to download the data to, and where to save the checkpoints.

To train on your own dataset, have a look at the `InputHandle` classes in the `data_provider/` folder. You have to write an analogous iterator object for your own dataset. 

At inference, the generated future frames will be saved in the `--results` folder.

## Citation
Please cite the following paper if you find this repository useful.
```latex
@inproceedings{xu2018predcnn,
  title={PredCNN: Predictive Learning with Cascade Convolutions.},
  author={Xu, Ziru and Wang, Yunbo and Long, Mingsheng and Wang, Jianmin and KLiss, MOE},
  booktitle={IJCAI},
  pages={2940--2947},
  year={2018}
}
```

## Contact

If you have any problem about our code, feel free to contact xzr12thss@gmail.com or describe your problem in Issues.

