# GMFSS_Fortuna

The All-In-One GMFSS: Dedicated for Anime Video Frame Interpolation

---

- The optimised training process is more stable.
- We offer several models for inference or as pre-training models for finetuning.

## Installation

Our code is developed based on PyTorch 1.13.1, CUDA 11.8 and Python 3.9. Lower version pytorch should also work well.

To install, run the following commands:

```
git clone https://github.com/98mxr/GMFSS_Fortuna.git
cd GMFSS_Fortuna
pip install -r requirements.txt
```

## Model Zoo

If you want to validate the results then you need the [GMFSS model](https://drive.google.com/file/d/1BKz8UDAPEt713IVUSZSpzpfz_Fi2Tfd_/view?usp=sharing) or [union model](https://drive.google.com/file/d/1Mvd1GxkWf-DpfE9OPOtqRM9KNk20kLP3/view?usp=sharing)

If you want to train your own model, you can use our [pre-trained model](https://drive.google.com/file/d/1y5Spgidahk12Q0MO-ZlSVLDMRQoj6FJI/view?usp=sharing) to skip the baseline training process

## Run Video Frame Interpolation

- Unzip the downloaded models and place the `train_log` folder in the root directory. Then run one of the following commands.

1. Using gmfss mode

```
python3 inference_video.py --img=demo/ --scale=1.0 --multi=2
```

2. Using union mode

```
python3 inference_video.py --img=demo/ --scale=1.0 --multi=2 --union
```

## Train

- Unzip the pre-trained models and place the `train_log` folder as well as dataset in the root directory. Modifying `model/dataset.py` is necessary to fit other datasets. Run one of the following commands.

1. Train gmfss with gan optimization

```sh
python3 train_pg.py
```

2. Train gmfss_union with gan optimization

```sh
python3 train_upg.py
```

3. Train pre-trained models

```sh
python3 train_nb.py
```

## Acknowledgment

This project is supported by [SVFI](https://steamcommunity.com/app/1692080) [Development Team](https://github.com/Justin62628/Squirrel-RIFE) 
