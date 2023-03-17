# CLIP-Lite
Pytorch Implementation of CLIP-Lite | Accepted at AISTATS 2023 | [paper](https://arxiv.org/abs/2112.07133#)

## Installation
### Environment
1. Install Anaconda or Miniconda distribution based on Python 3.7.x from their downloads site.

4. Clone the repository
```
git clone git@github.com:4m4n5/CLIP-Lite.git
```
    
5. Create a conda environment and install all the dependencies.
```
cd vislang-infomax
conda create -n vlinfo python=3.7 --file=environments.yml
conda activate vlinfo
pip install -r requirements.txt
```

5. Sign into weights and biases for logging
    - Run `wandb login` in the terminal or `import wandb; wandb.login()` in a python interpreter and follow the prompts
    
    
### Datasets
Datasets are assumed to exist in `./data/datasets/` directory (relative to the project root) following the structure specified below. 
1. COCO is used for pretraining. This structure is compatible when using Detectron2 for downstream tasks.
```
./data/datasets/coco/
    annotations/
        captions_{train,val}2017.json
        instances_{train,val}2017.json
    train2017/
        # images in train2017 split
    val2017/
        # images in val2017 split
```

2. PASCAL VOC is used for downstream classification/detection tasks
```
./data/datasets/VOC2007/
    Annotations/
    ImageSets/
        Main/
            trainval.txt
            test.txt
    JPEGImages/
```

3. ImageNet is used for downstream fine-tuning tasks
```
./data/datasets/imagenet/
    train/
        # One directory per category with images in it
    val/
        # One directory per category with images in it
    ILSVRC2012_devkit_t12.tar.gz
```

4. iNaturalist 2018 is used for downstream classification task
```
./data/datasets/inaturalist/
    train_val2018/
    annotations/
        train2018.json
        val2018.json
```

### Pre-process Data
Serialize COCO Captions (`train2017` and `val2017` splits) into LMDB files. These are faster for data reading during pretraining.
```
python scripts/coco_preprocess.py \
    --mode train_sbert \
    --data-root /data/datasets/coco/ \
    --split train \
    --output datasets/serialized/
```

## Training
First edit `factories.py` to include paths to the required datasets. Training parameters are specified by config files (YAML) located at `./configs/done/`.
For every run a new folder will be created in the `checkpoints-dir` directory for logs and checkpoints.
```
python train.py \
    --config configs/sbert/from_scratch/fs_bs1024_ni250k.yaml \
    --num-gpus-per-machine 8 \
    --cpu-workers 0 \
    --checkpoints-dir saves/checkpoints
```
