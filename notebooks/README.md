# High Resolution Canopy Height Maps Notebooks

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

## Setup

Install Conda (https://docs.conda.io/projects/conda/en/latest/index.html#)

## Example of successful environment creation for notebooks
```
conda create -n chm_demo python=3.9
conda activate chm_demo
conda install pytorch==2.0.1 -c pytorch
conda install torchvision -c pytorch
conda install conda-forge::pytorch-lightning==1.7
conda install torchmetrics==0.11.4
conda install geopandas jupyter rasterio boto3 scikit-image

jupyter notebook
```

Alternatively, create conda env from yaml file:
```
conda env create -f chm_demo.yml
```