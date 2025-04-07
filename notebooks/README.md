# High Resolution Canopy Height Maps Notebooks

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

## Setup

Install Conda (https://docs.conda.io/projects/conda/en/latest/index.html#)

## Example of successful environment creation for notebooks
```
conda create chm_demo
conda activate chm_demo
conda install torchvision
conda install pytorch==2.0.1
conda install conda-forge::pytorch-lightning
conda install torchmetrics
conda install geopandas jupyter rasterio boto3
conda install fiona=1.9
conda install scikit-image

jupyter notebook
```

Alternatively, create conda env from yaml file:
```
conda env create -f chm_demo.yml
```