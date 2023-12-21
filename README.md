# High Resolution Canopy Height Maps

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

Jamie Tolan,
Hung-I Yang, 
Benjamin Nosarzewski,
Guillaume Couairon, 
Huy V. Vo, 
John Brandt, 
Justine Spore, 
Sayantan Majumdar, 
Daniel Haziza, 
Janaki Vamaraju, 
Théo Moutakanni, 
Piotr Bojanowski, 
Tracy Johns, 
Brian White, 
Tobias Tiecke, 
Camille Couprie

[[`Paper`](https://doi.org/10.1016/j.rse.2023.113888)][[`ArxiV [same content]`](https://arxiv.org/abs/2304.07213)] [[`Blog`](https://research.facebook.com/blog/2023/4/every-tree-counts-large-scale-mapping-of-canopy-height-at-the-resolution-of-individual-trees/)] [[`BibTeX`](#citing-HighResCanopyHeight)]



PyTorch implementation and pretrained models for High resolution Canopy Height Prediction inference. For details, see the paper: 
**[Very high resolution canopy height maps from RGB imagery using self-supervised  vision transformer and convolutional decoder trained on Aerial Lidar](https://arxiv.org/abs/2304.07213)**.

In collaboration with the Physical Modeling and Sustainability teams at Meta, and the World Resource Institute, we applied [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) to the canopy height map (CHM) prediction problem. We used this technique to pretrain a backbone on about 18 millions satellite images around the globe. We then trained a CHM predictor on a modest sized training dataset covering a few thousand square kilometers of forests in the United States, with this backbone as feature extractor. 
We demonstrate in our paper quantitatively and qualitatively the advantages of large-scale self-supervised learning, the versatility of obtained representations allowing generalization to different geographic regions and input imagery.

The maps obtained with this model are available at https://wri-datalab.earthengine.app/view/submeter-canopyheight. 
  

![alt text](https://github.com/facebookresearch/HighResCanopyHeight/blob/main/fig_0_1_2.png)

## Requirements

pytorch,
pytorch lightning,
pandas

Example of successful environment creation for inference

```
conda create -n hrch python=3.9 -y
conda activate hrch
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pytorch_lightning==1.7 
pip install pandas
pip install matplotlib
pip install torchmetrics==0.11.4
```


## Data and pretrained models

You can download the data and saved checkpoints from 
```
s3://dataforgood-fb-data/forests/v1/models/
```

### Data

Although our method is designed to work from satellite images, it can also estimate canopy height from aerial images.

We share aerial images for the Neon test set we created for the paper in data.zip. 

To automate the color balancing without the need of Maxar images, we trained a network from aerial images (Neon train) to predict the 95th and 5th percentiles of the corresponding maxar images : saved_checkpoints/aerial_normalization_quantiles_predictor.ckpt

### SSL Pretrained models

In the saved_checkpoints directory, there are:

SSLhuge_satellite.pth (2.9G): encoder trained on satellite images, decoder trained on satellite images. Use this model for inference on GPUs. Best results using RGB satellite images in input. 

compressed_SSLhuge.pth (749M): SSLhuge_satellite.pth quantized. Model used in the evaluations of the paper.

compressed_SSLhuge_aerial.pth (749M): encoder trained on satellite images, decoder trained on aerial images.

compressed_SSLlarge.pth (400M): ablation using a large model.

## Evaluation

```
 python inference.py --checkpoint saved_checkpoints/SSLhuge_satellite.pth 
```
```
mae 3.15
r2_block 0.51
Bias: -1.60
```

Here are the performance on aerial images to expect with the different models released. Please note that the 3 first models in this table are trained exclusively on satellite data and are evaluated here in an out of domain context. 

| | SSL large | SSL huge | compressed SSL huge | SSL aerial|
| --- | ---| --- | --- | ---| 
| MAE| 3.31  | 3.15  | 3.08 | 2.5 |
| R2 block | 0.37 | 0.51 | 0.54 | 0.7 |
| Bias | -1.4| -1.6 | -1.6 | -2.1 |

## Notes

We do not include the GEDI correction step in this code release. 

The folder "models" contains code borrowed from the Dinov2 team, we thank all contributors.

The inference using compressed models has not been tested using GPUs (CPU only).

The backbone weights are the same for all SSL models. The backbone has been trained on  images filtered to contain mainly vegetation. 

## License

HighResCanopyHeight code and model weights are released under the Apache License 2.0. See [LICENSE](LICENSE) for additional details.

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Citing HighResCanopyHeight

If you find this repository useful, please consider giving a star :star: and citation :t-rex::

```
@article{tolan2024very,
  title={Very high resolution canopy height maps from RGB imagery using self-supervised vision transformer and convolutional decoder trained on aerial lidar},
  author={Tolan, Jamie and Yang, Hung-I and Nosarzewski, Benjamin and Couairon, Guillaume and Vo, Huy V and Brandt, John and Spore, Justine and Majumdar, Sayantan and Haziza, Daniel and Vamaraju, Janaki and others},
  journal={Remote Sensing of Environment},
  volume={300},
  pages={113888},
  year={2024}
}
```

