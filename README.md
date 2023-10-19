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

[[`Paper`](https://arxiv.org/abs/2304.07213)] [[`Blog`](https://research.facebook.com/blog/2023/4/every-tree-counts-large-scale-mapping-of-canopy-height-at-the-resolution-of-individual-trees/)] [[`BibTeX`](#citing-HighResCanopyHeight)]

https://wri-datalab.earthengine.app/view/submeter-canopyheight 

PyTorch implementation and pretrained models for High resolution Canopy Height Prediction. For details, see the paper: 
**[Very high resolution canopy height maps from RGB imagery using self-supervised  vision transformer and convolutional decoder trained on Aerial Lidar](https://arxiv.org/abs/2304.07213)**.

## Pretrained models

## Requirements

pytorch
pytorch lightning  
pandas

## Data

## Evaluation

```
 python inference.py --checkpoint mydir/model.pt --name new_dir_name
```

## License

HighResCanopyHeight code and model weights are released under the Apache License 2.0. See [LICENSE](LICENSE) for additional details.

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Citing HighResCanopyHeight

If you find this repository useful, please consider giving a star :star: and citation :t-rex::

```
@misc{tolan2023submeter,
      title={Sub-meter resolution canopy height maps using self-supervised learning and a vision transformer trained on Aerial and GEDI Lidar}, 
      author={Jamie Tolan and Hung-I Yang and Ben Nosarzewski and Guillaume Couairon and Huy Vo and John Brandt and Justine Spore and Sayantan Majumdar and Daniel Haziza and Janaki Vamaraju and Theo Moutakanni and Piotr Bojanowski and Tracy Johns and Brian White and Tobias Tiecke and Camille Couprie},
      year={2023},
      eprint={2304.07213},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

