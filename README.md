## *ViscoNet*: Bridging and Harmonizing Visual and Textual Conditioning for ControlNet
[Soon Yau Cheong](https://scholar.google.com/citations?user=dRot7GUAAAAJ&hl=en)
[Armin Mustafa](https://scholar.google.com/citations?user=0xOHqkMAAAAJ&hl=en)
[Andrew Gilbert](https://scholar.google.com/citations?user=NNhnVwoAAAAJ&hl=en)


<a href='https://soon-yau.github.io/visconet/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 
<a href='https://arxiv.org/abs/2312.03154'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/3_6Zq3hk86Q)

https://github.com/soon-yau/visconet/assets/19167278/ae58b7ab-fa76-4253-8a10-46656f234b20

### Requirements
A suitable [conda](https://conda.io/) environment named `control` can be created
and activated with:
```
conda env create -f environment.yaml
conda activate control
```
### Files
All model and data files are in [here](https://huggingface.co/soonyau/visconet/tree/main).
Including eval.zip containing all images used in human evaluation.

### Gradio App
[![App](./assets/app.png)](https://youtu.be/3_6Zq3hk86Q)
run ```python gradio_visconet.py```

### Citation
```
@inproceedings{visconet,
        author    = {Cheong, Soon Yau and Mustafa, Armin and Gilbert, Andrew},
        title     = {ViscoNet: Bridging and Harmonizing Visual and Textual Conditioning for ControlNet},
        booktitle = {ECCV Workshop Proceedings}
        month     = {September},
        year      = {2024}}
```
