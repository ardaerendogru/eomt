# Your ViT is Secretly an Image Segmentation Model  
**CVPR 2025 · Highlight Paper**

**[Tommie Kerssies](https://tommiekerssies.com)¹, [Niccolò Cavagnero](https://scholar.google.com/citations?user=Pr4XHRAAAAAJ)²\*, [Alexander Hermans](https://scholar.google.de/citations?user=V0iMeYsAAAAJ)³, [Narges Norouzi](https://scholar.google.com/citations?user=q7sm490AAAAJ)¹, [Giuseppe Averta](https://www.giuseppeaverta.me/)², [Bastian Leibe](https://scholar.google.com/citations?user=ZcULDB0AAAAJ)³, [Gijs Dubbelman](https://scholar.google.nl/citations?user=wy57br8AAAAJ)¹, [Daan de Geus](https://ddegeus.github.io)¹,³**

¹ Eindhoven University of Technology  
² Polytechnic of Turin  
³ RWTH Aachen University  
\* Work done while visiting RWTH Aachen University

📄 **Paper:** [arXiv:2503.19108](https://arxiv.org/abs/2503.19108)

## Overview

We present the **Encoder-only Mask Transformer (EoMT)**, a minimalist image segmentation model that repurposes a plain Vision Transformer (ViT) to jointly encode image patches and segmentation queries as tokens. No adapters. No decoders. Just the ViT.

Leveraging large-scale pre-trained ViTs, EoMT achieves accuracy similar to state-of-the-art methods that rely on complex, task-specific components. At the same time, it is significantly faster thanks to its simplicity, for example up to 4× faster with ViT-L.  

Turns out, *your ViT is secretly an image segmentation model*. EoMT shows that architectural complexity isn’t necessary, plain Transformer power is all you need.
Turns out, *your ViT is secretly an image segmentation model*. EoMT demonstrates that architectural complexity isn't necessary, plain Transformer power is all you need.

---

## Installation

If you don't have Conda installed, install Miniconda and restart your shell:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then create the environment, activate it, and install the dependencies:

```bash
conda create -n eomt python==3.13.2
conda activate eomt
python3 -m pip install -r requirements.txt
```

[Weights & Biases](https://wandb.ai/) (wandb) is used for experiment logging and visualization. To enable wandb, log in to your account:

```bash
wandb login
```

## Data preparation

Download the datasets below depending on which datasets you plan to use.  
You do **not** need to unzip any of the downloaded files.  
Simply place them in a directory of your choice and provide that path via the `--data.path` argument.  
The code will read the `.zip` files directly.

**COCO**
```bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
```

**ADE20K**
```bash
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
wget http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
tar -xf annotations_instance.tar
zip -r -0 annotations_instance.zip annotations_instance/
rm -rf annotations_instance.tar
rm -rf annotations_instance
```

**Cityscapes**
```bash
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=<your_username>&password=<your_password>&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
```

🔧 Replace `<your_username>` and `<your_password>` with your actual [Cityscapes](https://www.cityscapes-dataset.com/) login credentials.  

## Usage

### Training

To train EoMT from scratch, run:

```bash
python3 main.py fit \
  -c configs/coco/panoptic/eomt_large_640.yaml \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset
```

This command trains the `EoMT-L` model with a 640×640 input size on COCO panoptic segmentation using 4 GPUs. Each GPU processes a batch of 4 images, for a total batch size of 16.  

✅ Make sure the total batch size is `devices × batch_size = 16`  
🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.

> This configuration takes ~6 hours on 4×NVIDIA H100 GPUs, each using ~26GB VRAM.

To fine-tune a pre-trained EoMT model, add:

```bash
  --model.ckpt_path /path/to/pytorch_model.bin \
  --model.load_ckpt_class_head False
```

🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to fine-tune.  
> `--model.load_ckpt_class_head False` skips loading the classification head when fine-tuning on a dataset with different classes. 

### Evaluating

To evaluate a pre-trained EoMT model, run:

```bash
python3 main.py validate \
  -c configs/coco/panoptic/eomt_large_640.yaml \
  --model.network.masked_attn_enabled False \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset \
  --model.ckpt_path /path/to/pytorch_model.bin
```

This command evaluates the same `EoMT-L` model using 4 GPUs with a batch size of 4 per GPU.

🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.  
🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to evaluate.

A [notebook](inference.ipynb) is available for quick inference and visualization with auto-downloaded pre-trained models.

## Model Zoo

> All FPS values were measured on an NVIDIA H100 GPU.

### Panoptic Segmentation

#### COCO

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">PQ</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 640x640 -->
<tr><td align="left"><a href="configs/coco_panoptic_eomt_large_640.yaml">EoMT-L</a></td>
<td align="center">640x640</td>
<td align="center">128</td>
<td align="center">56.0</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_large_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-L 1280x1280 -->
<tr><td align="left"><a href="configs/coco_panoptic_eomt_large_1280.yaml">EoMT-L</a></td>
<td align="center">1280x1280</td>
<td align="center">30</td>
<td align="center">58.3</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_large_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 640x640 -->
<tr><td align="left"><a href="configs/coco_panoptic_eomt_giant_640.yaml">EoMT-g</a></td>
<td align="center">640x640</td>
<td align="center">55</td>
<td align="center">57.0</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_giant_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 1280x1280 -->
<tr><td align="left"><a href="configs/coco_panoptic_eomt_giant_1280.yaml">EoMT-g</a></td>
<td align="center">1280x1280</td>
<td align="center">12</td>
<td align="center">59.2</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_giant_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

#### ADE20K

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">PQ</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 640x640 -->
<tr><td align="left"><a href="configs/ade20k_panoptic_eomt_large_640.yaml">EoMT-L</a></td>
<td align="center">640x640</td>
<td align="center">128</td>
<td align="center">50.6<sup>C</sup></td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_large_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-L 1280x1280 -->
<tr><td align="left"><a href="configs/ade20k_panoptic_eomt_large_1280.yaml">EoMT-L</a></td>
<td align="center">1280x1280</td>
<td align="center">30</td>
<td align="center">51.7<sup>C</sup></td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_large_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 640x640 -->
<tr><td align="left"><a href="configs/ade20k_panoptic_eomt_giant_640.yaml">EoMT-g</a></td>
<td align="center">640x640</td>
<td align="center">55</td>
<td align="center">51.3<sup>C</sup></td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_giant_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 1280x1280 -->
<tr><td align="left"><a href="configs/ade20k_panoptic_eomt_giant_1280.yaml">EoMT-g</a></td>
<td align="center">1280x1280</td>
<td align="center">12</td>
<td align="center">52.8<sup>C</sup></td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_giant_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

*<sub><sup>C</sup> models pre-trained on COCO panoptic segmentation. See above for how to load a checkpoint.</sub>*

### Semantic Segmentation

#### Cityscapes

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 1024x1024 -->
<tr><td align="left"><a href="configs/cityscapes_semantic_eomt_large_1024.yaml">EoMT-L</a></td>
<td align="center">1024x1024</td>
<td align="center">25</td>
<td align="center">84.2</td>
<td align="center"><a href="https://huggingface.co/tue-mps/cityscapes_semantic_eomt_large_1024/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

#### ADE20K

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 512x512 -->
<tr><td align="left"><a href="configs/ade20k_semantic_eomt_large_512.yaml">EoMT-L</a></td>
<td align="center">512x512</td>
<td align="center">92</td>
<td align="center">58.4</td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_semantic_eomt_large_512/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

### Instance Segmentation

#### COCO

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">mAP</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 640x640 -->
<tr><td align="left"><a href="configs/coco_instance_eomt_large_640.yaml">EoMT-L</a></td>
<td align="center">640x640</td>
<td align="center">128</td>
<td align="center">45.2*</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_instance_eomt_large_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-L 1280x1280 -->
<tr><td align="left"><a href="configs/coco_instance_eomt_large_1280.yaml">EoMT-L</a></td>
<td align="center">1280x1280</td>
<td align="center">30</td>
<td align="center">48.8*</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_instance_eomt_large_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

*<sub>\* mAP reported using pycocotools; TorchMetrics (used by default) yields ~0.7 lower.</sub>*

## Citation
If you find this work useful in your research, please cite it using the BibTeX entry below:

```BibTeX
@inproceedings{kerssies2025eomt,
  author    = {Kerssies, Tommie and Cavagnero, Niccolò and Hermans, Alexander and Norouzi, Narges and Averta, Giuseppe and Leibe, Bastian and Dubbelman, Gijs and de Geus, Daan},
  title     = {Your ViT is Secretly an Image Segmentation Model},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
}
```

## Acknowledgements

This project builds upon code from the following libraries and repositories:

- [Hugging Face Transformers](https://github.com/huggingface/transformers) (Apache-2.0 License)  
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) (Apache-2.0 License)  
- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) (Apache-2.0 License)  
- [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) (Apache-2.0 License)  
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) (Apache-2.0 License)
- [Detectron2](https://github.com/facebookresearch/detectron2) (Apache-2.0 License)


# Model Comparison Visualizer

A Gradio application for comparing image segmentation model outputs across different models, with a focus on ADE20K dataset predictions.

## Features

- **Class-based Image Selection**: Filter and find images containing specific classes
- **Multi-Model Comparison**: View predictions from different models side by side
- **Detailed Metrics**: Compare PQ (Panoptic Quality), SQ (Segmentation Quality), and RQ (Recognition Quality) metrics
- **Per-Class Analysis**: View per-class performance metrics for each model
- **Model Difference Visualization**: Compare the differences between models with color-coded metrics
- **Interactive Zooming**: Easily inspect and zoom in on images for detailed analysis

## Getting Started

### Prerequisites

- Python 3.x
- Required packages (install with `pip install -r requirements.txt`):
  - pandas
  - matplotlib
  - numpy
  - pillow
  - gradio
  - seaborn

### Running the App

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python gradio_app.py
   ```
4. Open your browser at the URL displayed in the terminal (typically `http://127.0.0.1:7860`)

## Usage

1. **Select Classes**: Choose one or more classes from the dropdown menu
2. **Select Models**: Select models you want to compare
3. **Find Images**: Click the "Find Images & Visualize" button to retrieve images containing the selected classes
4. **Navigate Images**: Use the Previous/Next buttons to browse through found images
5. **Jump to Specific Images**: Enter an image index in the text field and click "Go"
6. **View Metrics**: Check the Metrics tab for detailed comparison metrics
7. **Zoom Images**: Use the Zoom Individual Images tab to examine images in detail
   - Hover over images to enlarge slightly
   - Click and hold to zoom in more
   - Click gallery images to view in full screen mode

## Data Structure

The application expects the following data structure:
- Images in `/home/arda/thesis/eomt/output/` with naming convention:
  - Input images: `val_{idx}_input.png`
  - Target segmentations: `val_{idx}_target.png`
  - Model predictions: `val_{idx}_pred_{model_name}.png`
- Class names in `/home/arda/thesis/eomt/output/class_names.txt`
- Metrics and instance counts in `/home/arda/thesis/eomt/output/instance_counts.json`

## Customization

You can customize the app by modifying the file paths in the code to match your directory structure, or by adding additional visualizations as needed.