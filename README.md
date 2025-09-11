# AngleCam V2: Deep learning for leaf angle estimation

AngleCam V2 is a deep learning framework for estimating leaf angle distributions from images. It can track how plants move their leaves throughout the day and night, using nothing more than 
regular photos. Works with everything from smartphone snapshots to research-grade time-lapse cameras across many plant species.

![](https://github.com/Luis-Kr/AngleCamV2/blob/main/animation/maranta-leuconeura-timelapse.gif)

## Installation

### Prerequisites
<a href="https://docs.conda.io/en/latest/miniconda.html" target="_blank">Conda</a>

### Quick setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Luis-Kr/AngleCamV2.git
   cd AngleCamV2
   ```

2. **Create and activate the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate anglecam_v2
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```

4. **Verify installation:**
   ```bash
   python -c "import anglecam; print('Installation successful!')"
   ```


## Data and model setup

### For complete new training
Download the complete dataset including the labels from <a href="https://zenodo.org/records/17086253" target="_blank">Zenodo</a> (45 GB):

```bash
unzip AngleCamV2_Dataset.zip

# After downloading from Zenodo, move it to:
mv -v AngleCamV2_Dataset/* data/
```

### For prediction and re-training only
Download only the pre-trained model from <a href="https://doi.org/10.5281/zenodo.17101166" target="_blank">Zenodo</a> (103 MB):

```bash
# Create the checkpoint directory if it doesn't exist
mkdir -p data/checkpoint

# Download and place the pre-trained model
# After downloading from Zenodo, move it to:
mv AngleCamV2.pth data/checkpoint/
```


## Quick start

### Python API
```python
from omegaconf import OmegaConf
from anglecam.main import AngleCam

# Load config (path relative to repository root)
config = OmegaConf.load("anglecam/config/main.yaml")

# Load model
model = AngleCam.from_checkpoint("data/checkpoint/AngleCamV2.pth", config)

# Make predictions
results = model.predict("path/to/images/")
```

### Command line interface

**Train a new model from scratch:**
```bash
python -m anglecam.cli.train
```
Trains a model on all available training images. Use this when starting with a new dataset or training architecture.

**Retrain the pre-trained AngleCamV2 model:**
```bash
python -m anglecam.cli.retrain data.train_csv=path/to/new_training.csv
```
Loads the pre-trained AngleCamV2 model and retrains it on new labeled images. Use this to adapt the model to your specific species or conditions while preserving learned features.

**Make predictions:**
```bash
python -m anglecam.cli.predict inference.image_path=path/to/image.jpg
```

**Predict on multiple images:**
```bash
python -m anglecam.cli.predict inference.image_path=path/to/images/
```

### Essential configuration options

Override any config parameter from command line:

**Device and performance:**
```bash
python -m anglecam.cli.train device=cpu                    # Use CPU instead of GPU
python -m anglecam.cli.train device=cuda:1                 # Use specific GPU
python -m anglecam.cli.train training.batch_size=16        # Adjust batch size
python -m anglecam.cli.train training.epochs=100           # Set number of epochs
```

**Data paths:**
```bash
python -m anglecam.cli.train data.train_csv=my_data.csv         # Custom training data
python -m anglecam.cli.train data.data_dir=path/to/images       # Custom image directory
python -m anglecam.cli.retrain data.train_csv=new_species.csv   # Retrain on new data
```

**Model settings:**
```bash
python -m anglecam.cli.train model.head.dropout=0.3                           # Adjust dropout rate
python -m anglecam.cli.train training.optimizer.lr=1e-5                       # Set learning rate
python -m anglecam.cli.predict inference.pretrained_model_path=my_model.pth   # Use custom model weights
```

## Training data format

Your training CSV should contain image filenames and species information:

```csv
filename,species
image_001.png,Acer pseudoplatanus
image_002.png,Tilia platyphyllos
image_003.png,Quercus robur
```

Images should be placed in the directory specified by `data.data_dir` in the config (default: `data/01_Training_Validation_Data/image_data`). 

## Configuration

The model uses Hydra for configuration management. Key settings in `anglecam/config/main.yaml`:

- `device`: cuda:0, cpu, or auto
- `data.train_csv`: Path to training CSV
- `data.val_csv`: Path to validation CSV  
- `data.data_dir`: Directory containing images and simulation files (labels)

Override any config parameter from command line:
```bash
python -m anglecam.cli.train device=cpu training.epochs=100
```

## Labeling tool

Create your own training data using the included labeling tool:

```bash
cd scripts/labeling-tool
python app.py
```

The tool provides an interface for manually annotating leaf angles in images. Place the simulation files (`_sim.csv`; labels) and their corresponding images in this folder: `data/01_Training_Validation_Data/image_data`. Or specify a different location using the command line interface.

## Model architecture

- **Backbone**: DINOv2 ViT-S/14 (384-dimensional features)
- **Head**: Dropout → Linear → Dropout → GELU → Linear → Softmax
- **Output**: 43-bin probability distribution (0-90° in 2° steps)
