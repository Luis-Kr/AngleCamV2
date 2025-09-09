# AngleCam V2: Deep learning for leaf angle estimation

AngleCam V2 is a deep learning framework for estimating leaf angle distributions from images. It can track how plants move their leaves throughout the day and night, using nothing more than 
regular photos. Works with everything from smartphone snapshots to research-grade time-lapse cameras across many plant species.

## Installation

### Prerequisites
- [Conda](https://docs.conda.io/en/latest/miniconda.html)

### Quick Setup

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
   python -c "import anglecam_v2; print('Installation successful!')"
   ```

## Data Setup

Download the complete dataset and pre-trained models from [Zenodo](https://zenodo.org/records/17086253) (44.5 GB):

1. **Download and extract the dataset:**
   ```bash
   # Download AngleCamV2_Dataset.zip from Zenodo
   unzip AngleCamV2_Dataset.zip
   cp -r AngleCamV2_Dataset/* data/
   ```

## Quick Start

### Python API
```python
from anglecam.main import AngleCam

# Initialize from config
model = AngleCam.from_checkpoint(CHECKPOINT_PATH)

# Make predictions on a single image
predictions = model.predict("path/to/image.jpg")

# Predict on multiple images
predictions = model.predict_batch(["image1.jpg", "image2.jpg"])
```

### Command Line Interface

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
python -m anglecam.cli.predict image_path=path/to/image.jpg
```

**Predict on multiple images:**
```bash
python -m anglecam.cli.predict image_path=path/to/images/
```

### Essential Configuration Options

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
python -m anglecam.cli.train data.train_csv=my_data.csv                           # Custom training data
python -m anglecam.cli.train data.data_dir=path/to/images                        # Custom image directory
python -m anglecam.cli.retrain data.train_csv=new_species.csv                    # Retrain on new data
```

**Model settings:**
```bash
python -m anglecam.cli.train model.head.dropout=0.3        # Adjust dropout rate
python -m anglecam.cli.train training.lr=1e-5              # Set learning rate
python -m anglecam.cli.predict inference.checkpoint_path=my_model.pth            # Use custom model weights
```

## Training Data Format

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
- `data.data_dir`: Directory containing images

Override any config parameter from command line:
```bash
python -m anglecam.cli.train device=cpu training.epochs=100
```

## Labeling Tool

Create your own training data using the included labeling tool:

```bash
cd scripts/labeling-tool
python app.py
```

The tool provides an interface for manually annotating leaf angles in images.

## Model Architecture

- **Backbone**: DINOv2 ViT-S/14 (384-dimensional features)
- **Head**: Linear → Dropout → GELU → Linear → Softmax
- **Output**: 43-bin probability distribution (0-90° in 2° steps)
- **Training**: Mixed RGB/NIR modality with data augmentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and issues, please:
- Contact: [luis.kremer@geosense.uni-freiburg.de]