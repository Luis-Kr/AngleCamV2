# AngleCam V2: Deep Learning for Leaf Angle Estimation

AngleCam V2 is a comprehensive deep learning framework for estimating leaf angles from camera images.

## Quick Start

```python
from anglecam.main import AngleCam

# Initialize from config
model = AngleCam.from_config("main.yaml")

# Train model
results = model.train("path/to/training.csv")

# Make predictions
predictions = model.predict("path/to/image.png")
```

## Data Format

Training data expected in CSV format:
```csv
filename,species
image_001.png,Acer pseudoplatanus
image_002.png,Tilia platyphyllos
```

## Model Architecture

- **Backbone**: DINOv2 ViT-S/14 (384-dim features)
- **Head**: Linear → Dropout → GELU → Linear → Softmax
- **Output**: 43-bin probability distribution (0-90° in 2° steps)