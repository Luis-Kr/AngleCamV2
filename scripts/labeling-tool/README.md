# Leaf Angle Labeling Tool

A tool for manually labeling leaf inclination angles in images and generating simulation files.

## Overview

This tool allows you to load images, mark points on leaves, and record their inclination angles. When you save the data, it automatically generates beta distribution simulations that can be used for retraining AngleCam or performance evaluation on newly labeled images. 

For each image, **20 leaf measurements should be completed**. For each leaf, determine the following two parameters:

1. **Average leaf angle** of the leaf surface (relative to the entire leaf surface), where 0° = horizontal and 90° = vertical
2. **Degree of rolling** (how much the leaf is curled—a stress symptom), where 0 = completely flat leaf, 10 = leaf rolled like a cylinder


## How to Use

### Starting the Tool

Run the script:
```bash
conda activate anglecam_v2
python app.py
```

### Working with Images

1. **Load an image**: Click "New Project" to create a project folder, then "Add Image" to select your image
2. **Navigate**: Use "Previous" and "Next" buttons to move between images in a project
3. **Zoom**: Use "Zoom In", "Zoom Out", or "Fit to Window" for better viewing
4. **Mouse wheel**: Scroll to zoom in/out at cursor position

### Labeling Points

1. **Click on image**: Click anywhere on a leaf to create a point
2. **Set values**: Double-click on "angle" or "rolling" columns in the table to edit values
3. **Delete points**: Select a row and click "Delete Selected Leaf"
4. **View points**: Click on a table row to highlight the point on the image

### Grid Reference

The tool displays 20 red crosses (4x5 grid) on the image as reference points for consistent labeling.

### Saving Data

Click "Save CSV" to save your work. This creates three files in a "labels" folder:

1. **[name].csv** - Your labeled data (ID, image coordinates, angle, rolling)
2. **[name]_sim.csv** - Simulations (reference + 50 variations)
3. **[name]_simulation_plot.png** - Visualization with distribution curves

### Distribution Analysis

Click "Calculate Distribution" to view the angle distribution plot in the tool. The plot shows:
- Histogram of your angle data
- Fitted beta distribution curve
- Mean angle calculation

## File Structure

```
your_project/
├── images/
│   └── your_image.jpg
├── labels/
│   ├── your_image.csv
│   ├── your_image_sim.csv
│   └── your_image_simulation_plot.png
└── project.json
```

## Data Format

### CSV Output (labels/[name].csv)
- **id**: Point identifier
- **x, Y**: Image coordinates
- **angle**: Leaf inclination angle (0-90 degrees)
- **rolling**: Additional parameter of how much the leaf is curled (0=flat; 10=strongly curled)

### Simulation Output (labels/[name]_sim.csv)
- First row: Reference distribution (fitted to the labeled data)
- Rows 2-51: 50 simulation variants with parameter uncertainty (used as augmentation during the training phase)
- Distributions are normalized to sum to 1.0

## Workflow Example

1. Create new project in your data folder
2. Add plant images to the project
3. For each image:
   - Click on at least 20 leaf points (if possible) with regular spacing across the image
   - Edit angle and rolling values in the table
   - Save when finished
4. Use the generated simulation files for retraining AngleCam or performance evaluation on own data. For training, move the simulation files (`_sim.csv`; labels) and the corresponding images to this directory: `data/01_Training_Validation_Data/image_data`. Or specify a different location using the command line interface.

## Output Interpretation

The simulation files contain probability distributions that represent uncertainty in your measurements. The main fitted curve shows the best estimate, while the 50 simulation curves show how the distribution might vary given labeling uncertainty.
