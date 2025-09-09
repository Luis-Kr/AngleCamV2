# TLS Leaf Angle Analyzer

A tool for extracting leaf inclination angle distributions from Terrestrial Laser Scanning (TLS) point clouds.

## Overview

This analyzer processes LAS point cloud files to compute leaf inclination angle distributions (LIADs) that can be directly compared with AngleCam V2 predictions. The tool implements the methodology described in Section 2.3.2 "TLS data processing and AngleCam comparison" of the AngleCam V2 paper.

## Workflow

The tool extracts leaf inclination angles by:

1. **Normal Vector Calculation**: Uses PCA on local neighborhoods (1 cm radius) to estimate surface normals
2. **Orientation Correction**: Adjusts normals to consistently face the scanner 
3. **Angle Computation**: Calculates angles between surface normals and the zenith vector (0°=horizontal, 90°=vertical)
4. **Subsampling**: Subsamples the point cloud (1 cm voxels) to balance density differences and mitigate inflated vertical angles (vertical leaves intercept more laser beams)
5. **Distribution Analysis**: Generates angle histograms in 2° bins matching AngleCam output format