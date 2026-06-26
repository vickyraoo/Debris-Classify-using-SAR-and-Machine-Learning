# Debris-Classify-using-SAR-and-Machine-Learning
## Overview
This repository contains the methodology, workflow, and results for mapping the debris cover extent of the Drang Drung and Pensilungpa glaciers in the Ladakh Himalaya. By integrating Sentinel-1A Synthetic Aperture Radar (C-band) and Landsat-8/9 optical imagery, this project leverages machine learning to automate and enhance precision glacier monitoring. 

Traditional cryosphere workflows require time-intensive, human-in-the-loop processing. This project introduces a robust, multi-sensor methodology and an autonomous classification framework to streamline satellite analysis chains, translating raw SAR and optical imagery into actionable glacier information.

## Key Features
* **Multi-Sensor Integration:** Fuses Sentinel-1A SAR backscatter (sensitive to surface roughness/moisture) with Landsat-8/9 multispectral reflectance (sensitive to color/composition) to overcome traditional spectral ambiguity.
* **Automated Machine Learning Pipeline:** Utilizes a Random Forest (RF) classification framework to separate clean ice, debris-covered ice, snow, and rock zones based on physical and textural characteristics.
* **Advanced SAR Preprocessing:** Implements a comprehensive ESA SNAP workflow including TOPSAR debursting, radiometric calibration, polarimetric speckle filtering (Improved Lee Sigma), and C2 covariance matrix generation.
* **High-Altitude Precision:** Achieves pixel-level classification with an overall accuracy exceeding 89% in challenging, rugged Himalayan topography.

---

## Study Area
The analysis focuses on two major glaciers in the Zanskar region of Ladakh, India:
1. **Drang Drung Glacier:** A 20-23 km valley glacier originating from Doda Peak (~5,400 m). Features a large accumulation zone of clean ice and a broad ablation zone covered by a thin, patchy debris layer.
2. **Pensilungpa Glacier:** An 8 km glacier characterized by rugged surface features, thick debris layers concentrated along its tongue, and high sensitivity to climatic variations.

---

## Methodology & Tech Stack

### Data Acquisition
* **Active Sensor (SAR):** Sentinel-1A Level-1 SLC data (C-band, VV/VH polarizations) sourced from the Alaska Satellite Facility (ASF DAAC).
* **Passive Sensor (Optical):** Landsat-8/9 multispectral imagery (Bands 2, 6, 7) sourced from USGS Earth Explorer.
* **Topography:** Copernicus 30m Digital Elevation Model (DEM) for geometric terrain correction.

### Data Processing Pipeline
1. **SAR Preprocessing (ESA SNAP):** Splitting by sub-swaths -> Orbit Correction -> Radiometric Calibration -> TOPSAR Debursting -> Sub-swath Merging -> Multilooking -> Polarimetric Speckle Filtering -> Terrain Correction.
2. **Feature Extraction:** Deriving backscatter coefficients and textural parameters (VV, VH, local variance) from the polarimetric matrix.
3. **Training Sample Generation:** Manual creation of Regions of Interest (ROIs) for distinct surface classes using QGIS.
4. **Classification:** Training and deploying a Random Forest model to handle non-linear, multi-source data streams for final pixel classification.

---

## Results & Performance

The Random Forest model effectively isolated the spatial distribution of debris mantles, providing critical insights into surface conditions and melt dynamics. 

### 2020 Classification Metrics

| Glacier | Total Area (km²) | Clean Ice Area | Debris Area | Overall Accuracy | Kappa Coefficient |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Drang Drung** | 80.24 | 58.01 km² (72.3%) | 22.32 km² (27.3%) | 94.2% | 0.90 |
| **Pensilungpa** | 16.89 | 10.89 km² (64.0%) | 6.09 km² (36.0%) | 90.8% | 0.80 |

**Key Findings:**
* The **Drang Drung Glacier** exhibits a lower proportion of debris cover, primarily concentrated in the ablation area and lateral margins, suggesting high activity but ongoing melt at the terminus.
* The **Pensilungpa Glacier** demonstrates an extensive, thicker debris mantle (36% of its surface) relative to its size, indicating stronger rockfall and sediment deposition which acts as an insulator reducing melt rates in specific regions.

---

## Contributors
* **Salah Ud Din**
* **Punit Kumar**
* **Krish Bansal**

*Supervised by Prof. Gulab Singh* *Centre of Studies in Resources Engineering, Indian Institute of Technology Bombay (November 2025)*
