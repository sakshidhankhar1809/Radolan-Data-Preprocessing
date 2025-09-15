# Data preprocessing for RADOLAN Images
This repository provides preprocessing tools for working with DWD’s radar precipitation images. It focuses on transforming the raw radar data into structured, region-specific datasets that can be directly used for analysis or modeling. This repository is a part of the preprocessing module used in the paper: https://doi.org/10.1016/j.ejrh.2025.102571.

## Key Features
### Geographical coordinate conversion
Converts latitude and longitude values into the radar image’s coordinate system, ensuring precise mapping between geographic locations and pixel indices.

### Pixel index calculation
Determines the exact pixel positions (with a resolution of 1 km × 1 km per pixel) that correspond to given geographic coordinates.

### Regional cropping
Extracts precipitation data for a specified region by cropping radar images according to the target area’s bounding box or shape. This makes it possible to isolate rainfall information for cities, watersheds, or custom-defined areas of interest.

## Output
The preprocessing pipeline produces CSV files where each entry corresponds to precipitation intensity values represented as pixels. The CSVs preserve the spatial structure of the radar image subset, effectively turning raw radar data into a grid of precipitation values for the selected region.

## Use Case
This workflow is particularly useful for:
1. Hydrological modeling
2. Weather and climate research
3. Flood risk assessment
4. Training machine learning models on localized precipitation data

## Instalation
To install and setup:
1. Clone the repository:
```bash
git git@github.com:sakshidhankhar1809/Radolan-Data-Preprocessing.git
cd Radolan-Data-Preprocessing
```
2. create a new virtual environment and install the dependencies:
```bash
conda create --name <env_name> --file requirements.txt
conda activate <env_name>
```
## Usage
Using the `pp` class, the Radolan image data can be acquired by instantiating an target location object. To do so, The coordination of desired city should be given as a numpy.ndarray anlong with the radolan data resolution. Here is an example:
```python
from utils import pp

latlon_hildesheim = np.array([9.9580, 52.1548])
h = pp(1100, 900, latlon_hildesheim, 50, "Hildesheim")
``` 
by using `mask_region()` method, the indices for the target location can be achieved. Finally, you can use the method `plot_mask()` to get the visualization for the appropraite inpute `data`.
```python
mask_idx = h.mask_region()
h.plot_mask(data)
```

![alt text](hildesheim_mask.png)

