# Quantitative texture analysis

This project aims to analyse placental textures on ultrasound images for the early detection of late placental insufficiency in 
2D ultrasound images.

## Usage

There are two scripts. One in feature_extraction folder (feature_extraction.py), the other one in classification folder (compare_pipelines.py). 
For the feature_extraction.py script it is needed a folder parent with the folder of each patient 
containing their DICOM files and a PNG masks (with the same name). It is also needed the database CSV file with the patients information with these attributes: 
- Record ID (the patient identification)
- Image (name of the DICOM file with extension) 
- Label (1: with desease/s, 0: healthy) 

For the compare_pipelines.py it is needed that the textural features are extracted in a CSV file in the /data folder

### Command Examples
 
In feature extraction:  
```python feature_extraction.py --folder folder --output_features output --db_path database_csv --settings settings_1.yaml```   
The `--folder` parameter with the parent folder of the patients folders (with at least one DICOM and mask files each). 
If it is not specified it will be read from ./feature_extraction/config.ini.  
The `--output_features` parameter is the name of the folder created to save the outputs in the ./data folder. 
If it is not specified it will be read from ./feature_extraction/config.ini.  
The `--db_path` parameter with the database path, this parameter can be specified otherwise it will be read from 
./feature_extraction/config.ini.  
The `--settings` parameter with the PyRadiomics settings file to extract features can be specified, otherwise it will be read 
from ./feature_extraction/config.ini.  

In classification:  
```python compare_pipelines.py --input_features input_path```  
The `--input_features` parameter is the path to the CSV file with the extracted features to be used as input of the pipelines. 
If it is not specified it will be read from ./classification/config.ini. 

#### Structure
- ```./notebooks```: Jupyter Notebook with EDA, feature extraction, statistical study and framework classification
- ```./feature_extraction```:
- ```./feature_extraction/feature_extraction.py```: launches the code for extracting features with Pyradiomics 
- ```./feature_extraction/config.ini```: configuration file with the folder paths:``patients_path`` folder parent of the patients 
folders, ``settings_name``: YAML file for PyRadiomics, ``db_path``: file with the Record ID, Image, Label, pixel spacing x, ROI pixels of every image
- ```./classification```: executes a K-fold cross-validation with Scikit-Learn for a combination of models defined in algorithms.py
- ```./classification/results```: CSV with the scores and information of the models for every experiment
- ```./data```: contains the CSV of the extracted texture features extracted from PyRadiomics library and in-house code (see notebook 1.prepare_data.ipynb). 


# Installation and requirements
git clone https://github.com/Lcobbar/quantitative-texture-analysis   
pip install -r requirements

## Requirements
Python v3.10.11   
Pyradiomics v3.0.1

## Acknowledgments
The classification framework is based on [Biorad project](https://github.com/ahmedalbuni/biorad)