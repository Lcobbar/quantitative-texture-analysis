import numpy as np
import pandas as pd
import csv
import logging
from multiprocessing import cpu_count, Pool
import os
import threading
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk
import yaml
from radiomics import featureextractor
from collections import OrderedDict
from utils import erode_mask, pre_process, lbp_features_extractor


# class based on https://github.com/AIM-Harvard/pyradiomics/blob/master/examples/batchprocessing_parallel.py
class Extractor:
    def __init__(self, input_csv, images_folder, features_output, settings_name):
        # Logger
        self.logger = logging.getLogger('ExtractorLogger')
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler('../data/log.txt', mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        logger_pyr = logging.getLogger('radiomics')
        logger_pyr.addHandler(file_handler)
        logger_pyr.setLevel(logging.INFO)
        self.logger.info('Extractor init')

        # Parallel processing variables
        self.NUM_OF_WORKERS = cpu_count() - 1  # Number of processors to use, keep one processor free for other work
        if self.NUM_OF_WORKERS < 1:  # in case only one processor is available, ensure that it is used
            self.NUM_OF_WORKERS = 1
        self.logger.info('Workers ' + str(self.NUM_OF_WORKERS))

        # Paths
        self.settings_path = os.path.join('./', settings_name)
        self.input_csv = input_csv
        self.images_path = images_folder

        try:
            settings_number = settings_name.split('_')[1].split('.')[0]
        except IndexError:
            settings_number = '1'
        self.output_path = os.path.join('../data/{}/'.format(features_output), "features_{}.csv".format(settings_number))

    def post_process_features(self):
        roi_transforms = {
            'original_firstorder_Energy': lambda Np, feature: feature / Np,
            'original_firstorder_Entropy': lambda Np, feature: feature / np.log(Np),
            'LBP_R_1_P_8_Energy': lambda Np, feature: feature / Np,
            'LBP_R_1_P_8_Entropy': lambda Np, feature: feature / np.log(Np),
            'LBP_R_2_P_16_Energy': lambda Np, feature: feature / Np,
            'LBP_R_2_P_16_Entropy': lambda Np, feature: feature / np.log(Np),
            'LBP_R_3_P_24_Energy': lambda Np, feature: feature / Np,
            'LBP_R_3_P_24_Entropy': lambda Np, feature: feature / np.log(Np),
            'original_glcm_InverseVariance': lambda Np, feature: feature / Np,
            'original_glrlm_RunLengthNonUniformity': lambda Np, feature: feature / Np,
            'original_glrlm_GrayLevelNonUniformity': lambda Np, feature: feature / Np,
            'original_ngtdm_Strength': lambda Np, feature: feature / Np,
            'original_ngtdm_Coarseness': lambda Np, feature: feature / Np
        }

        columns_to_remove = []
        try:
            features = pd.read_csv(self.output_path)
            features_modified = features.copy()
            for column_name, function in roi_transforms.items():
                new_column_name = f"{column_name}_pix_norm"
                features_modified[new_column_name] = features.apply(
                    lambda row: function(row['ROI pixels'], row[column_name]), axis=1)
                columns_to_remove.append(column_name)
            features_modified.drop(columns=columns_to_remove, inplace=True)

            features_modified.to_csv(os.path.splitext(self.output_path)[0] + "_pix_norm.csv", index=False)
        except Exception:
            self.logger.error('Problem reading or writing the final file', exc_info=True)
            exit(-1)

    def run(self, case):

        feature_vector = OrderedDict(case)

        try:
            threading.current_thread().name = case['Record ID'] + '_' + case['Image']

            if os.path.isfile(self.settings_path):
                extractor = featureextractor.RadiomicsFeatureExtractor(self.settings_path)
            else:
                self.logger.error('SETTINGS READ FAILED')
                exit(-1)

            with open(self.settings_path, 'r') as file:
                data_yaml = yaml.safe_load(file)

            feature_vector = OrderedDict(case)

            patient_folder = os.path.join(self.images_path, str(case['Record ID']))
            dicom_path = os.path.join(patient_folder, case['Image'])
            mask_path = os.path.join(patient_folder, os.path.splitext(os.path.basename(dicom_path))[0] + '_mask.png')

            image_3d = sitk.ReadImage(dicom_path, sitk.sitkUInt8)
            mask = plt.imread(mask_path).astype(np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask_eroded = erode_mask(mask)
            mask_2d = sitk.GetImageFromArray(mask_eroded)

            mask_3d = sitk.JoinSeries(mask_2d)
            mask_3d.SetSpacing(image_3d.GetSpacing())
            feature_vector.update(extractor.execute(image_3d, mask_3d))

            _, _image, _mask = pre_process(image_3d, mask_3d, data_yaml, normalize=False)
            feature_vector.update(lbp_features_extractor(_image, _mask))
            self.logger.info('Patient %s image %s', case['Record ID'], case['Image'])

        except Exception:
            self.logger.error('Feature extraction failed', exc_info=True)

        return feature_vector

    def extract(self):

        sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

        try:
            with open(self.input_csv, 'r') as inFile:
                cr = csv.DictReader(inFile, lineterminator='\n')
                cases = []
                for row_idx, row in enumerate(cr, start=1):
                    cases.append(row)
        except Exception:
            self.logger.error('CSV READ FAILED', exc_info=True)
            exit(-1)

        self.logger.info('Loaded %d jobs', len(cases))

        self.logger.info('Starting parallel pool with %d workers out of %d CPUs', self.NUM_OF_WORKERS, cpu_count())

        pool = Pool(self.NUM_OF_WORKERS)
        results = pool.map(self.run, cases)

        try:
            with open(self.output_path, mode='w') as outputFile:
                writer = csv.DictWriter(outputFile,
                                        fieldnames=list(results[0].keys()),
                                        restval='',
                                        extrasaction='raise',
                                        lineterminator='\n')
                writer.writeheader()
                writer.writerows(results)

        except Exception:
            self.logger.error('Error storing results', exc_info=True)

        self.post_process_features()
