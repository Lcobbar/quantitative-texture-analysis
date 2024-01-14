import argparse
import configparser

from extractor import Extractor


def _read_config(section, key):
    config = configparser.ConfigParser()
    config.read('config.ini')
    if section in config and key in config[section]:
        return config[section][key]
    else:
        raise ValueError(f"Key '{key}' not found in section '{section}' in config.ini")


def extract(images_folder, db_path, features_output, settings_name):
    print(f">>Extracting features with patients in {images_folder}.")
    extractor = Extractor(db_path, images_folder, features_output, settings_name)
    extractor.extract()
    print(f">>Extracting")


def main(folder=None, output_features=None,  db_path=None, settings_name=None):
    images_folder = folder or _read_config('PATH', 'patients_path')
    db_path = db_path or _read_config('PATH', 'db_path')
    settings_name = settings_name or _read_config('PATH', 'settings_name')
    output_features = output_features or _read_config('PATH', 'output_features')
    extract(images_folder, db_path, output_features, settings_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Texture feature extractor')
    parser.add_argument('--folder', type=str, help='Path to the training folder.')
    parser.add_argument('--output_features', type=str, help='Folder for features in /data.')
    parser.add_argument('--db_path', type=str, help='Path to the CSV database for training.')
    parser.add_argument('--settings', type=str, help='Name of the pyradiomics settings file for training.')

    args = parser.parse_args()
    main(args.folder, args.output_features, args.db_path, args.settings)
