import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Materialize dataset files.')

    # Add arguments
    parser.add_argument('--output_dir', type=str, help='output directory where dataset files will be saved.', default=os.getcwd())
    parser.add_argument('--name', type=str, help='name of the dataset', default='iris')

    # Parse arguments
    args = parser.parse_args()

    iris = datasets.load_iris(as_frame=True)

    # Prepare features
    features = iris.data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

    # Prepare labels
    labels_df = iris.target.to_frame(name='label')

    # Write to disk
    features_scaled_df.to_csv(Path(args.output_dir) / f'{args.name}.features.csv', index_label='id')
    labels_df.to_csv(Path(args.output_dir) / f'{args.name}.labels.csv', index_label='id')


if __name__ == "__main__":
    main()
