import argparse
import os
from pathlib import Path

from sklearn import datasets


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Materialize dataset files.')

    # Add arguments
    parser.add_argument('--output_dir', type=str, help='output directory where dataset files will be saved.', default=os.getcwd())
    parser.add_argument('--name', type=str, help='name of the dataset', default='iris')

    # Parse arguments
    args = parser.parse_args()

    iris = datasets.load_iris(as_frame=True)

    # Write to disk
    iris.data.to_csv(Path(args.output_dir) / f'{args.name}.features.csv', index_label='id')
    iris.target.to_csv(Path(args.output_dir) / f'{args.name}.labels.csv', index_label='id')


if __name__ == "__main__":
    main()
