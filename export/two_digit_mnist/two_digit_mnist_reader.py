import os
import csv
import sys
import argparse

import stable_nalu

# Parse arguments
parser = argparse.ArgumentParser(description='Export results from simple function task')
parser.add_argument('--tensorboard-dir',
                    action='store',
                    type=str,
                    help='Specify the directory for which the data is stored')
parser.add_argument('--csv-out',
                    action='store',
                    type=str,
                    help='Specify the file for which the csv data is stored at')
parser.add_argument('--from-parent-folder',
                    action='store_true',
                    default=False,
                    help='tb folder is the parent directory containing all the experiments tb folders of subfolders.')
args = parser.parse_args()

# Set threads
if 'LSB_DJOB_NUMPROC' in os.environ:
    allowed_processes = int(os.environ['LSB_DJOB_NUMPROC'])
else:
    allowed_processes = None


def matcher(tag):
    return (
            (
                    tag.startswith('metric/train') or
                    tag.startswith('metric/valid') or
                    tag.startswith('metric/test')
            ) and (tag.endswith('/mse') or tag.endswith('/acc') or tag.endswith('/loss')) or
            tag.endswith('epoch')
            # tag.endswith('/sparsity_error') or

    )


def create_reader(tensorboard_dir):
    return stable_nalu.reader.TensorboardMetricTwoDigitMNISTReader(
        tensorboard_dir,
        metric_matcher=matcher,
        step_start=1,
        processes=allowed_processes
    )


def write_csv_results(csv_out, reader):
    with open(csv_out, 'w') as csv_fp:
        for index, df in enumerate(reader):
            df.to_csv(csv_fp, header=(index == 0), index=False)
            csv_fp.flush()


def main():
    parent_path = args.tensorboard_dir
    if args.from_parent_folder:
        tb_dirs = os.listdir(parent_path)
        for tb_dir in tb_dirs:
            absolute_tb_dir = os.path.join(parent_path, tb_dir)
            reader = create_reader(absolute_tb_dir)
            csv_save_path = os.path.join(args.csv_out, tb_dir) + '.csv'
            write_csv_results(csv_save_path, reader)
    else:
        reader = create_reader(args.tensorboard_dir)
        write_csv_results(args.csv_out, reader)


if __name__ == '__main__':
    # Windows OS requires code to be encapsulated int main() otherwise multiprocessing will not work
    main()
