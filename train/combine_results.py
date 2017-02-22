import argparse
import sys
import os
import csv


def main(args):
    prefix_l = len(args.prefix)
    files = os.listdir(args.input_dir)
    file_asb_paths = [os.path.join(args.input_dir, f) for f in files if f[:prefix_l]==args.prefix]

    combine = []
    for f_p in file_asb_paths:
        with open(f_p, "rb") as f_csv:
            reader = csv.reader(f_csv)
            for row in reader:
                combine.append(row)

    write_csv(args.output_file, combine)

    return


def write_csv(path, data):
    with open(path, 'a') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(data)
        print("result saved to {}".format(path))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory containing csv files to combine')
    parser.add_argument('output_file', type=str, help='Output file abs path')
    parser.add_argument('prefix', type=str, help='Prefix of the files')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))