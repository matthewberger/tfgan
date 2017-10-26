import argparse
import os
import sys
import re
from tqdm import tqdm
import numpy as np


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def indexing(base_dir, view, opacity, color):
    if base_dir[-1] != "/":
        base_dir += "/"
    if not os.path.exists(base_dir):
        eprint(base_dir, " doesn't exists")
        exit(1)
    inputs_dir = base_dir + "inputs/"
    if not os.path.exists(inputs_dir):
        os.makedirs(inputs_dir)

    imgs = [f for f in tqdm(os.listdir(base_dir + "imgs/")) if f.startswith("vimage")]
    imgs.sort(key=lambda x: int(re.search("\d+", x).group(0)))

    files_index = open(base_dir + "files.csv", "w")
    inputs_index = open(base_dir + "inputs.csv", "w")

    for img in tqdm(imgs):
        files_index.write(img + "\n")
        i = int(re.search("\d+", img).group(0))
        vs, op, cs = view[i, :], opacity[i, :], color[i, :]
        input_name = "input%d.csv" % i
        np.savetxt(inputs_dir + input_name,
                   np.hstack((vs, op.flatten(), cs.flatten())),
                   delimiter=',', fmt='%1.4f')
        inputs_index.write(input_name + "\n")
    files_index.close()
    inputs_index.close()


def main():
    parser = argparse.ArgumentParser("python indexing.py")
    parser.add_argument("dataroot", help="root directory of the output directory")
    parser.add_argument("view", help="path to npy version view file")
    parser.add_argument("opacity", help="path to npy version opacity file")
    parser.add_argument("color", help="path to npy version color file")
    args = parser.parse_args()
    r = args.dataroot
    v = np.load(args.view)
    o = np.load(args.opacity)
    c = np.load(args.color)
    indexing(r, v, o, c)


if __name__ == '__main__':
    main()
