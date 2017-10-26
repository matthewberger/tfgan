"""
Author: Jixian Li <jixianli@email.arizona.edu>
Description:
    Entry point for volume renderer
"""
from multiprocessing import Process
import argparse
import os
import numpy as np
from datetime import datetime


def worker(tid, files, outdir, opts):
    print("%s: thread %d started, process files %d .. %d " % (str(datetime.now()), tid, opts[-2], opts[-1] - 1))
    from renderer import Renderer
    renderer = Renderer(files, outdir, opts, tid)
    renderer.render_images()
    print("%s: thread %d ended" % (str(datetime.now()), tid))


def main():
    parser = argparse.ArgumentParser('pvpython master.py')
    # positional arguments
    parser.add_argument('volume', help='volume file')
    parser.add_argument('view', help='view file')
    parser.add_argument('opacity', help='opacity file')
    parser.add_argument('color', help='color file')
    # optional arguments
    parser.add_argument('-s', '--scalar', default='Scalars_', help='name of the scalar field')
    parser.add_argument('--samples', type=int, default=8, help='number of sample per pixel')
    parser.add_argument('--ambients', type=int, default=4, help='number of ambient samples')
    parser.add_argument('-p', '--procs', type=int, default=1, help='number of processes')
    parser.add_argument('-o', '--outdir', default='./', help='output directory')
    parser.add_argument('-n', default=100, type=int,
                        help='number of images to render, assumed to be a reasonable number, DEFAULT=100')

    args = parser.parse_args()

    """
    Setup the directories
    """
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    if args.outdir[-1] != '/':
        args.outdir += '/'

    img_folder = args.outdir + 'imgs_%dss%das/' % (args.samples, args.ambients)
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    inputs_folder = args.outdir + 'inputs/'
    if not os.path.exists(inputs_folder):
        os.makedirs(inputs_folder)

    """
    Setup the processes
    """
    files = (args.volume, args.view, args.opacity, args.color)
    outdir = (args.outdir, img_folder, inputs_folder)
    procs = []
    count = args.n // args.procs  # job per proc
    for i in range(args.procs):
        start = i * count
        end = i * count + count
        if i == args.procs - 1:
            end = args.n
        opts = [args.scalar, args.samples, args.ambients, start, end]
        p = Process(target=worker, args=(i, files, outdir, opts))
        p.start()
        procs.append(p)

    """
    Post processing
    """
    for p in procs:
        p.join()  # wait till all finishes

    # TODO merge index files


if __name__ == '__main__':
    main()
