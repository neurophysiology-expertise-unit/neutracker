import cv2
from tifffile import imread
from multiprocessing import Pool,cpu_count
from functools import partial
from time import time as tic
from time import sleep
import numpy as np
import sys

def process_tiff(fname,parameters = None):
    '''
Process a tiff file with mptracker given parameters
    '''
    from .tracker import MPTracker
    cv2.setNumThreads(0)
    tracker = MPTracker(parameters)
    dat = imread(fname)
    if len(dat.shape) < 3:
        dat = [dat]
    res = []
    for frame in dat:
        res.append(tracker.apply(frame))
    del tracker
    return res


def par_process_tiff(filenames, parameters,
                     verbose = True,
                     nprocesses = int(cpu_count())):
    '''
Process a set of tiff files in parallel given tracker parameters
    '''
    if verbose:
        ts = tic()
    res = []
    pool = Pool(nprocesses)
    rs = pool.map_async(partial(
        process_tiff,
        parameters = parameters),
                        filenames,
                        callback = res.extend)
    pool.close()
    if verbose:
        from tqdm import tqdm
        ntasks = len(filenames)
        with tqdm(desc = 'Processing tiff files',total = ntasks) as pbar:
            while True:
                if rs.ready(): break
                pbar.update((ntasks - rs._number_left) - pbar.n)
    rs.wait()
    pool.join()
    # Flatten the list of lists
    res = [item for sublist in res for item in sublist]
    
    if verbose:
        toc = tic() - ts
        print('\t Analysed {0} frames in {1:.1f} s [{2:4.0f} fps]'.format(
            len(res), toc, len(res)/toc))
        sys.stdout.flush()
    return res
