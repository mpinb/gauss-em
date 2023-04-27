#!/usr/bin/env python3
"""SliceOrderSolver.py

This is the main file for the module-less example of computationally
solving serial section slice ordering. The example can be run from the
accompanying jupyter notebook (slice_order_solving.ipynb), or using this
file directly from the command line.

Example:
    Run from command line using a single GPU and the provided sample
    slice images:

        $ python SliceOrderSolver.py --ngpus 1

Copyright (C) 2023 Max Planck Institute for Neurobiology of Behavior - caesar

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import time
import argparse
import glob
import psutil
import warnings
import traceback

import numpy as np
import cv2
import tifffile
import multiprocessing as mp
import queue

from sklearn.linear_model import RANSACRegressor
from sklearnex.neighbors import NearestNeighbors

import faiss

from RigidRegression import RigidRegression, rigid_regression_loss


## multiprocessing worker functions

def compute_matches_job(gpu_index, ind, ntimgs, xinds, yinds, xdescriptors, ydescriptors,
        xkeypoints, ykeypoints, result_queue, lowe_ratio, min_feature_matches, ransac):
    """Defines python multiprocessing worker function computing percent matches.

    Args:
        gpu_index (int): Which gpu to use (-1 for no gpu).
        ind (int): Integer index identifying this worker.
        ntimgs (int): Total number of images in percent matches matrix.
        xinds (int list): First dimension matrix indices to compute percent matches for.
        yinds (int list): Second dimension matrix indices to compute percent matches for.
        xdescriptors (list): parallel to xinds, list of descriptors for each image
        ydescriptors (list): parallel to yinds, list of descriptors for each image
        xkeypoints (list): parallel to xinds, list of keypoints for each image
        ykeypoints (list): parallel to yinds, list of keypoints for each image
        result_queue (multiprocessing Queue): returns results for each percent matches element
        lowe_ratio (float): Lowe ratio threshold for descriptor matches.
        min_feature_matches (int): Threshold of minimum number of matches to add to percent matches.
        ransac (sklean ransac class): RANSAC regressor to use for fitting keypoints.

    Returns:
        Percent matches results are pushed to result_queue.

    """
    print('\tworker{}: started'.format(ind))
    if gpu_index >= 0:
        print('\tusing gpu index {}'.format(gpu_index))
        gpu_res = faiss.StandardGpuResources()      # use a single GPU
    nximgs = len(xdescriptors)
    nyimgs = len(ydescriptors)
    for x in range(nximgs):
        if gpu_index >= 0:
            index_cpu = faiss.IndexFlatL2(xdescriptors[x].shape[1])
            index_cpu.add(xdescriptors[x])
            index = faiss.index_cpu_to_gpu(gpu_res, gpu_index, index_cpu)
        else:
            nbrs = NearestNeighbors(n_neighbors=2, metric='l2', algorithm='kd_tree').fit(xdescriptors[x])

        for y in range(nyimgs):
            # diagonal is always left at zero (comparison of image to itself).
            if xinds[x] == yinds[y]: continue

            # get matching descriptors using the lowe ratio test.
            if gpu_index >= 0:
                distances, indices = index.search(ydescriptors[y], 2)
            else:
                distances, indices = nbrs.kneighbors(ydescriptors[y])
            sel = (distances[:,0] > 0)
            distances[np.logical_not(sel),0] = 1
            msk = np.logical_and(sel, distances[:,1] / distances[:,0] > lowe_ratio)
            msum = msk.sum()

            # threshold percent_matches at the specified minimum matching features level.
            if msum >= min_feature_matches:
                pypts_src = [x[0] for x in xkeypoints[x]]
                pts_src = np.array(pypts_src)[indices[:,0],:]
                pts_dst = np.array([x[0] for x in ykeypoints[y]])

                # fit the matching keypoints to an affine model, using ransac.
                pts_src = pts_src[msk,:]
                pts_dst = pts_dst[msk,:]
                try:
                    # NOTE: dangerous try-except-pass, do not put anything else inside this try.
                    with warnings.catch_warnings():
                        warnings.simplefilter(action='ignore')
                        ransac.fit(pts_src, pts_dst)
                        #ccoef = ransac.estimator_.coef_.copy()
                        cmask = ransac.inlier_mask_.copy()
                except:
                    print('ransac fit failed:')
                    print(traceback.format_exc())
                    #ccoef = None
                    cmask = None

                # return percent matches as percent of ransac inliers taken from matching SIFT descriptors
                result = {'indx':xinds[x], 'indy':yinds[y], 'percent_match':cmask.sum() / msk.size, 'iworker':ind}
            else:
                result = {'indx':xinds[x], 'indy':yinds[y], 'percent_match':0., 'iworker':ind}

            result_queue.put(result)
        #for y in range(nyimgs):
    #for x in range(nximgs):
    print('\tworker{}: completed'.format(ind))


## helper functions

def ANN(A):
    """All nearest neighbors algorithm.
    This is a greedy algorithm for solving the Traveling Salesman Problem.
    Takes the minimum path for the nearest neighbor greedy algorithm
    across all possible starting nodes.

    Args:
        A (ndarray shape (nnodes, nnodes)): pairwise distance matrix
            between nodes (locations / cities).

    Returns:
        min_path (list int): ordering of nodes corresponding to shortest TSP route.
        min_cost (float): corresponding cost (distance) of the min_path route.

    """
    N = A.shape[0]
    min_path = None
    min_cost = np.inf
    for i in range(N):
        path, cost = NN(A, i)
        if cost < min_cost:
            min_path = path
            min_cost = cost
    return min_path, min_cost

def NN(A, start):
    """Nearest neighbor algorithm.
    This is a greedy algorithm for solving the Traveling Salesman Problem.

    NOTE:
        https://stackoverflow.com/questions/17493494/nearest-neighbour-algorithm

    Args:
        A (ndarray shape (nnodes, nnodes)): pairwise distance matrix
            between nodes (locations / cities).
        start (int): which node to start at.

    Returns:
        min_path (list int): ordering of nodes corresponding to shortest TSP route.
        min_cost (float): corresponding cost (distance) of the min_path route.

    """
    path = [start]
    cost = 0
    N = A.shape[0]
    mask = np.ones(N, dtype=bool)  # boolean values indicating which
                                   # locations have not been visited
    mask[start] = False

    for i in range(N-1):
        last = path[-1]
        next_ind = np.argmin(A[last][mask]) # find minimum of remaining locations
        next_loc = np.arange(N)[mask][next_ind] # convert to original location
        path.append(next_loc)
        mask[next_loc] = False
        cost += A[last, next_loc]

    return path, cost


## order solving class, encapsulated to avoid global config.

class SliceOrderSolver():

    ## unexposed parameters

    # Will have an effect on runtime.
    # Typically 1000 or so is sufficient, but requires a higher value when the data is noisier,
    #   i.e., there is a low percentage of points that fit the model (20% or less).
    ransac_max = 5000

    # Tolerance for the ransanc regression.
    # It is better to define this based on the resolution.
    # For 256 nm pixels, 20 pixels == 5.12 um.
    residual_threshold = 20

    # lowe_ratio must be > 1, further from one is more conservative (rejects more matches)
    lowe_ratio = 1.42

    # this is the timeout for multiprocessing queues before checking for dead workers.
    # did not see a strong need for this to be drive from command line.
    queue_timeout = 180 # seconds

    # threshold for minimum matching SIFT features.
    # need at least a few features for the affine regression to work.
    min_feature_matches = 10

    def __init__(self, in_dir, out_dir, nthreads, ngpus):
        """Encapsulation of example of computationally solving serial section slice ordering.

        Args:
            in_dir (str): Input directory containing tiff files of unordered downsampled slice images.
            out_dir (str): Output directory into which to write ordered slice images (tiff files).
            nthreads (int): Number of parallel workers to use, recommend number of available cores.
                Use -1 to check for the number of available cores and use this value.
            ngpus (int): Number gpus to utilize for KNN descriptor computations. Use 0 for CPU-only.
                Use -1 to check for the number of available cores and use this value.

        """
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.ngpus = ngpus

        # setup for parallel workers / threads on multiple cores
        if nthreads < 1:
            # https://psutil.readthedocs.io/en/latest/index.html?highlight=cpu_count#psutil.cpu_count
            nthreads = len(psutil.Process().cpu_affinity())
        self.nthreads = nthreads
        cv2.setNumThreads(nthreads)


    def load_images_and_detect_keypoints(self):
        """Loads unordered slice images and detects SIFT keypoints.

        NOTE:
            SIFT keypoint computation is parallelized internally in opencv library.

        Returns:
            fns (list str): List of filenames of loaded slice images.
            keypoints (list list SIFT keypoints): List of SIFT keypoints for each loaded image.
            descriptors (list list SIFT descriptors): List of SIFT descriptors for each loaded image.

        """

        print( 'Loading images and detecting keypoints using {} threads'.format(self.nthreads) )
        t = time.time()

        fns = sorted(glob.glob(os.path.join(self.in_dir, '*')))
        nimgs = len(fns)
        img_shape = None
        keypoints = [None]*nimgs
        descriptors = [None]*nimgs
        sift = cv2.SIFT_create()
        for i,fn in zip(range(nimgs), fns):
            #print('\t' + fn)
            img = tifffile.imread(fn)
            if img_shape is None:
                img_shape = np.array(img.shape)
            else:
                assert( (img_shape == np.array(img.shape)).all() ) # all input images must be same size

            ckeypoints, descriptors[i] = sift.detectAndCompute(img, None)

            # https://stackoverflow.com/questions/10045363/pickling-cv2-keypoint-causes-picklingerror
            # this is needed for saving to dill file, but also can not insert the keypoints into multiprocessing
            #   queues at all without this. the queue implementation also uses pickling.
            keypoints[i] = [(point.pt, point.size, point.angle, point.response, point.octave,
                    point.class_id) for point in ckeypoints]
            # for reference, to restore:
            #keypoints[i] = [cv2.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], angle=point[2],
            #        response=point[3], octave=point[4], class_id=point[5]) for point in keypoints[i]]

        print('\tdone in {:.3f} s'.format(time.time() - t))

        return fns, keypoints, descriptors


    def compute_percent_matches(self, fns, keypoints, descriptors):
        """Use python multiprocessing (and optionally gpus) to parallelize percent matches computation.

        Args:
            fns (list str): List of filenames of loaded slice images.
            keypoints (list list SIFT keypoints): List of SIFT keypoints for each loaded image.
            descriptors (list list SIFT descriptors): List of SIFT descriptors for each loaded image.

        Returns:
            percent_matches (ndarray shape (nimages, nimages)): Percent matching SIFT features
                image similarity matrix. Bounded between [0, 1].

        """

        print( 'Computing percent matches matrix')
        t = time.time()
        nimgs = len(fns)

        # parallelize with workers using the second dimension of the percent matches matrix.
        use_nthreads = nimgs if self.nthreads > nimgs else self.nthreads
        print('\tusing {} processes on {} gpus'.format(use_nthreads, self.ngpus))
        nxblks = 1
        nyblks = use_nthreads
        nworkers = nxblks*nyblks
        xinds = np.array_split(np.arange(nimgs), nxblks)
        yinds = np.array_split(np.arange(nimgs), nyblks)

        # compute the number of comparisons for each worker.
        ncompares = np.zeros(nworkers, dtype=np.int64)
        i = 0
        for x in range(nxblks):
            for y in range(nyblks):
                xi, yi = np.meshgrid(xinds[x], yinds[y], indexing='ij')
                sel = (xi.reshape(-1) != yi.reshape(-1))
                ncompares[i] = sel.sum()
                i += 1
        ntcompares = ncompares.sum()
        print('\ttotal comparisons = {}'.format(ntcompares))
        del xi, yi, sel

        ransac = RANSACRegressor(estimator=RigidRegression(), stop_probability=1-1e-6, max_trials=self.ransac_max,
                loss=rigid_regression_loss, residual_threshold=self.residual_threshold**2,
                min_samples=self.min_feature_matches)

        # start the workers using python multiprocessing.
        workers = [None]*nworkers
        result_queue = mp.Queue(ntcompares)
        i = 0
        for x in range(nxblks):
            for y in range(nyblks):
                xd = descriptors[xinds[x][0]:xinds[x][-1]+1]
                yd = descriptors[yinds[y][0]:yinds[y][-1]+1]
                xk = keypoints[xinds[x][0]:xinds[x][-1]+1]
                yk = keypoints[yinds[y][0]:yinds[y][-1]+1]
                gpu_index = i % self.ngpus if self.ngpus > 0 else -1
                workers[i] = mp.Process(target=compute_matches_job, daemon=True,
                        args=(gpu_index, i, nimgs, xinds[x], yinds[y], xd, yd, xk, yk, result_queue, self.lowe_ratio,
                            self.min_feature_matches, ransac))
                workers[i].start()
                i += 1
        # NOTE: only call join after queue is emptied

        # retreive from the result queue and populate percent matches matrix.
        percent_matches = np.zeros((nimgs,nimgs), dtype=np.double)
        worker_cnts = np.zeros((nworkers,), dtype=np.int64)
        dead_workers = np.zeros((nworkers,), dtype=bool)
        nprint = 100
        i = 0
        dt = time.time()
        while i < ntcompares:
            if i > 0 and i % nprint ==0:
                print('{} through queue in {:.3f} s, worker_cnts:'.format(nprint, time.time()-dt,))
                print(worker_cnts)
                dt = time.time()

            try:
                res = result_queue.get(block=True, timeout=self.queue_timeout)
            except queue.Empty:
                for x in range(nworkers):
                    if not workers[x].is_alive() and worker_cnts[x] != ncompares[x]:
                        if dead_workers[x]:
                            print('worker {} is dead and worker cnt is {} / {}'.format(x,
                                worker_cnts[x], ncompares[x]))
                            assert(False) # a worker exitted with an error or was killed without finishing
                        else:
                            # to make sure this is not a race condition, try the queue again before error exit
                            dead_workers[x] = 1
                continue
            percent_matches[res['indx'], res['indy']] = res['percent_match']
            worker_cnts[res['iworker']] += 1
            i += 1
        assert(result_queue.empty())
        [x.join() for x in workers if x is not None]
        [x.close() for x in workers if x is not None]

        print('\tdone in {:.3f} s'.format(time.time() - t))

        return percent_matches


    def export_proposed_ordering(self, fns, path):
        """Export images as tiff files, with file names sortable by the solved ordering.

        Args:
            fns (list str): List of filenames of loaded slice images.
            path (list int): solved ordering of images (indices into fns).

        """

        print( 'Exporting images in proposed ordering')
        t = time.time()
        os.makedirs(self.out_dir, exist_ok=True)

        nimgs = len(fns)
        for i in range(nimgs):
            fn = fns[path[i]]
            bfn = os.path.basename(fn)
            nfn = 'order{:03d}_'.format(i) + bfn
            ofn = os.path.join(self.out_dir, nfn)
            #print('\t' + ofn)

            tifffile.imwrite(ofn, tifffile.imread(fn))
        print('\tdone in {:.3f} s'.format(time.time() - t))


# define main function for optional usage from command line.

def main():

    parser = argparse.ArgumentParser(description='SliceOrderSolver.py')
    parser.add_argument('--in-dir', nargs=1, type=str, default=['images'],
        help='path to the unordered image stack')
    parser.add_argument('--out-dir', nargs=1, type=str, default=['ordered'],
        help='path to save the ordered images')
    parser.add_argument('--nthreads', nargs=1, type=int, default=[0],
        help='number of workers (threads) to use')
    parser.add_argument('--ngpus', nargs=1, type=int, default=[0],
        help='number of gpus to use for knn searches')

    args = parser.parse_args()
    args = vars(args)

    # Instantiate the solver using command-line parameters.
    solver = SliceOrderSolver(args['in_dir'][0], args['out_dir'][0], args['nthreads'][0], args['ngpus'][0])

    # Load the images and compute keypoints. This should run in 30-60 seconds.
    fns, keypoints, descriptors = solver.load_images_and_detect_keypoints()

    # Compute the percent matches image similarity (distance) matrix.
    # This is computationally expensive, but with a single GPU enabled, it should run in about 5 minutes.
    # It will also run when ngpus==0 (when specified above) but will take many hours,
    #   even whne a large nthreads is specified (i.e., even when using many cores).
    # A GPU-enabled KNN search for the SIFT descriptors, or an approximate KNN search is critical for production usage,
    #   in addition to scaling across many nodes (i.e., needs to be process-parallelized and run on a cluster).
    percent_matches = solver.compute_percent_matches(fns, keypoints, descriptors)

    # Convert to the symmetric Traveling Salesman Problem.
    # Although one could restrict the percent matches computation to upper or lower triangular,
    #   the KNN searches are not symmetric, and for challenging ordering locations
    #   the full matrix computation is advantageous.
    percent_matches = np.maximum(np.triu(percent_matches), np.tril(percent_matches).T)
    percent_matches = percent_matches + percent_matches.T

    # Get minimum path for TSP using nearest neighbor greedy algorithm.
    # For production usage an optimal TSP solver is recommended, for example the concorde TSP solver:
    # https://www.math.uwaterloo.ca/tsp/concorde.html
    path, cost = ANN(1. - percent_matches)
    print('Proposed slice ordering:')
    print(path)

    # Export the images in the solved ordering.
    solver.export_proposed_ordering(fns, path)

if __name__=="__main__":
    #mp.set_start_method('spawn') # fork is dead, long live fork.
    main()
