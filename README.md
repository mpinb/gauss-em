# GAUSS-EM: Guided accumulation of ultrathin serial sections with a static magnetic field for volume electron microscopy

Example for computationally solving the order of unordered serial slices. This overall approach is presented in the manuscript:

xxx - manuscript citation (at least with title / authors)

This example can be run either via command line or via the included jupyter notebook. It orders 20 example sequential slice images that are downsampled to a resolution of 256 nm and are originally only sorted by the order in which they were imaged. In the GAUSS-EM method the imaged order does not correspond to the order in which the slices were cut from the tissue block. Although the example will work without a gpu, it will take several hours of runtime. With a gpu (only tested with nVidia) it should run in 5-10 minutes. Scaling up to larger number of slices requires parallelization on a cluster.

In general the algorithm is O(n<sup>2</sup>) where n is the number of slices. This is because it computes a full distance (slice dissimilarity) matrix between all pairs of slice images. The most compute-intensive portion performs KNN searches between SIFT descriptors to find matching keypoints between images. The runtime and memory usage for these searches scales with the number of SIFT descriptors / keypoints per image, which in turn typically scales with higher resolution of the utilized slice images. Options for reducing run time include:
- computation of upper or lower triangular only portion of the distance matrix
- limiting the maxiumum number of SIFT features
- using an approximate KNN search library instead of exact KNN searches

The choice of solver for the resulting Traveling Salesman Problem (TSP) could also theoretically effect runtime. Generally the distance matrix tends to be rather sparse in the sense that the similarity between images drops off very quickly for further neighbors of each slice in the original ordering. This has the tendency of greatly reducing the runtime of the TSP solver. This example contains images where the similarity falls off quickly, is relatively uniform between the images and the direct neighbors always have the highest similarity. For this reason a greedy-algorithm approximation for solving the TSP can be utilized. These conditions are not in general the case, so an optimal TSP-solver is recommended for production usage.

## Install with Dependencies

Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [mambaforge](https://github.com/conda-forge/miniforge#mambaforge) base environment.

### Recommended: conda create environment from yaml

Create an environment with all necessary dependencies:
```
conda env create -f gaussem-environment.yaml
```

### Alternative: shell install script

Only tested with `bash` and using mambaforge or miniconda (need to modify some variables in script for miniconda):

```
bash install-with-mambaforge.sh
```

## Usage

Either load the [python notebook](slice_order_solver.ipynb) that works through the example, or use the command line interface, for example:

```
python SliceOrderSolver.py --ngpus 1
```
