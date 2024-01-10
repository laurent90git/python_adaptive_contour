# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:03:14 2023

adapted from https://github.com/rubind/adaptive_contour
@author: lfrancoi
"""

import time
from scipy.interpolate import griddata, interp2d
import numpy as np
# from matplotlib import use
# use("PDF")
import matplotlib.pyplot as plt
import tqdm
from multiprocessing import Pool

def chi2fn_demo1(x, y):
    if x > y:
        return np.inf
    return (np.sin(x) + np.sin(y))**2.

def f0(z):
    if np.abs(z)>1e-6:
      return np.abs(z*z - 0.6/z)
    else:
      return np.inf

def chi2fn_demo2(x, y):
    z = x+1j*y
    return f0(f0(f0(f0(f0(f0(z))))))

_func = None

def worker_init(func):
    global _func
    _func = func

def worker(x):
    return _func(x)



import multiprocessing

bParallel=False
pool = None
nprocs = None

def xmap(func, iterable, bProgress=False):

    if bParallel:
      global pool
      if pool is None:
        print('Creating parallel pool...')
        pool = Pool(processes=nprocs, initializer=worker_init, initargs=(func,))


      nb_tasks = len(iterable)
      if bProgress:
        # return list(tqdm.tqdm(pool.map(worker, iterable), total=nb_tasks))
        return list(tqdm.tqdm(pool.imap(worker, iterable, chunksize=nprocs*4), total=nb_tasks, desc='1st level evaluation'))
      else:
        # return list( pool.imap(worker, iterable, chunksize=50) )
        return pool.map(worker, iterable)
    else:
      if bProgress:
        return [func(i) for i in tqdm.tqdm(iterable)]
      else:
        return [func(i) for i in iterable]


def recursive_adapt(chi2fn, x_1D, y_1D, z_2D, z_2D_mask,
                    contour_levels, all_xyz, cur_depth, max_depth,
                    atol, rtol, max_points=np.inf):
    all_to_run = []
    all_to_run_ij = []

    for i in range(len(x_1D)):
        for j in range(len(y_1D)):
            if z_2D_mask[j,i] == 0:
                all_to_run.append([x_1D[i], y_1D[j]])
                all_to_run_ij.append([i, j])

    # sample function on current grid (on the not yet evaluated points)
    all_results = xmap(chi2fn, all_to_run, bProgress=cur_depth==1)
    assert len(all_results) == len(all_to_run_ij)

    # store results
    for i in range(len(all_results)):
        z_2D[all_to_run_ij[i][1], all_to_run_ij[i][0]] = all_results[i]
        z_2D_mask[all_to_run_ij[i][1], all_to_run_ij[i][0]] = 1


    # go through current structure to see where more points are needed
    bContinue = True
    import itertools
    indices = itertools.product( range(len(x_1D) - 1),
                                 range(len(y_1D) - 1) )
    indices = list(indices)

    if cur_depth==1:
      iterable = tqdm.trange(len(indices), desc='refinement')
    else:
      iterable = range(len(indices))
    for it in iterable:
            i,j = indices[it]
            if not bContinue:
                break
            sub_z = z_2D[j:j+2, i:i+2] # include neighbours
            # print(x_1D[i:i+2], y_1D[j:j+2]) # current square
            inds = np.where(1 - np.isnan(sub_z) - np.isinf(sub_z)) # find where there are no NaNs or InF
            good_sub_z = sub_z[inds]

            if len(sub_z[inds] > 1):
              # get max and min values of function in the current points set
                max_z = np.max(good_sub_z)
                min_z = np.min(good_sub_z)

                delta_z = max_z - min_z

                # current minimum distance to one of the user-specified values
                distances = np.vstack([np.abs(item - contour_levels) for item in good_sub_z])
                imin = np.argmin(distances)

                jj = imin % distances.shape[1]
                ii = imin // distances.shape[1]
                closest_contour = contour_levels[jj]
                # min_diff_from_contour = distances[ii,jj]
                error = abs(delta_z)/(atol+ rtol*abs(closest_contour)) # relative distance to the closest contour

                # do we need to refine further ?
                if cur_depth == max_depth:
                    good_square = 1

                elif any((max_z > contour_levels)*(min_z < contour_levels)):
                    # we cross one of the contours in the current square
                    if error < 1:
                        # sufficiently refined
                        good_square = 1
                    else:
                        good_square = 0

                else:
                    good_square = 1

                if good_square: # do not refine further
                    # store point as final
                    # if 1 - np.isnan(z_2D[j,i]) - np.isinf(z_2D[j,i]):
                      all_xyz.append((x_1D[i], y_1D[j], z_2D[j,i], cur_depth))
                else:
                    # create refined grid
                    #
                    #     x-----O-----x
                    #     |     |     |
                    #     |     |     |
                    #     O-----O-----O
                    #     |     |     |
                    #     |     |     |
                    #     x-----O-----x
                    #
                    # x: current points
                    # O: new points for refinement
                    sub_x = np.linspace(x_1D[i], x_1D[i+1], 3)
                    sub_y = np.linspace(y_1D[j], y_1D[j+1], 3)
                    new_sub_z = np.zeros([3,3], dtype=np.float64)
                    new_sub_z[0::2, 0::2] = sub_z # plug in data already computed
                    new_sub_mask = np.ones([3,3], dtype=np.int32)
                    # refine a cross between the four current level points
                    new_sub_mask[1,:] = 0
                    new_sub_mask[:,1] = 0

                    all_xyz.extend(recursive_adapt(chi2fn = chi2fn,
                                                   x_1D = sub_x, y_1D = sub_y,
                                                   z_2D = new_sub_z,
                                                   z_2D_mask = new_sub_mask,
                                                   contour_levels =contour_levels,
                                                   all_xyz = [],
                                                   atol=atol, rtol=rtol,
                                                   cur_depth = cur_depth + 1,
                                                   # max_depth = min(max_depth, cur_depth + 2)))
                                                    max_depth = max_depth))
                    # print(len(all_xyz))
                    # if len(all_xyz)>max_points:
                    #   bContinue=False
                    # TODO: no way to stop before the max number of points
                    # TODO: perform one level refinment at a time and store
                    # new candidates for refinement afterwards..
                    # TODO: we have to do it by patch...


    # add missing points ?
    # todo: why loop over all indices if we only seek the last few ones ???
    for i in range(len(x_1D)):
        for j in range(len(y_1D)):
            if i == len(x_1D) - 1 or j == len(y_1D) - 1:
                # if 1 - np.isnan(z_2D[j,i]) - np.isinf(z_2D[j,i]):
                    all_xyz.append((x_1D[i], y_1D[j], z_2D[j,i], cur_depth))

    return all_xyz

def adaptive_contour(chi2fn, x_1D, y_1D, contour_levels,
                     atol = 1e-3, rtol=1e-2,
                     max_depth = 4, max_points = np.inf,
                     interp_points = 500,
                     parallel=1):
    """ Perform an adaptive sampling of the function chi2fn on the initial cartesian grid
        defined by x_1D and y_1D. The sampling grid is refined adaptively with a quadtree approach
        to accurately capture the contours of the function near the specified contour_levels.
        Comapred to the original grid size, the local grid size cannot
        be lower than 1/2**max_depth the original size.
        The final result is interpolated on a uniform grid with
        interp_points * (interp_points + 1) points"""

    global bParallel
    global nprocs
    nprocs = parallel

    if parallel>1:
      bParallel=True
      try:
        multiprocessing.set_start_method("fork")
      except RuntimeError as e:
        if str(e)=='context has already been set':
          pass
        else:
          raise e
    else:
      bParallel=False

    all_xyz = recursive_adapt(chi2fn = lambda xy: chi2fn(xy[0], xy[1]),
                              x_1D = x_1D,
                              y_1D = y_1D,
                              z_2D = np.zeros([len(y_1D), len(x_1D)], dtype=np.float64),
                              z_2D_mask = np.zeros([len(y_1D), len(x_1D)], dtype=np.int16),
                              contour_levels = contour_levels,
                              all_xyz = [], cur_depth = 1,
                              atol=atol, rtol=rtol,
                              max_depth = max_depth)

    # print(len(all_xyz))
    # all_xyz = set(all_xyz)
    all_xyz = np.array(all_xyz)
    print('Adaptive sampling finished with {} points'.format(all_xyz.shape[0]))
    print('Level stats:')
    for level in range( int(np.min(all_xyz[:,3])), int(np.max(all_xyz[:,3]))+1):
      print('   lvl {}: {} nodes'.format(level, np.count_nonzero((all_xyz[:,3]==level))))
    # print(all_xyz.size/3)

    all_xyz = np.array(list(all_xyz)).T # final list of refined points

    grid_x = np.linspace(x_1D[0], x_1D[-1], interp_points)
    grid_y = np.linspace(y_1D[0], y_1D[-1], interp_points + 1) # Different length so that the grid can't get mistakenly transposed.
    grid_z = griddata(all_xyz[:2].T, all_xyz[2], tuple(np.meshgrid(grid_x, grid_y)), method='linear')#'cubic')

    return all_xyz, grid_x, grid_y, grid_z




if __name__ == "__main__":
    cntfun = chi2fn_demo1
    x_1D_start = np.linspace(0, 5, 10)
    y_1D_start = np.linspace(0, 5, 10)
    target_levels = np.array([0.3, 2.])

    # cntfun = chi2fn_demo2
    # x_1D_start = np.linspace(-0.85, 1.35, 1000)
    # y_1D_start = np.linspace(-1.4, 1.4, 1001)
    # target_levels = np.array([1])

    all_xyz, grid_x, grid_y, grid_z = adaptive_contour(cntfun, x_1D = x_1D_start,
                                                       y_1D = y_1D_start,
                                                       rtol=1e-10, atol=1e-40,
                                                       contour_levels = target_levels,
                                                       max_depth=5)
    # Plot contour and node with one color for each level
    fig,ax = plt.subplots(1,2,figsize = (12, 6), sharex=True, sharey=True)
    from matplotlib import cm
    max_level = np.max(all_xyz[3])
    min_level = np.min(all_xyz[3])
    sc = ax[0].scatter(all_xyz[0], all_xyz[1], c = all_xyz[3],
                       vmin=min_level,
                       vmax=max_level,
                       cmap = cm.get_cmap('gist_rainbow', 1+max_level-min_level))
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sc, cax=cax, orientation='vertical')
    ax[1].contour(grid_x, grid_y, grid_z, levels = target_levels)
    for a in ax:
      a.set_xlabel('x')
      a.set_xlabel('x')
    ax[0].set_title('Nodes and refinement')
    ax[1].set_title('Contour')

