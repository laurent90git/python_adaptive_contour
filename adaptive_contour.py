import time
import multiprocessing
multiprocessing.set_start_method("fork")
from scipy.interpolate import griddata, interp2d
import numpy as np
from matplotlib import use
use("PDF")
import matplotlib.pyplot as plt
import tqdm
from multiprocessing import Pool

def chi2fn_demo(x, y):
    if x > y:
        return np.inf
    return (np.sin(x) + np.sin(y))**2.


_func = None

def worker_init(func):
    global _func
    _func = func
  
def worker(x):
    return _func(x)

def xmap(func, iterable, processes=None):
    with Pool(processes, initializer=worker_init, initargs=(func,)) as p:
        return p.map(worker, iterable)



def recursive_adapt(chi2fn, x_1D, y_1D, z_2D, z_2D_mask, contour_levels, all_xyz, cur_depth, max_depth, use_parallel):
    all_to_run = []
    all_to_run_ij = []
    
    for i in range(len(x_1D)):
        for j in range(len(y_1D)):
            if z_2D_mask[j,i] == 0:
                all_to_run.append([x_1D[i], y_1D[j]])
                all_to_run_ij.append([i, j])
                #z_2D[j,i] = chi2fn(x_1D[i], y_1D[j])
                #z_2D_mask[j,i] = 1


    if use_parallel:
        all_results = xmap(chi2fn, all_to_run)
    else:
        all_results = []

        if len(all_to_run) > 5:
            range_to_use = tqdm.tqdm
        else:
            range_to_use = lambda x: x

        for item in range_to_use(all_to_run):
            all_results.append(chi2fn(item))
        
    assert len(all_results) == len(all_to_run_ij)

    for i in range(len(all_results)):
        z_2D[all_to_run_ij[i][1], all_to_run_ij[i][0]] = all_results[i]
        z_2D_mask[all_to_run_ij[i][1], all_to_run_ij[i][0]] = 1

        
    if len(x_1D) > 3:
        range_to_use = tqdm.trange
    else:
        range_to_use = range
                
    for i in range_to_use(len(x_1D) - 1):
        for j in range(len(y_1D) - 1):
            sub_z = z_2D[j:j+2, i:i+2]
            inds = np.where(1 - np.isnan(sub_z) - np.isinf(sub_z))
            good_sub_z = sub_z[inds]
            
            if len(sub_z[inds] > 1):
                max_z = np.max(good_sub_z)
                min_z = np.min(good_sub_z)

                delta_z = max_z - min_z
                
                min_diff_from_contour = min([np.abs(item - contour_levels).min() for item in good_sub_z])

                #if min_diff_from_contour < 50:
                #    print("min_diff_from_contour", min_diff_from_contour, good_sub_z, contour_levels, delta_z)
                #    ffff
                    
                if cur_depth == max_depth:
                    good_square = 1
                elif delta_z < 0.25:
                    good_square = 1
                elif min_diff_from_contour > 200:
                    good_square = 1
                elif delta_z > min_diff_from_contour*2:
                    good_square = 0
                elif any((max_z > contour_levels)*(min_z < contour_levels)):
                    good_square = 0
                else:
                    good_square = 1


                if good_square:
                    if 1 - np.isnan(z_2D[j,i]) - np.isinf(z_2D[j,i]):
                        all_xyz.append((x_1D[i], y_1D[j], z_2D[j,i]))
                else:
                    sub_x = np.linspace(x_1D[i], x_1D[i+1], 3)
                    sub_y = np.linspace(y_1D[j], y_1D[j+1], 3)
                    new_sub_z = np.zeros([3,3], dtype=np.float64)
                    new_sub_z[0::2, 0::2] = sub_z
                    new_sub_mask = np.ones([3,3], dtype=np.int32)
                    new_sub_mask[1,:] = 0
                    new_sub_mask[:,1] = 0

                    all_xyz.extend(recursive_adapt(chi2fn = chi2fn, x_1D = sub_x, y_1D = sub_y, z_2D = new_sub_z, z_2D_mask = new_sub_mask, contour_levels = contour_levels, all_xyz = [], cur_depth = cur_depth + 1, max_depth = max_depth, use_parallel = use_parallel))

    for i in range(len(x_1D)):
        for j in range(len(y_1D)):
            if i == len(x_1D) - 1 or j == len(y_1D) - 1:
                if 1 - np.isnan(z_2D[j,i]) - np.isinf(z_2D[j,i]):
                    all_xyz.append((x_1D[i], y_1D[j], z_2D[j,i]))
                
    return all_xyz

def adaptive_contour(chi2fn, x_1D, y_1D, contour_levels, max_depth = 4, interp_points = 500, use_parallel = True):
    all_xyz = recursive_adapt(chi2fn = lambda xy: chi2fn(xy[0], xy[1]), x_1D = x_1D, y_1D = y_1D,
                              z_2D = np.zeros([len(y_1D), len(x_1D)], dtype=np.float64),
                              z_2D_mask = np.zeros([len(y_1D), len(x_1D)], dtype=np.int16), contour_levels = contour_levels, all_xyz = [], cur_depth = 1, max_depth = max_depth, use_parallel = use_parallel)

    print(len(all_xyz))
    all_xyz = set(all_xyz)
    print(len(all_xyz))
    
    all_xyz = np.array(list(all_xyz)).T

    grid_x = np.linspace(x_1D[0], x_1D[-1], interp_points)
    grid_y = np.linspace(y_1D[0], y_1D[-1], interp_points + 1) # Different length so that the grid can't get mistakenly transposed.
    grid_z = griddata(all_xyz[:2].T, all_xyz[2], tuple(np.meshgrid(grid_x, grid_y)), method='linear')#'cubic')

    return all_xyz, grid_x, grid_y, grid_z




if __name__ == "__main__":
    x_1D_start = np.linspace(0, 5, 10)
    y_1D_start = np.linspace(0, 5, 10)
    
    all_xyz, grid_x, grid_y, grid_z = adaptive_contour(chi2fn_demo, x_1D = x_1D_start, y_1D = y_1D_start,
                                                       contour_levels = np.array([1., 2.]))
    
    print(grid_x.shape, grid_y.shape, grid_z.shape)

    
    plt.figure(figsize = (48, 24))
    
    plt.subplot(1,2,1)
    plt.scatter(all_xyz[0], all_xyz[1], c = all_xyz[2], cmap = "gist_rainbow")
    plt.colorbar()
    plt.subplot(1,2,2)
    print(grid_z)
    plt.contourf(grid_x, grid_y, grid_z, levels = [-1, 1, 2, 10])

    plt.savefig("adapt.pdf")
    plt.close()

