# from pandas import rolling_mean
from pylab import *

def read_tracer_outfile(filename, f4=False, step=1):
    """ 
    """
    if f4:
        out_dtype=[('t', int64), ('n', int64), ('f2', float64), ('f3', float64),
            ('f4', float64)]
    else:
        out_dtype=[('t', int64), ('n', int64), ('f2', float64), ('f3', float64)]
    data = loadtxt(filename, dtype=out_dtype)
    if step > 1:
        # Downsample the data
        imax = len(data['t']) - 1
        indices = np.append(arange(0, imax, step), imax)
        data = data[indices]
    data['f2'][data['f2'] == -1] = nan 
    data['f3'][data['f3'] == -1] = nan 
    if f4:
        data['f4'][data['f4'] == -1] = nan 
    return data

# # Slow way to store a bunch of runs together:
# 
# filenames = ['lineage' + str(i) + '.txt' for i in range(0, 10000)]
# # li = [read_tracer_outfile(f) for f in filenames]
# # allruns = array(li, dtype=[('t', int64), ('n', int64), ('fst', float64), ('gamma', float64)])
# 
# # or just
# filenames = ['lineage' + str(i) + '.txt' for i in range(1, 1001)]
# filenames = ['lineage' + str(i) for i in range(1, 1001)]
# allruns = array([read_tracer_outfile(f, f4=True) for f in filenames])
# np.save('runs_5000.npy', allruns)

def consolidate_files(filenames, outfile, f4=False, step=1):
    allruns = array([read_tracer_outfile(f, f4, step) for f in filenames])
    np.save(outfile, allruns)
    return allruns

# 
# ####
# # For getting rid of nan's
# # sed -i 's/nan/1/' run{0..4999}
# # sed -i 's/\s\{2,\}/ /g' run{0..4999} 
# 
# # Estimate the standard deviation of Fst between generations 200 and 500 from
# # all 5000 runs
# np.std(arr_n1000[:,200:500]['fst'], ddof=1)
# 
# np.mean(arr_n1000[:,200:500]['fst'])

def cumulative_weights(lineages):
    """Calculate cumulative weight $W(t)$ of the lineage up to time $t$.
    """
    def weight_traj(lineage):
        time_intervals = lineage['t'][1:] - lineage['t'][:-1]
        weights = np.cumsum(time_intervals * lineage['n'][:-1], axis=0)
        return weights
    try: 
        weights = array([weight_traj(lineage) for lineage in lineages])
    # If lineages is actually a single lineage, this will yield an IndexError.
    # In this case, convert to a list of one lineage first
    except IndexError:
        lineages = [lineages]
        weights = array([weight_traj(lineage) for lineage in lineages])
    return weights

def final_weights(lineages):
    """Calculate cumulative weight $W(t)$ of the lineage up to time $t$.
    """
    def weight(lineage):
        time_intervals = lineage['t'][1:] - lineage['t'][:-1]
        weight = np.sum(time_intervals * lineage['n'][:-1], axis=0)
        return weight
    try: 
        weights = array([weight(lineage) for lineage in lineages])
    # If lineages is actually a single lineage, this will yield an IndexError.
    # In this case, convert to a list of one lineage first
    except IndexError:
        lineages = [lineages]
        weights = array([weight(lineage) for lineage in lineages])
    return weights

def zero_assortment_times(lineages):
    def za_time(lineage):
        time_intervals = lineage['t'][1:] - lineage['t'][:-1]
        tot_time = np.sum(time_intervals * (1-lineage['f2'][:-1]), axis=0)
        return tot_time
    try: 
        times = array([za_time(lineage) for lineage in lineages])
    # If lineages is actually a single lineage, this will yield an IndexError.
    # In this case, convert to a list of one lineage first
    except IndexError:
        lineages = [lineages]
        times = array([za_time(lineage) for lineage in lineages])
    return times

def mean_assortments(lineages):
    try: 
        mean_f2s = array([nanmean(lineage['f2']) for lineage in lineages])
    # If lineages is actually a single lineage, this will yield an IndexError.
    # In this case, convert to a list of one lineage first
    except IndexError:
        lineages = [lineages]
        mean_f2s = array([nanmean(lineage['f2']) for lineage in lineages])
    return mean_f2s

