"""Functions for processing data from valley-crossing time simulations.

Data to be read in must be generated by the valley_crossing_time program.
"""
import numpy as np

def read_outfile(filename):
    """Read the outcomes of all runs from a file.

    filename -- a string giving a single filename
    """
    sim_dtype=[('seed', '>u8'), ('n0', '>u8'), ('n1', '>u8'), ('n2', '>u8'),
            ('t2', '>u8'), ('tfix', '>u8')]
    data = np.loadtxt(filename, dtype=sim_dtype, delimiter=' ')
    return data

def summarize_outfiles(filenames):
    sim_dtype=[('seed', '>u8'), ('n0', '>u8'), ('n1', '>u8'), ('n2', '>u8'),
            ('t2', '>u8'), ('tfix', '>u8')]
     # Time when first successful 2-mutant occurs
    est_t2 = np.zeros(len(filenames), 
            dtype=[('mean', '>f8'), ('sd', '>f8'), ('se', '>f8')])
    # Time when 2-mutant fixes
    est_tfix = np.zeros(len(filenames), 
            dtype=[('mean', '>f8'), ('sd', '>f8'), ('se', '>f8')])
    # Calculate the estimated mean, standard deviation, and standard error of
    # t2 and tfix for each file
    for i in range(len(filenames)):
        data = np.loadtxt(filenames[i], dtype=sim_dtype, delimiter=' ')
        # 2-mutant arises
        s = np.std(data['t2'], ddof=1)
        se = s / np.sqrt(len(data['t2']))
        est_t2[i] = (np.mean(data['t2']), s, se)
        # 2-mutant fixes
        s = np.std(data['tfix'], ddof=1)
        se = s / np.sqrt(len(data['tfix']))
        est_tfix[i] = (np.mean(data['tfix']), s, se)
    return est_t2, est_tfix

