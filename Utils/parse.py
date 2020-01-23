import numpy as np

def parse_dat(dat_path):
    data = np.genfromtxt('sample.dat',
                     skip_header=1,
                     skip_footer=1,
                     names=True,
                     dtype=None,
                     delimiter=' ')
    print(data)