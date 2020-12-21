from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import numpy as np

# path = 'desc_dat_1231/desc_dat/'
path = 'desc_dat/'
files = ['desc_001d_p001.dat','desc_002d_p001.dat']
for file in files:
    with open(path+file,'r') as fin:
        alllines = fin.readlines()
        del alllines[0]
        measure = []
        fitting = []
        for i in range(0,alllines.__len__()):
            index = alllines[i].strip().split()
            measure.append(float(index[1]))
            fitting.append(float(index[2]))
        print(r2_score(measure,fitting))
        print(np.sqrt(mean_squared_error(measure,fitting)))
        print(mean_absolute_error(measure,fitting))
