from sklearn.metrics import r2_score
from sklearn.metrics import max_error
# from sklearn.metrics import mean_squared_error

path = 'desc_dat3/'
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
        # print(max_error(measure,fitting))
