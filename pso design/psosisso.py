# Plot particle history as animation

import numpy as np

from sko.PSO import PSO

import time





def demo_func(x):

    DT, Cr,Ni,C,TT,THT = x

    return -50.58*pow(DT*(Cr+Ni),1/3)-2.24536*C*(TT+DT-THT)-158.7

time_start = time.time()

pso = PSO(func=demo_func, dim=6, pop=20, max_iter=400, lb=[30, 0 , 0 , 0 , 0 , 10], ub=[850 , 0.55 , 1.69 , 0.23 , 160 , 30])

pso.record_mode = True

pso.run()

print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

time_end = time.time()

# %% Now Plot the animation

# import matplotlib.pyplot as plt
#
# from matplotlib.animation import FuncAnimation



# record_value = pso.record_value
#
# X_list, V_list = record_value['X'], record_value['V']



# fig, ax = plt.subplots(1, 1)

# ax.set_title('title', loc='center')

# line = ax.plot([], [], 'b.')



# X1_grid, X2_grid,X3_grid,X4_grid,X5_grid,X6_grid = np.meshgrid(np.linspace(1.0,1.0, 0.1,0.1,0.1,1.0, 1.0), np.linspace(1.0,1.0, 0.1,0.1,0.1,1.0, 1.0))
#
# Z_grid = demo_func((X1_grid, X2_grid,X3_grid,X4_grid,X5_grid,X6_grid))
#
# # ax.contour(X_grid, Y_grid, Z_grid, 20)
# print(Z_grid)
print(time_end-time_start)

# ax.set_xlim(-1, 1)

# ax.set_ylim(-1, 1)



# plt.ion()

# p = plt.show()




#
# def update_scatter(frame):
#
#     i, j = frame // 10, frame % 10
#
#     ax.set_title('iter = ' + str(i))
#
#     X_tmp = X_list[i] + V_list[i] * j / 10.0
#
#     plt.setp(line, 'xdata', X_tmp[:, 0], 'ydata', X_tmp[:, 1])
#
#     return line





# ani = FuncAnimation(fig, update_scatter, blit=True, interval=25, frames=300)

# plt.show()



# ani.save('pso.gif', writer='pillow')