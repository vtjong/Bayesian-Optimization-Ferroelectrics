from re import T
import matplotlib.pyplot as plt
import numpy as np

# plt.plot([100,1024,10240],[80,7.8,5.78], 'o-', c = '#580F41')
# plt.plot(5, c = '#A9561E')
# plt.ylabel('delay time (ms)')
# plt.xlabel('bandwidth (Kbs)')
# plt.title(device + " PUND "+ iter)
# plt.legend(loc="upper right")
# plt.savefig(dir + "/PUND/" + device + "_" + iter +"_PUND-plot")  
# plt.show() 

# print(np.ones([2]))
# print([1,1])
# n=30
# print([n for i in range(2)])

# A = np.array([[56.0, 0.0, 4.4, 68.0],
#             [1.2, 104.0, 52.0, 8.0],
#             [1.8, 135.0, 99.0, 0.9]])
# cal = A.sum(axis=0)
# cal = cal.reshape(1,4)
# print(cal)
# print(A/cal)

A = np.array([1., 2., 3., 4., 5.])
B = np.array([6., 7., 8., 9., 10.])
var_array = [A,B]

def x_normalizer(X, var_array = var_array):
    def max_min_scaler(x, x_max, x_min):
        return (x-x_min)/(x_max-x_min)
    x_norm = []
    for x in (X):
        print("x", x)
        x_norm.append([max_min_scaler(x[i], 
                         max(var_array[i]), 
                         min(var_array[i])) for i in range(len(x))])
        print(x_norm)
    return np.array(x_norm)

X = np.vstack((A, B)).T
x_normalizer(X)