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

A = np.array([[56.0, 0.0, 4.4, 68.0],
            [1.2, 104.0, 52.0, 8.0],
            [1.8, 135.0, 99.0, 0.9]])
cal = A.sum(axis=0)
cal = cal.reshape(1,4)
print(cal)
# print(A/cal)