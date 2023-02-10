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

def addingOne(arr):
    eArr = [] 
    for i in arr:
        eArr.append(i +1)
    print (eArr)
    return eArr

def adding_one_shifu(arr):
    eArr = [i+1 for i in arr] 
    return eArr


addingOne([1, 2])