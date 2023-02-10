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

# import re
# s1 = "thishasadigit4here"
# m = re.search(r"\d", s1).start()
cut_ext = lambda fn:fn[:fn.rfind('.')]
get_iter = lambda fn:fn[fn.rfind('_'):]
print(get_iter("fuckyou_4.txt"))