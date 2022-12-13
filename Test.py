import numpy as np
target_points = np.array([[-0.314,1.661,0.45],[0,1.661,0.45],[0.314,1.661,0.45],[-0.314,1.347,0.45],[0,1.347,0.45],[0.314,1.347,0.45],[-0.314,1.033,0.45],[0,1.033,0.45],[0.314,1.033,0.45]])
r= 0.58
temp = np.zeros((9,3))
for i in range(9): 
    temp[i][:] = np.array([target_points[i][0]*r,(target_points[i][1]-target_points[4][1])*r+target_points[4][1], target_points[i][2]])
target_points = temp
print(target_points)