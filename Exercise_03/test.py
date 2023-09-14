#from __future__ import print_function
import pickle
import numpy as np
import cv2
import pymp


#data = pickle.load(open('C:/Users/HP/Downloads/1-IMG_MAX_9964_SIFT_patch_pr.pkl', 'rb'))
# max_files = 3
# indices = np.random.permutation(max_files)
# files = np.array(['a','b','c'])[indices]
# # print(files)
# #bf = cv2.BFMatcher.matcher.knnMatch(cv2.NORM_HAMMING, crossCheck=True)
# a = np.array([1,2,3])
# b = np.array([4,5,6])
# # a = np.append(a,b)
# c = np.concatenate(b, axis=0)
# # a = np.append(a,b)
# print(c)

# ex_array = pymp.shared.array((100,), dtype='uint8')
# with pymp.Parallel(4) as p:
#     for index in p.range(0, 100):
#         ex_array[index] = 1
#         print(index)
#         # The parallel print function takes care of asynchronous output.
# #         p.print('Yay! {} done!'.format(index))
#
# from multiprocessing import Process, freeze_support


ex_array = pymp.shared.array((100,), dtype='uint8')
with pymp.Parallel(4) as p:
    for index in p.range(0, 100):
        ex_array[index] = 1
        print(index)
        # The parallel print function takes care of asynchronous output.
#         p.print('Yay! {} done!'.format(index))

