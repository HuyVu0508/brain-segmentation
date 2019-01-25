from __future__ import print_function
import numpy as np
import tensorflow as tf
import scipy.io as sio
import os
from six.moves import cPickle as pickle
import h5py

# IBSR SIZE
c = np.array([255. / 2, 127. / 2])
b = np.array([255., 127.])

# # LBPA SIZE
# c = np.array([255. / 2, 123. / 2])
# b = np.array([255., 123.])

# OASIS SIZE
# c = np.array([175. / 2, 207. / 2])
# b = np.array([175., 207.])


# Go throuh all test brain
for k in range(6):

    # CHANGE DIR
    pathtd = "/home/ubuntu/Desktop/Features/Test each brain - each slice/"+str(k+1) + "/td"
    pathx = "/home/ubuntu/Desktop/Features/Test each brain - each slice/"+str(k+1) + "/X"
    pathy = "/home/ubuntu/Desktop/Features/Test each brain - each slice/"+str(k+1) + "/Y"
    pathsave="/home/ubuntu/PycharmProjects/HuyCode/Test Pickles/New feature - Test each brain - each slice/" + str(k+1)
    os.mkdir(pathsave)

    os.chdir(pathsave)

    filetd = [f for f in os.listdir(pathtd) if f.endswith(".mat")]
    filex = [f for f in os.listdir(pathx) if f.endswith(".mat")]
    filey = [f for f in os.listdir(pathy) if f.endswith(".mat")]

    filetd.sort()
    filex.sort()
    filey.sort()

    # READ MATLAB
    for i in range (len(filetd)):    #
    # for i in range(100):  #
        print('Process File ' + filetd[i])

        mat_contents = h5py.File("".join((pathtd,"/",filetd[i])))
        td = mat_contents['td']
        td = np.matrix(td).transpose()
        td = td * 1.0

        mat_contents = h5py.File("".join((pathx, "/", filex[i])))
        X = mat_contents['X']
        X = np.matrix(X).transpose()
        X = X * 1.0
        print(X.shape)

        mat_contents = h5py.File("".join((pathy, "/", filey[i])))
        Y = mat_contents['Y']
        Y = np.matrix(Y).transpose()
        Y = Y * 1.0

        # ----------Center normalize-------------

        # # MERGE
        # print('Merge Stack Data')
        # X_test = np.hstack((td, X))
        X_test = X

        # # Tranpose
        # X_test = X_test.transpose()
        # Y = Y.transpose()


        # SAVE

        NAME = "".join((filetd[i][3:-4],".h5"))
        print('Saving ' + NAME)
        try:
            f = h5py.File(NAME, 'w')
            # save = {
            #     'X_test': X_test,
            #     "Y_test" : Y
            # }
            f.create_dataset('X_test',data = X_test )
            f.create_dataset('Y_test', data=Y)
            f.close()
        except Exception as e:
            print('Unable to save data to ', NAME, ':', e)
            raise

        statinfo = os.stat(NAME)
        print('Compressed pickle size: ', statinfo.st_size)


print('FINISHED')