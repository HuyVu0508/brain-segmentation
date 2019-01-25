from __future__ import print_function
import numpy as np
import tensorflow as tf
import scipy.io as sio
import os
from six.moves import cPickle as pickle
import h5py



# LBPA SIZE
# c = np.array([255. / 2, 123. / 2])
# b = np.array([255., 123.])

# OASIS SIZE
# c = np.array([175. / 2, 207. / 2])
# b = np.array([175., 207.])

# CHANGE DIR
pathtd0 = "/home/ubuntu/Desktop/Features/Train/0/td"
pathx0 = "/home/ubuntu/Desktop/Features/Train/0/X"
pathy0 = "/home/ubuntu/Desktop/Features/Train/0/Y"
pathtd1 = "/home/ubuntu/Desktop/Features/Train/1/td"
pathx1 = "/home/ubuntu/Desktop/Features/Train/1/X"
pathy1 = "/home/ubuntu/Desktop/Features/Train/1/Y"

filetd0 = [f for f in os.listdir(pathtd0) if f.endswith(".mat")]
filetd1 = [f for f in os.listdir(pathtd1) if f.endswith(".mat")]
filex0 = [f for f in os.listdir(pathx0) if f.endswith(".mat")]
filex1 = [f for f in os.listdir(pathx1) if f.endswith(".mat")]
filey0 = [f for f in os.listdir(pathy0) if f.endswith(".mat")]
filey1 = [f for f in os.listdir(pathy1) if f.endswith(".mat")]

filetd0.sort()
filetd1.sort()
filex0.sort()
filex1.sort()
filey0.sort()
filey1.sort()

# READ MATLAB
for i in range(len(filetd0)):
    print('Process File ' + filetd0[i])

    # mat_contents = sio.loadmat("".join((pathtd0,"/",filetd0[i])))
    # tdtr0 = mat_contents['tdtr0']
    mat_contents = h5py.File("".join((pathtd0,"/",filetd0[i])),"r")
    tdtr0 = mat_contents['tdtr0']
    tdtr0 = np.matrix(tdtr0)
    print(tdtr0.shape)
    tdtr0 = tdtr0 * 1.0

    mat_contents = h5py.File("".join((pathtd1, "/", filetd1[i])))
    tdtr1 = mat_contents['tdtr1']
    tdtr1 = np.matrix(tdtr1)
    print(tdtr1.shape)
    tdtr1 = tdtr1 * 1.0

    # mat_contents = sio.loadmat("".join((pathx0, "/", filex0[i])))
    mat_contents = h5py.File("".join((pathx0, "/", filex0[i])))
    Xtr0 = mat_contents.get('Xtr0')
    Xtr0 = np.matrix(Xtr0)
    print(Xtr0.shape)
    Xtr0 = Xtr0 * 1.0

    mat_contents = h5py.File("".join((pathx1, "/", filex1[i])))
    Xtr1 = mat_contents['Xtr1']
    Xtr1 = np.matrix(Xtr1)
    print(Xtr1.shape)
    Xtr1 = Xtr1 * 1.0

    # mat_contents = sio.loadmat("".join((pathy0, "/", filey0[i])))
    mat_contents = h5py.File("".join((pathy0, "/", filey0[i])), "r")
    Ytr0 = mat_contents['Ytr0']
    Ytr0 = np.matrix(Ytr0)
    print(Ytr0.shape)
    Ytr0 = Ytr0 * 1.0
    #
    mat_contents = h5py.File("".join((pathy1, "/", filey1[i])))
    Ytr1 = mat_contents['Ytr1']
    Ytr1 = np.matrix(Ytr1)
    print(Ytr1.shape)
    Ytr1 = Ytr1 * 1.0

    # ----------Center normalize-------------
    # print('Center normalize')
    # tdtr0 = (tdtr0 - c) / b
    # tdtr1 = (tdtr1 - c) / b
    # Xtr0 = Xtr0 - 0.5
    # Xtr1 = Xtr1 - 0.5
    # print('Done Center normalize')

    # MERGE
    print('Merge Stack Data')
    # X_train_0 = np.hstack((tdtr0, Xtr0))
    # X_train_1 = np.hstack((tdtr1, Xtr1))
    X_train_0 = Xtr0.transpose()
    X_train_1 = Xtr1.transpose()

    # SAVE

    # NAME = "Train Pickles/".join((filetd0[i][3:-4], ".h5"))

    NAME = "Train Pickles/" + filetd0[i][3:-4] + ".h5"

    print('Saving ' + NAME)
    try:
        # f = open(NAME, 'wb')
        # save = {
        #     'X_train_0': X_train_0,
        #     'X_train_1': X_train_1
        # }
        # pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        # f.close()

        f = h5py.File(NAME,"w")
        f.create_dataset('X_train_0',data=X_train_0)
        f.create_dataset('X_train_1', data=X_train_1)
        f.close()

    except Exception as e:
        print('Unable to save data to ', NAME, ':', e)
        raise

    statinfo = os.stat(NAME)
    print('Compressed pickle size: ', statinfo.st_size)







    ####
    # NAME = "".join((filetd0[i][3:-4], ".pickle"))
    # print('Saving ' + NAME)
    # try:
    #     import tables
    #     f = tables.openFile("test.h5",mode="w")
    #     root = f.root
    #     save = {
    #         'X_train_0': X_train_0,
    #         'X_train_1': X_train_1
    #     }
    #     f.createDataset(root,"test",save)
    #     f.close()
    # except Exception as e:
    #     print('Unable to save data to ', NAME, ':', e)
    #     raise
    #
    # statinfo = os.stat(NAME)
    # print('Compressed pickle size: ', statinfo.st_size)
    # ######


print('FINISHED')