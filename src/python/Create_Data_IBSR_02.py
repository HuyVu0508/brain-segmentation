
## Create Data Train for IBSR Area 02

##  Include library
from __future__ import print_function
import numpy as np
import tensorflow as tf
import scipy.io as sio
import os
import h5py
from six.moves import cPickle as pickle


## Function which using

# Num features and num classes
num_classes = 2
num_ft = 363*3+3


def make_arrays(nb_rows, nb_cols):
  if nb_rows:
    dataset = np.ndarray((nb_rows, nb_cols), dtype=np.float32)
    labels = np.ndarray((nb_rows, num_classes), dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels




# Function merge dataset
def merge_datasets(X, Y, train_size, valid_size=0):
  valid_dataset, valid_labels = make_arrays(valid_size, num_ft)
  train_dataset, train_labels = make_arrays(train_size, num_ft)
  vsize_per_class = valid_size
  tsize_per_class = train_size

  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  try:
    np.random.shuffle(X)
    if valid_dataset is not None:
        valid_dataset[start_v:end_v, :] = X[:vsize_per_class, :]
        valid_labels[start_v:end_v, :] = Y[start_v:end_v, :]
        start_v += vsize_per_class
        end_v += vsize_per_class

    train_dataset[start_t:end_t, :] = X[vsize_per_class:end_l, :]
    train_labels[start_t:end_t, :] = Y[start_t:end_t, :]
  except Exception as e:
    print('Unable to process data from', X, ':', e)
    raise

  return valid_dataset, valid_labels, train_dataset, train_labels

# Function random after merging dataset

print('Random Again')

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:]
  shuffled_labels = labels[permutation, :]
  return shuffled_dataset, shuffled_labels

# Load, Process and Save Data Train pickle

path='/home/ubuntu/PycharmProjects/HuyCode/Train Pickles/';
os.chdir(path)



X_train_1_allpart = []
Y_train_1_allpart = []
t_train_1_allpart = []
X_train_0_allpart = []
Y_train_0_allpart = []
t_train_0_allpart = []

for i in range(12):
    j=i+1
    pickle_file = 'train_no_'+str(j)+'.h5'
    f1=h5py.File(pickle_file, 'r')
    # save = pickle.load(f1)
    save =  f1

    X_train_1_pi=(np.float16(save['X_train_1']))
    X_train_0_pi=(np.float16(save['X_train_0']))

    t_train_1_pi=X_train_1_pi.shape
    print(X_train_1_pi.shape)
    Y_train_1_pi=np.zeros((t_train_1_pi[0],2),dtype=np.int8)
    Y_train_1_pi[:,1]=np.ones(t_train_1_pi[0])

    t_train_0_pi=X_train_0_pi.shape
    Y_train_0_pi=np.zeros((t_train_0_pi[0],2),dtype=np.int8)
    Y_train_0_pi[:,0]=np.ones(t_train_0_pi[0])

    X_train_1_allpart.append(X_train_1_pi)
    Y_train_1_allpart.append(Y_train_1_pi)
    t_train_1_allpart.append(t_train_1_pi)
    X_train_0_allpart.append(X_train_0_pi)
    Y_train_0_allpart.append(Y_train_0_pi)
    t_train_0_allpart.append(t_train_0_pi)


    del save
    del X_train_1_pi, X_train_0_pi,  t_train_1_pi, t_train_0_pi, Y_train_1_pi, Y_train_0_pi
    print('Load data training part '+str(j)+' has finished')

X_train_1_allpart[0].shape



#
# #----------------------------------------------
#
# pickle_file_1 = 'train_0_1.pickle'
# f1=open(pickle_file_1, 'rb')
# save = pickle.load(f1)
#
# X_train_1_p1=save['X_train_1']
# X_train_0_p1=save['X_train_0']
#
# t_train_1_p1=X_train_1_p1.shape
# Y_train_1_p1=np.zeros((t_train_1_p1[0],2))
# Y_train_1_p1[:,1]=np.ones(t_train_1_p1[0])
#
# t_train_0_p1=X_train_0_p1.shape
# Y_train_0_p1=np.zeros((t_train_0_p1[0],2))
# Y_train_0_p1[:,0]=np.ones(t_train_0_p1[0])
#
# del save
# print('Load data training part 1 has finished')
#
#
#
#
#
#
#
# #----------------------------------------------
# pickle_file_1 = 'train_0_1.pickle'
# f1=open(pickle_file_1, 'rb')
# save = pickle.load(f1)
#
# X_train_1_p1=save['X_train_1']
# X_train_0_p1=save['X_train_0']
#
# t_train_1_p1=X_train_1_p1.shape
# Y_train_1_p1=np.zeros((t_train_1_p1[0],2))
# Y_train_1_p1[:,1]=np.ones(t_train_1_p1[0])
#
# t_train_0_p1=X_train_0_p1.shape
# Y_train_0_p1=np.zeros((t_train_0_p1[0],2))
# Y_train_0_p1[:,0]=np.ones(t_train_0_p1[0])
#
# del save
# print('Load data training part 1 has finished')
#
# #--------------------------------------------
#
# pickle_file_2 = 'train_0_5.pickle'
# f2=open(pickle_file_2, 'rb')
# save = pickle.load(f2)
#
# X_train_1_p2=save['X_train_1']
# X_train_0_p2=save['X_train_0']
#
# t_train_1_p2=X_train_1_p2.shape
# Y_train_1_p2=np.zeros((t_train_1_p2[0],2))
# Y_train_1_p2[:,1]=np.ones(t_train_1_p2[0])
#
# t_train_0_p2=X_train_0_p2.shape
# Y_train_0_p2=np.zeros((t_train_0_p2[0],2))
# Y_train_0_p2[:,0]=np.ones(t_train_0_p2[0])
#
# del save
# print('Load data training part 2 has finished')
#
# #------------------------------------------------
#
# pickle_file_3 = 'train_0_9.pickle'
# f3=open(pickle_file_3, 'rb')
# save = pickle.load(f3)
#
# X_train_1_p3=save['X_train_1']
# X_train_0_p3=save['X_train_0']
#
# t_train_1_p3=X_train_1_p3.shape
# Y_train_1_p3=np.zeros((t_train_1_p3[0],2))
# Y_train_1_p3[:,1]=np.ones(t_train_1_p3[0])
#
# t_train_0_p3=X_train_0_p3.shape
# Y_train_0_p3=np.zeros((t_train_0_p3[0],2))
# Y_train_0_p3[:,0]=np.ones(t_train_0_p3[0])
#
# del save
# print('Load data training part 3 has finished')

#-----------------------------------------------------
print('Finish load data training include: 1, 2, .., 12')


# X_train_1= np.vstack(X_train_1_allpart[:])
X_train_1= np.vstack(X_train_1_allpart[:])
X_train_1.shape
del X_train_1_allpart

# X_train_0= np.vstack(X_train_0_allpart)
X_train_0= np.vstack(X_train_0_allpart[:])
print(X_train_0.shape)
del X_train_0_allpart

# Y_train_1= np.vstack(Y_train_1_allpart)
Y_train_1= np.vstack(Y_train_1_allpart[:])
Y_train_1.shape
del Y_train_1_allpart

# Y_train_0= np.vstack(Y_train_0_allpart)
Y_train_0= np.vstack(Y_train_0_allpart[:])
Y_train_0.shape
del Y_train_0_allpart

print("Finish merge data training")

#--------------------------------------------

valid_size_1 = 20000

valid_size_0 = 30000

train_size_1 = X_train_1.shape[0] - valid_size_1
train_size_0 = X_train_0.shape[0] - valid_size_0


valid_dataset0, valid_labels0, train_dataset0, train_labels0 = merge_datasets(
  X_train_0, Y_train_0, train_size_0, valid_size_0)

del X_train_0, Y_train_0

valid_dataset1, valid_labels1, train_dataset1, train_labels1 = merge_datasets(
  X_train_1, Y_train_1, train_size_1, valid_size_1)

del X_train_1, Y_train_1

train_dataset0 = np.float16(train_dataset0)
train_dataset1 = np.float16(train_dataset1)
valid_dataset0 = np.float16(valid_dataset0)
valid_dataset1 = np.float16(valid_dataset1)
train_labels0 = np.int16(train_labels0)
train_labels1 = np.int16(train_labels1)

train_dataset = np.vstack((train_dataset0, train_dataset1))
train_labels = np.vstack((train_labels0, train_labels1))
valid_dataset = np.vstack((valid_dataset0, valid_dataset1))
valid_labels = np.vstack((valid_labels0, valid_labels1))

del train_dataset0, train_dataset1, valid_dataset1, valid_dataset0

train_dataset, train_labels = randomize(train_dataset, train_labels)
train_dataset.shape
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
valid_dataset.shape
print('Saving')
h5_file = 'Train_Validation_Dataset_all_16.h5'
with h5py.File(h5_file, 'w') as hf:
    try:
        hf.create_dataset('train_dataset', data=train_dataset)
        hf.create_dataset('train_labels', data=train_labels)
        hf.create_dataset('valid_dataset', data=valid_dataset)
        hf.create_dataset('valid_labels', data=valid_labels)
    except Exception as e:
        print('Unable to save data to', h5_file, ':', e)
        raise

    statinfo = os.stat(h5_file)
    print('Compressed pickle size:', statinfo.st_size)

del train_dataset, train_labels, valid_labels, valid_dataset

