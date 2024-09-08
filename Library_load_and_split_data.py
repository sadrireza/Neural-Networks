import numpy as np
import scipy.io
import pickle
from sklearn.model_selection import KFold
import itertools
from tensorflow import keras
def data_generator_dist(indices, epochs, batch_size,steps):
    for _ in range(epochs):
        np.random.shuffle(indices)
        zipped = itertools.cycle(indices)
        for _ in range(steps):
            X = []
            Y = []
            for _ in range(batch_size):
                index= next(zipped)

                x = np.load("../dataset/Z24/ForIterator/X_" + str(index).zfill(7)+".npy")

                coeff = 512
                tw = 64
                nsens = 8

                y = np.load("../dataset/Z24/ForIterator/Y_" + str(index).zfill(7)+".npy")
                y = keras.utils.to_categorical((y + 1) / 2, num_classes=2)

                X.append(x)
                Y.append([y])

            X = np.array(X)
            X_3d = -777777 * np.ones((X.shape[0], nsens, coeff, tw))
            for i in range(0, X.shape[0]):
                for j in range(0, nsens):
                    X_3d[i, j, :, :] = np.swapaxes(np.reshape(X[i, j, :tw * coeff], [1, 1, tw, coeff]), 2, 3)
            X_3d = np.moveaxis(X_3d, [1], [3])

            X = [np.expand_dims(X_3d[:,:, :, 0], axis=-1), np.expand_dims(X_3d[:,:, :, 1], axis=-1),
                 np.expand_dims(X_3d[:,:, :, 2], axis=-1), np.expand_dims(X_3d[:,:, :, 3], axis=-1),
                 np.expand_dims(X_3d[:,:, :, 4], axis=-1), np.expand_dims(X_3d[:,:, :, 5], axis=-1),
                 np.expand_dims(X_3d[:,:, :, 6], axis=-1), np.expand_dims(X_3d[:,:, :, 7], axis=-1)]

            Y = [np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),
                 np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y))]

            yield (X,Y)

def data_generator_dist_LMS6DSOX(indices, epochs, batch_size,steps):
    for _ in range(epochs):
        np.random.shuffle(indices)
        zipped = itertools.cycle(indices)
        for _ in range(steps):
            X = []
            Y = []
            for _ in range(batch_size):
                index= next(zipped)

                x = np.load("../dataset/Z24/ForIterator_LMS6DSOX/X_" + str(index).zfill(7)+".npy")

                coeff = 512
                tw = 64
                nsens = 8

                y = np.load("../dataset/Z24/ForIterator_LMS6DSOX/Y_" + str(index).zfill(7)+".npy")
                y = keras.utils.to_categorical((y + 1) / 2, num_classes=2)

                X.append(x)
                Y.append([y])

            X = np.array(X)
            X_3d = -777777 * np.ones((X.shape[0], nsens, coeff, tw))
            for i in range(0, X.shape[0]):
                for j in range(0, nsens):
                    X_3d[i, j, :, :] = np.swapaxes(np.reshape(X[i, j, :tw * coeff], [1, 1, tw, coeff]), 2, 3)
            X_3d = np.moveaxis(X_3d, [1], [3])

            X = [np.expand_dims(X_3d[:,:, :, 0], axis=-1), np.expand_dims(X_3d[:,:, :, 1], axis=-1),
                 np.expand_dims(X_3d[:,:, :, 2], axis=-1), np.expand_dims(X_3d[:,:, :, 3], axis=-1),
                 np.expand_dims(X_3d[:,:, :, 4], axis=-1), np.expand_dims(X_3d[:,:, :, 5], axis=-1),
                 np.expand_dims(X_3d[:,:, :, 6], axis=-1), np.expand_dims(X_3d[:,:, :, 7], axis=-1)]

            Y = [np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),
                 np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y))]

            yield (X,Y)
def data_generator_dist_ADXL355(indices, epochs, batch_size,steps):
    for _ in range(epochs):
        np.random.shuffle(indices)
        zipped = itertools.cycle(indices)
        for _ in range(steps):
            X = []
            Y = []
            for _ in range(batch_size):
                index= next(zipped)

                x = np.load("../dataset/Z24/ForIterator_ADXL355/X_" + str(index).zfill(7)+".npy")

                coeff = 512
                tw = 64
                nsens = 8

                y = np.load("../dataset/Z24/ForIterator_ADXL355/Y_" + str(index).zfill(7)+".npy")
                y = keras.utils.to_categorical((y + 1) / 2, num_classes=2)

                X.append(x)
                Y.append([y])

            X = np.array(X)
            X_3d = -777777 * np.ones((X.shape[0], nsens, coeff, tw))
            for i in range(0, X.shape[0]):
                for j in range(0, nsens):
                    X_3d[i, j, :, :] = np.swapaxes(np.reshape(X[i, j, :tw * coeff], [1, 1, tw, coeff]), 2, 3)
            X_3d = np.moveaxis(X_3d, [1], [3])

            X = [np.expand_dims(X_3d[:,:, :, 0], axis=-1), np.expand_dims(X_3d[:,:, :, 1], axis=-1),
                 np.expand_dims(X_3d[:,:, :, 2], axis=-1), np.expand_dims(X_3d[:,:, :, 3], axis=-1),
                 np.expand_dims(X_3d[:,:, :, 4], axis=-1), np.expand_dims(X_3d[:,:, :, 5], axis=-1),
                 np.expand_dims(X_3d[:,:, :, 6], axis=-1), np.expand_dims(X_3d[:,:, :, 7], axis=-1)]

            Y = [np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),
                 np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y))]

            yield (X,Y)

def data_generator_dist_30dB(indices, epochs, batch_size,steps):
    for _ in range(epochs):
        np.random.shuffle(indices)
        zipped = itertools.cycle(indices)
        for _ in range(steps):
            X = []
            Y = []
            for _ in range(batch_size):
                index= next(zipped)

                x = np.load("../dataset/Z24/ForIterator_30dB/X_" + str(index).zfill(7)+".npy")

                coeff = 512
                tw = 64
                nsens = 8

                y = np.load("../dataset/Z24/ForIterator_30dB/Y_" + str(index).zfill(7)+".npy")
                y = keras.utils.to_categorical((y + 1) / 2, num_classes=2)

                X.append(x)
                Y.append([y])

            X = np.array(X)
            X_3d = -777777 * np.ones((X.shape[0], nsens, coeff, tw))
            for i in range(0, X.shape[0]):
                for j in range(0, nsens):
                    X_3d[i, j, :, :] = np.swapaxes(np.reshape(X[i, j, :tw * coeff], [1, 1, tw, coeff]), 2, 3)
            X_3d = np.moveaxis(X_3d, [1], [3])

            X = [np.expand_dims(X_3d[:,:, :, 0], axis=-1), np.expand_dims(X_3d[:,:, :, 1], axis=-1),
                 np.expand_dims(X_3d[:,:, :, 2], axis=-1), np.expand_dims(X_3d[:,:, :, 3], axis=-1),
                 np.expand_dims(X_3d[:,:, :, 4], axis=-1), np.expand_dims(X_3d[:,:, :, 5], axis=-1),
                 np.expand_dims(X_3d[:,:, :, 6], axis=-1), np.expand_dims(X_3d[:,:, :, 7], axis=-1)]

            Y = [np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),
                 np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y))]

            yield (X,Y)
def data_generator_dist_20dB(indices, epochs, batch_size,steps):
    for _ in range(epochs):
        np.random.shuffle(indices)
        zipped = itertools.cycle(indices)
        for _ in range(steps):
            X = []
            Y = []
            for _ in range(batch_size):
                index= next(zipped)

                x = np.load("../dataset/Z24/ForIterator_20dB/X_" + str(index).zfill(7)+".npy")

                coeff = 512
                tw = 64
                nsens = 8

                y = np.load("../dataset/Z24/ForIterator_20dB/Y_" + str(index).zfill(7)+".npy")
                y = keras.utils.to_categorical((y + 1) / 2, num_classes=2)

                X.append(x)
                Y.append([y])

            X = np.array(X)
            X_3d = -777777 * np.ones((X.shape[0], nsens, coeff, tw))
            for i in range(0, X.shape[0]):
                for j in range(0, nsens):
                    X_3d[i, j, :, :] = np.swapaxes(np.reshape(X[i, j, :tw * coeff], [1, 1, tw, coeff]), 2, 3)
            X_3d = np.moveaxis(X_3d, [1], [3])

            X = [np.expand_dims(X_3d[:,:, :, 0], axis=-1), np.expand_dims(X_3d[:,:, :, 1], axis=-1),
                 np.expand_dims(X_3d[:,:, :, 2], axis=-1), np.expand_dims(X_3d[:,:, :, 3], axis=-1),
                 np.expand_dims(X_3d[:,:, :, 4], axis=-1), np.expand_dims(X_3d[:,:, :, 5], axis=-1),
                 np.expand_dims(X_3d[:,:, :, 6], axis=-1), np.expand_dims(X_3d[:,:, :, 7], axis=-1)]

            Y = [np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),
                 np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y))]

            yield (X,Y)

def data_generator_dist_10dB(indices, epochs, batch_size,steps):
    for _ in range(epochs):
        np.random.shuffle(indices)
        zipped = itertools.cycle(indices)
        for _ in range(steps):
            X = []
            Y = []
            for _ in range(batch_size):
                index= next(zipped)

                x = np.load("../dataset/Z24/ForIterator_10dB/X_" + str(index).zfill(7)+".npy")

                coeff = 512
                tw = 64
                nsens = 8

                y = np.load("../dataset/Z24/ForIterator_10dB/Y_" + str(index).zfill(7)+".npy")
                y = keras.utils.to_categorical((y + 1) / 2, num_classes=2)

                X.append(x)
                Y.append([y])

            X = np.array(X)
            X_3d = -777777 * np.ones((X.shape[0], nsens, coeff, tw))
            for i in range(0, X.shape[0]):
                for j in range(0, nsens):
                    X_3d[i, j, :, :] = np.swapaxes(np.reshape(X[i, j, :tw * coeff], [1, 1, tw, coeff]), 2, 3)
            X_3d = np.moveaxis(X_3d, [1], [3])

            X = [np.expand_dims(X_3d[:,:, :, 0], axis=-1), np.expand_dims(X_3d[:,:, :, 1], axis=-1),
                 np.expand_dims(X_3d[:,:, :, 2], axis=-1), np.expand_dims(X_3d[:,:, :, 3], axis=-1),
                 np.expand_dims(X_3d[:,:, :, 4], axis=-1), np.expand_dims(X_3d[:,:, :, 5], axis=-1),
                 np.expand_dims(X_3d[:,:, :, 6], axis=-1), np.expand_dims(X_3d[:,:, :, 7], axis=-1)]

            Y = [np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),
                 np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y)),np.squeeze(np.array(Y))]

            yield (X,Y)

def load_DUMMY_uncompressed_DEBUG():
    n_data =  2000
    window = 512
    tw = 64
    nsens = 8
    I_first_damage = 49


    Xnp = np.random.rand(n_data, nsens, tw*window)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    return X,Y


def load_fresh_data_Z24_W512_uncompressed():
    I_first_damage = 4923

    files_list =["Z4_EMS01.1_CS_data_CR_1.mat","Z4_EMS01.2_CS_data_CR_1.mat",
                 "Z4_EMS02.1_CS_data_CR_1.mat","Z4_EMS02.2_CS_data_CR_1.mat",
                 "Z4_EMS03.1_CS_data_CR_1.mat","Z4_EMS03.2_CS_data_CR_1.mat","Z4_EMS03.3_CS_data_CR_1.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/Z24/CR1_W512_NoiseFree/"+files_list[i])
        X_list.append(mat["data"])

    Xnp = np.concatenate(X_list,axis =0)
    X_list = []
    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    return X,Y

def load_fresh_data_Z24_W512_uncompressed_LMS6DSOX():
    I_first_damage = 4923

    files_list =["Z4_EMS01.1_CS_data_CR_1_LMS6DSOX.mat","Z4_EMS01.2_CS_data_CR_1_LMS6DSOX.mat",
                 "Z4_EMS02.1_CS_data_CR_1_LMS6DSOX.mat","Z4_EMS02.2_CS_data_CR_1_LMS6DSOX.mat",
                 "Z4_EMS03.1_CS_data_CR_1_LMS6DSOX.mat","Z4_EMS03.2_CS_data_CR_1_LMS6DSOX.mat","Z4_EMS03.3_CS_data_CR_1_LMS6DSOX.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/Z24/CR1_W512_LMS6DSOX/"+files_list[i])
        X_list.append(mat["data"])

    Xnp = np.concatenate(X_list,axis =0)
    X_list = []
    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    return X,Y

def load_fresh_data_Z24_W512_uncompressed_30dB():
    I_first_damage = 4923

    files_list =["Z4_EMS01.1_CS_data_CR_1_30dB.mat","Z4_EMS01.2_CS_data_CR_1_30dB.mat",
                 "Z4_EMS02.1_CS_data_CR_1_30dB.mat","Z4_EMS02.2_CS_data_CR_1_30dB.mat",
                 "Z4_EMS03.1_CS_data_CR_1_30dB.mat","Z4_EMS03.2_CS_data_CR_1_30dB.mat","Z4_EMS03.3_CS_data_CR_1_30dB.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/Z24/CR1_W512_30dB/"+files_list[i])
        X_list.append(mat["data"])

    Xnp = np.concatenate(X_list,axis =0)
    X_list = []
    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    return X,Y

def load_fresh_data_Z24_W512_uncompressed_20dB():
    I_first_damage = 4923

    files_list =["Z4_EMS01.1_CS_data_CR_1_20dB.mat","Z4_EMS01.2_CS_data_CR_1_20dB.mat",
                 "Z4_EMS02.1_CS_data_CR_1_20dB.mat","Z4_EMS02.2_CS_data_CR_1_20dB.mat",
                 "Z4_EMS03.1_CS_data_CR_1_20dB.mat","Z4_EMS03.2_CS_data_CR_1_20dB.mat","Z4_EMS03.3_CS_data_CR_1_20dB.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/Z24/CR1_W512_20dB/"+files_list[i])
        X_list.append(mat["data"])

    Xnp = np.concatenate(X_list,axis =0)
    X_list = []
    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    return X,Y

def load_fresh_data_Z24_W512_uncompressed_10dB():
    I_first_damage = 4923

    files_list =["Z4_EMS01.1_CS_data_CR_1_10dB.mat","Z4_EMS01.2_CS_data_CR_1_10dB.mat",
                 "Z4_EMS02.1_CS_data_CR_1_10dB.mat","Z4_EMS02.2_CS_data_CR_1_10dB.mat",
                 "Z4_EMS03.1_CS_data_CR_1_10dB.mat","Z4_EMS03.2_CS_data_CR_1_10dB.mat","Z4_EMS03.3_CS_data_CR_1_10dB.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/Z24/CR1_W512_10dB/"+files_list[i])
        X_list.append(mat["data"])

    Xnp = np.concatenate(X_list,axis =0)
    X_list = []
    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    return X,Y

def load_fresh_data_Z24_W512_uncompressed_ADXL355():
    I_first_damage = 4923

    files_list =["Z4_EMS01.1_CS_data_CR_1_ADXL355.mat","Z4_EMS01.2_CS_data_CR_1_ADXL355.mat",
                 "Z4_EMS02.1_CS_data_CR_1_ADXL355.mat","Z4_EMS02.2_CS_data_CR_1_ADXL355.mat",
                 "Z4_EMS03.1_CS_data_CR_1_ADXL355.mat","Z4_EMS03.2_CS_data_CR_1_ADXL355.mat","Z4_EMS03.3_CS_data_CR_1_ADXL355.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/Z24/CR1_W512_ADXL355/"+files_list[i])
        X_list.append(mat["data"])

    Xnp = np.concatenate(X_list,axis =0)
    X_list = []
    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    return X,Y

def load_fresh_data_half_time_Z24_W512_uncompressed():
    I_first_damage = 4923

    files_list =["Z4_EMS01.1_CS_data_CR_1.mat","Z4_EMS01.2_CS_data_CR_1.mat",
                 "Z4_EMS02.1_CS_data_CR_1.mat","Z4_EMS02.2_CS_data_CR_1.mat",
                 "Z4_EMS03.1_CS_data_CR_1.mat","Z4_EMS03.2_CS_data_CR_1.mat","Z4_EMS03.3_CS_data_CR_1.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/Z24/CR1_W512_NoiseFree/"+files_list[i])
        X_list.append(mat["data"][:,:,:16384])

    Xnp = np.concatenate(X_list,axis =0)
    X_list = []
    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    return X,Y

def load_fresh_data_half_freq_Z24_W512_uncompressed():
    I_first_damage = 4923

    num =round(32768*30/50)

    files_list =["Z4_EMS01.1_CS_data_CR_1.mat","Z4_EMS01.2_CS_data_CR_1.mat",
                 "Z4_EMS02.1_CS_data_CR_1.mat","Z4_EMS02.2_CS_data_CR_1.mat",
                 "Z4_EMS03.1_CS_data_CR_1.mat","Z4_EMS03.2_CS_data_CR_1.mat","Z4_EMS03.3_CS_data_CR_1.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/Z24/CR1_W512_NoiseFree/"+files_list[i])
        X_list.append(scipy.signal.resample(mat["data"],num,axis=2))

    Xnp = np.concatenate(X_list,axis =0)
    X_list = []
    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    return X,Y

def load_fresh_data_Z24_W512_CS(CR = 8):
    I_first_damage = 4923

    CR_S = str(CR)

    files_list =["Z4_EMS01.1_CS_data_CR_"+CR_S+".mat","Z4_EMS01.2_CS_data_CR_"+CR_S+".mat",
                 "Z4_EMS02.1_CS_data_CR_"+CR_S+".mat","Z4_EMS02.2_CS_data_CR_"+CR_S+".mat",
                 "Z4_EMS03.1_CS_data_CR_"+CR_S+".mat","Z4_EMS03.2_CS_data_CR_"+CR_S+".mat","Z4_EMS03.3_CS_data_CR_"+CR_S+".mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/Z24/CR"+CR_S+"_W512_NoiseFree/"+files_list[i])
        X_list.append(mat["data"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/Z24/preloadedData'+CR_S+'_W512_NoiseFalse.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f, protocol=4)

    return X,Y

def load_fresh_data_Z24_W512_randComp(CR = 8):
    I_first_damage = 4923

    CR_S = str(CR)
    files_list =["Z4_EMS01.1_CS_data_CR_"+CR_S+".mat","Z4_EMS01.2_CS_data_CR_"+CR_S+".mat",
                 "Z4_EMS02.1_CS_data_CR_"+CR_S+".mat","Z4_EMS02.2_CS_data_CR_"+CR_S+".mat",
                 "Z4_EMS03.1_CS_data_CR_"+CR_S+".mat","Z4_EMS03.2_CS_data_CR_"+CR_S+".mat","Z4_EMS03.3_CS_data_CR_"+CR_S+".mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/Z24/CR"+CR_S+"_W512_NoiseFree/"+files_list[i])
        X_list.append(mat["data_cr"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/Z24/preloadedData_RNDCR'+CR_S+'_W512_NoiseFalse.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y

def load_fresh_KW51_W512_NoiseFree(CR = 8):
    I_first_damage = 1239

    files_list =["KW51_CS_data_CR"+str(CR)+"_W512_NoiseFree.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/KW51/"+files_list[i])
        X_list.append(mat["data_crMRak"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/KW51/preloadedData_CR'+str(CR)+'_W512_NoiseFalse.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y

def load_fresh_KW51_W512_randComp(CR = 8):
    I_first_damage = 1239

    files_list =["KW51_CS_data_CR"+str(CR)+"_W512_NoiseFree.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/KW51/"+files_list[i])
        X_list.append(mat["data_crRND"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/KW51/preloadedData_RNDCR'+str(CR)+'_W512_NoiseFalse.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y

def load_fresh_KW51_CR4_W512_NoiseFree():
    I_first_damage = 1239

    files_list =["KW51_CS_data_CR4_W512_NoiseFree.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/KW51/"+files_list[i])
        X_list.append(mat["data_crMRak"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/KW51/preloadedData_CR4_W512_NoiseFalse.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y

def load_fresh_data_CR256_W512_NoiseFree():
    I_first_damage = 4923
    files_list =["Z4_EMS01.1_CS_data_CR_256.mat","Z4_EMS01.2_CS_data_CR_256.mat",
                 "Z4_EMS02.1_CS_data_CR_256.mat","Z4_EMS02.2_CS_data_CR_256.mat",
                 "Z4_EMS03.1_CS_data_CR_256.mat","Z4_EMS03.2_CS_data_CR_256.mat","Z4_EMS03.3_CS_data_CR_256.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/CR256_W512_NoiseFree/"+files_list[i])
        X_list.append(mat["data_cr"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/preloadedData_CR256_W512_NoiseFalse.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y

def load_fresh_data_CR128_W512_NoiseFree():
    I_first_damage = 4923
    files_list =["Z4_EMS01.1_CS_data_CR_128.mat","Z4_EMS01.2_CS_data_CR_128.mat",
                 "Z4_EMS02.1_CS_data_CR_128.mat","Z4_EMS02.2_CS_data_CR_128.mat",
                 "Z4_EMS03.1_CS_data_CR_128.mat","Z4_EMS03.2_CS_data_CR_128.mat","Z4_EMS03.3_CS_data_CR_128.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/CR128_W512_NoiseFree/"+files_list[i])
        X_list.append(mat["data_cr"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/preloadedData_CR128_W512_NoiseFalse.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y

def load_fresh_data_CR64_W512_NoiseFree():
    I_first_damage = 4923
    files_list =["Z4_EMS01.1_CS_data_CR_64.mat","Z4_EMS01.2_CS_data_CR_64.mat",
                 "Z4_EMS02.1_CS_data_CR_64.mat","Z4_EMS02.2_CS_data_CR_64.mat",
                 "Z4_EMS03.1_CS_data_CR_64.mat","Z4_EMS03.2_CS_data_CR_64.mat","Z4_EMS03.3_CS_data_CR_64.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/CR64_W512_NoiseFree/"+files_list[i])
        X_list.append(mat["data_cr"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/preloadedData_CR64_W512_NoiseFalse.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y

def load_fresh_data_CR32_W512_NoiseFree():
    I_first_damage = 4923
    files_list =["Z4_EMS01.1_CS_data_CR_32.mat","Z4_EMS01.2_CS_data_CR_32.mat",
                 "Z4_EMS02.1_CS_data_CR_32.mat","Z4_EMS02.2_CS_data_CR_32.mat",
                 "Z4_EMS03.1_CS_data_CR_32.mat","Z4_EMS03.2_CS_data_CR_32.mat","Z4_EMS03.3_CS_data_CR_32.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/CR32_W512_NoiseFree/"+files_list[i])
        X_list.append(mat["data_cr"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/preloadedData_CR32_W512_NoiseFalse.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y

def load_fresh_data_CR16_W512_NoiseFree():
    I_first_damage = 4923
    files_list =["Z4_EMS01.1_CS_data_CR_16.mat","Z4_EMS01.2_CS_data_CR_16.mat",
                 "Z4_EMS02.1_CS_data_CR_16.mat","Z4_EMS02.2_CS_data_CR_16.mat",
                 "Z4_EMS03.1_CS_data_CR_16.mat","Z4_EMS03.2_CS_data_CR_16.mat","Z4_EMS03.3_CS_data_CR_16.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/CR16_W512_NoiseFree/"+files_list[i])
        X_list.append(mat["data_cr"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/preloadedData_CR16_W512_NoiseFalse.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y


def load_fresh_data_CR8_W512_30dB():
    I_first_damage = 4923
    files_list =["Z4_EMS01.1_CS_data_CR_8_30dB.mat","Z4_EMS01.2_CS_data_CR_8_30dB.mat",
                 "Z4_EMS02.1_CS_data_CR_8_30dB.mat","Z4_EMS02.2_CS_data_CR_8_30dB.mat",
                 "Z4_EMS03.1_CS_data_CR_8_30dB.mat","Z4_EMS03.2_CS_data_CR_8_30dB.mat","Z4_EMS03.3_CS_data_CR_8_30dB.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/CR8_W512_30dB/"+files_list[i])
        X_list.append(mat["data_cr"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/preloadedData_CR8_W512_NoiseTrue_NL30.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y

def load_fresh_data_CR8_W512_20dB():
    I_first_damage = 4923
    files_list =["Z4_EMS01.1_CS_data_CR_8_20dB.mat","Z4_EMS01.2_CS_data_CR_8_20dB.mat",
                 "Z4_EMS02.1_CS_data_CR_8_20dB.mat","Z4_EMS02.2_CS_data_CR_8_20dB.mat",
                 "Z4_EMS03.1_CS_data_CR_8_20dB.mat","Z4_EMS03.2_CS_data_CR_8_20dB.mat","Z4_EMS03.3_CS_data_CR_8_20dB.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/CR8_W512_20dB/"+files_list[i])
        X_list.append(mat["data_cr"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/preloadedData_CR8_W512_NoiseTrue_NL20.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y

def load_fresh_data_CR8_W512_10dB():
    I_first_damage = 4923
    files_list =["Z4_EMS01.1_CS_data_CR_8_10dB.mat","Z4_EMS01.2_CS_data_CR_8_10dB.mat",
                 "Z4_EMS02.1_CS_data_CR_8_10dB.mat","Z4_EMS02.2_CS_data_CR_8_10dB.mat",
                 "Z4_EMS03.1_CS_data_CR_8_10dB.mat","Z4_EMS03.2_CS_data_CR_8_10dB.mat","Z4_EMS03.3_CS_data_CR_8_10dB.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/CR8_W512_10dB/"+files_list[i])
        X_list.append(mat["data_cr"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/preloadedData_CR8_W512_NoiseTrue_NL10.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y

def load_fresh_data_CR4_W512_NoiseFree():
    I_first_damage = 4923

    files_list =["Z4_EMS01.1_CS_data_CR4_W512_NoiseFree.mat","Z4_EMS01.2_CS_data_CR4_W512_NoiseFree.mat",
                 "Z4_EMS02.1_CS_data_CR4_W512_NoiseFree.mat","Z4_EMS02.2_CS_data_CR4_W512_NoiseFree.mat",
                 "Z4_EMS03.1_CS_data_CR4_W512_NoiseFree.mat","Z4_EMS03.2_CS_data_CR4_W512_NoiseFree.mat","Z4_EMS03.3_CS_data_CR4_W512_NoiseFree.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/CR4_W512_NoiseFree_New/"+files_list[i])
        X_list.append(mat["data_cr"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/preloadedData_CR4_W512_NoiseFalse.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y

def load_fresh_data_CR4_W512_NoiseLSM5DSL():
    I_first_damage = 4923

    files_list =["Z4_EMS01.1_CS_data_CR4_LSM6DSL.mat","Z4_EMS01.2_CS_data_CR4_LSM6DSL.mat",
                 "Z4_EMS02.1_CS_data_CR4_LSM6DSL.mat","Z4_EMS02.2_CS_data_CR4_LSM6DSL.mat",
                 "Z4_EMS03.1_CS_data_CR4_LSM6DSL.mat","Z4_EMS03.2_CS_data_CR4_LSM6DSL.mat","Z4_EMS03.3_CS_data_CR4_LSM6DSL.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/CR4_W512_LSM5DSL/"+files_list[i])
        X_list.append(mat["data_cr"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/preloadedData_CR4_W512_NoiseTrue.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y

def load_fresh_data_CR6_W512_NoiseFree():
    I_first_damage = 4923

    files_list =["Z4_EMS01.1_CS_data_CR6_W512_NoiseFree.mat","Z4_EMS01.2_CS_data_CR6_W512_NoiseFree.mat",
                 "Z4_EMS02.1_CS_data_CR6_W512_NoiseFree.mat","Z4_EMS02.2_CS_data_CR6_W512_NoiseFree.mat",
                 "Z4_EMS03.1_CS_data_CR6_W512_NoiseFree.mat","Z4_EMS03.2_CS_data_CR6_W512_NoiseFree.mat","Z4_EMS03.3_CS_data_CR6_W512_NoiseFree.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/CR6_W512_NoiseFree_New/"+files_list[i])
        X_list.append(mat["data_cr"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/preloadedData_CR6_W512_NoiseFalse.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y

def load_fresh_data_CR8_W512_NoiseFree():
    I_first_damage = 4923

    files_list =["Z4_EMS01.1_CS_data_CR8_W512_NoiseFree.mat","Z4_EMS01.2_CS_data_CR8_W512_NoiseFree.mat",
                 "Z4_EMS02.1_CS_data_CR8_W512_NoiseFree.mat","Z4_EMS02.2_CS_data_CR8_W512_NoiseFree.mat",
                 "Z4_EMS03.1_CS_data_CR8_W512_NoiseFree.mat","Z4_EMS03.2_CS_data_CR8_W512_NoiseFree.mat","Z4_EMS03.3_CS_data_CR8_W512_NoiseFree.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/CR8_W512_NoiseFree_New/"+files_list[i])
        X_list.append(mat["data_cr"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/preloadedData_CR8_W512_NoiseFalse.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y

def load_fresh_data_CR8_W512_NoiseLSM5DSL():
    I_first_damage = 4923

    files_list =["Z4_EMS01.1_CS_data_CR8_LSM6DSL.mat","Z4_EMS01.2_CS_data_CR8_LSM6DSL.mat",
                 "Z4_EMS02.1_CS_data_CR8_LSM6DSL.mat","Z4_EMS02.2_CS_data_CR8_LSM6DSL.mat",
                 "Z4_EMS03.1_CS_data_CR8_LSM6DSL.mat","Z4_EMS03.2_CS_data_CR8_LSM6DSL.mat","Z4_EMS03.3_CS_data_CR8_LSM6DSL.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/CR8_W512_LSM6DSL/"+files_list[i])
        X_list.append(mat["data_cr"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/preloadedData_CR8_W512_NoiseTrue.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y

def load_fresh_data_CR8_W1024_NoiseFree():
    I_first_damage = 4923

    files_list =["Z4_EMS01.1_CS_data_CR8_W1024.mat","Z4_EMS01.2_CS_data_CR8_W1024.mat",
                 "Z4_EMS02.1_CS_data_CR8_W1024.mat","Z4_EMS02.2_CS_data_CR8_W1024.mat",
                 "Z4_EMS03.1_CS_data_CR8_W1024.mat","Z4_EMS03.2_CS_data_CR8_W1024.mat","Z4_EMS03.3_CS_data_CR8_W1024.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("../dataset/CR8_W1024_NoiseFree/"+files_list[i])
        X_list.append(mat["data_cr"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn = Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('../dataset/preloadedData_CR8_W1024_NoiseFalse.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y

def load_preloaded_data(CR=4, W=512, Noise=False, Noise_level=-1, datasetName="Z24"):
    if Noise ==False:
        with open('../dataset/'+datasetName+'/preloadedData_CR'+str(CR)+'_W'+str(W)+'_Noise'+str(Noise)+'.pickle', 'rb') as f:
            loaded_obj = pickle.load(f)
            X = loaded_obj['X']
            Y = loaded_obj['Y']
    else:
        if(Noise_level==-1):
            with open('../dataset/'+datasetName+'/preloadedData_CR' + str(CR) + '_W' + str(W) + '_Noise' + str(Noise) + '.pickle',
                      'rb') as f:
                loaded_obj = pickle.load(f)
                X = loaded_obj['X']
                Y = loaded_obj['Y']
        else:
            with open('../dataset/'+datasetName+'/preloadedData_CR'+str(CR)+'_W'+str(W)+'_Noise'+str(Noise)+'_NL'+str(Noise_level)+'.pickle', 'rb') as f:
                loaded_obj = pickle.load(f)
                X = loaded_obj['X']
                Y = loaded_obj['Y']
    return X,Y

def reshape_data_keras(X, cr=4, window = 512, nsens=8):
    coeff= int(window/cr)
    tw = int(X.shape[2]/coeff)

    X_3d = -777777*np.ones((X.shape[0],nsens,coeff,tw))
    for i in range(0,X.shape[0]):
        for j in range(0,nsens):
            X_3d[i,j,:,:]=np.swapaxes(np.reshape(X[i,j,:tw*coeff],[1,1,tw,coeff]),2,3)

    X_3d = np.moveaxis(X_3d, [1], [3])
    return X_3d

## SciKitLearn
def fold_index(X):
    # Split Data into 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_train_ind = []
    fold_test_ind = []
    for train, test in kf.split(X):
        fold_train_ind.append(train)
        fold_test_ind.append(test)
    return   fold_train_ind,  fold_test_ind

def get_fold_split(X,Y,fold_train_ind,  fold_test_ind, i_fold):

    X_train_full = X[fold_train_ind[i_fold],:]
    y_train_full = Y[fold_train_ind[i_fold]]
    X_test = X[fold_test_ind[i_fold],:]
    y_test = Y[fold_test_ind[i_fold]]

    return X_train_full, y_train_full, X_test, y_test

def get_fold_split_keras(X,Y,fold_train_ind,  fold_test_ind, i_fold):

    X_train_full = X[fold_train_ind[i_fold],:,:,:]
    y_train_full = Y[fold_train_ind[i_fold]]
    X_test = X[fold_test_ind[i_fold],:,:,:]
    y_test = Y[fold_test_ind[i_fold]]

    return X_train_full, y_train_full, X_test, y_test

def get_fold_split_keras_TCN(X,Y,fold_train_ind,  fold_test_ind, i_fold):
    X= np.moveaxis(X, [1], [2])

    X_train_full = X[fold_train_ind[i_fold],:,:]
    y_train_full = Y[fold_train_ind[i_fold]]
    X_test = X[fold_test_ind[i_fold],:,:]
    y_test = Y[fold_test_ind[i_fold]]

    return X_train_full, y_train_full, X_test, y_test



"""

def load_fresh_data_CR8_W1024_NoiseFree():
    I_first_damage = 4923

    files_list =["Z4_EMS01.1_CS_data_CR8_W1024.mat","Z4_EMS01.2_CS_data_CR8_W1024.mat",
                "Z4_EMS02.1_CS_data_CR8_W1024.mat","Z4_EMS02.2_CS_data_CR8_W1024.mat",
                "Z4_EMS03.1_CS_data_CR8_W1024.mat","Z4_EMS03.2_CS_data_CR8_W1024.mat","Z4_EMS03.3_CS_data_CR8_W1024.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("./dataset/CR8_W1024_NoiseFree/"+files_list[i])
        X_list.append(mat["data_cr"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn =Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('preloadedData_CR8_W1024_NoiseFree.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y
    
def load_preloaded_data_CR8_W1024_NoiseFree():
    with open('preloadedData_CR8_W1024_NoiseFree.pickle', 'rb') as f:
        loaded_obj = pickle.load(f)
        X = loaded_obj['X']
        Y = loaded_obj['Y']
    return X,Y

def load_fresh_data_CR8_W512_LSM6DSL():
    I_first_damage = 4923

    files_list =["Z4_EMS01.1_CS_data_CR8_LSM6DSL.mat","Z4_EMS01.2_CS_data_CR8_LSM6DSL.mat",
                "Z4_EMS02.1_CS_data_CR8_LSM6DSL.mat","Z4_EMS02.2_CS_data_CR8_LSM6DSL.mat",
                "Z4_EMS03.1_CS_data_CR8_LSM6DSL.mat","Z4_EMS03.2_CS_data_CR8_LSM6DSL.mat","Z4_EMS03.3_CS_data_CR8_LSM6DSL.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("./dataset/CR8_W512_LSM6DSL/"+files_list[i])
        X_list.append(mat["data_cr"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn =Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('preloadedData_CR8_W512_LSM6DSL.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y

def load_preloaded_data_CR8_W512_LSM6DSL():
    with open('preloadedData_CR8_W512_LSM6DSL.pickle', 'rb') as f:
        loaded_obj = pickle.load(f)
        X = loaded_obj['X']
        Y = loaded_obj['Y']
    return X,Y

def load_preloaded_data_CR4_W512_LSM5DSL():
    with open('preloadedData_CR4_W512_LSM5DSL.pickle', 'rb') as f:
        loaded_obj = pickle.load(f)
        X = loaded_obj['X']
        Y = loaded_obj['Y']
    return X,Y


def load_preloaded_data_CR8_W512_NoiseFree():
    with open('preloadedData_CR8_W512_NoiseFree.pickle', 'rb') as f:
        loaded_obj = pickle.load(f)
        X = loaded_obj['X']
        Y = loaded_obj['Y']
    return X,Y

def load_fresh_data_CR8_W512_NoiseFree():
    I_first_damage = 4923

    files_list =["Z4_EMS01.1_CS_data_CR8.mat","Z4_EMS01.2_CS_data_CR8.mat",
                "Z4_EMS02.1_CS_data_CR8.mat","Z4_EMS02.2_CS_data_CR8.mat",
                "Z4_EMS03.1_CS_data_CR8.mat","Z4_EMS03.2_CS_data_CR8.mat","Z4_EMS03.3_CS_data_CR8.mat"]

    X_list=[]
    for i in range(len(files_list)):
        mat = scipy.io.loadmat("./dataset/CR8_W512_NoiseFree/"+files_list[i])
        X_list.append(mat["data_cr"])

    Xnp = np.concatenate(X_list,axis =0)

    Xp = Xnp[0:I_first_damage,:,:]
    Xn =Xnp[I_first_damage::,:,:]

    X = np.concatenate([Xp, Xn])
    Y = -np.concatenate([-np.ones((Xp.shape[0],1)), np.ones((Xn.shape[0],1))])

    with open('preloadedData_CR8_W512_NoiseFree.pickle', 'wb') as f:
        pickle.dump({"X":X,"Y":Y},f)

    return X,Y
"""