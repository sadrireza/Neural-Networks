import Library_load_and_split_data
import Library_Block
import Library_Net

"""LOAD DATASET"""
cr = 8
window = 512
Noise = False
NL = -1
sens = 8
coeff = 64
tw = 64

X, Y = Library_load_and_split_data.load_preloaded_data(CR=cr, W=window, Noise=Noise, Noise_level=NL, datasetName= "Z24")
Xd = Library_load_and_split_data.reshape_data_keras(X, cr=cr, window=window, nsens=sens)
Yd = Y
fold_train_ind, fold_test_ind = Library_load_and_split_data.fold_index(X)

b1 = Library_Block.Block(n_filters= 19, kernel_size = 2, activation="relu", padding="same", is_pool = False, input_size = 64, is_dropout = False, is_childrebn = False, has_trained_weigths = False )
b2 = Library_Block.Block(n_filters= 26, kernel_size = 1, activation="relu", padding="same", is_pool = True,  input_size = 64, is_dropout = True,  is_childrebn = False, has_trained_weigths = False )
b3 = Library_Block.Block(n_filters= 5,  kernel_size = 1, activation="relu", padding="same", is_pool = False, input_size = 32, is_dropout = False, is_childrebn = False, has_trained_weigths = False )
b4 = Library_Block.Block(n_filters= 6,  kernel_size = 5, activation="relu", padding="same", is_pool = False, input_size = 32, is_dropout = False, is_childrebn = False, has_trained_weigths = False )
b5 = Library_Block.Block(n_filters= 19, kernel_size = 1, activation="relu", padding="same", is_pool = True,  input_size = 32, is_dropout = True,  is_childrebn = False, has_trained_weigths = False )
b6 = Library_Block.Block(n_filters= 12, kernel_size = 1, activation="relu", padding="same", is_pool = True,  input_size = 16, is_dropout = True,  is_childrebn = False, has_trained_weigths = False )
sel_net = Library_Net.Net(cr, window, coeff, tw , sens, [b1,b2,b3,b4,b5,b6])
model=sel_net.ins_keras_model()
print(sel_net.hw_measures())

#Refine Selected Models
res_cnn_last = sel_net.proxy_train_routine(is_dev=False, fold_train_ind=fold_train_ind, fold_test_ind=fold_test_ind)
