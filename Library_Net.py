import Library_load_and_split_data
from sklearn.model_selection import train_test_split
import gc
import Library_compute_stats
from tensorflow import keras
import tensorflow as tf
#from keras_flops import get_flops
import numpy as np
#from keras_flops import get_flops


class Net:
    def __init__(self, cr, window, coeff, tw , n_sens, block_list, dataset = "Z24", nas_saver_name="NAS_logger", i_sens = None ):
        self.cr = cr
        self.window = window
        self.block_list = block_list
        self.coeff = coeff
        self.tw = tw
        self.n_sens = n_sens
        self.trained_fully = None
        self.dataset = dataset

        self.nas_saver_name = nas_saver_name
        self.i_sens = i_sens

    def fetch_data(self, Noise= False,NL=-1):
        if ((self.cr ==1) and (self.i_sens != None)):
            X, Y = Library_load_and_split_data.load_fresh_data_Z24_W512_uncompressed()
            fold_train_ind, fold_test_ind = Library_load_and_split_data.fold_index(X)
            Xd = Library_load_and_split_data.reshape_data_keras(X, cr=self.cr, window=self.window)
            Yd = Y
            Xd = np.expand_dims(Xd[:, :, :, self.i_sens], -1)
        else:
            if(self.i_sens != None):
                X, Y = Library_load_and_split_data.load_preloaded_data(CR=self.cr, W=self.window, Noise=Noise, Noise_level=NL, datasetName = self.dataset)
                Xd = Library_load_and_split_data.reshape_data_keras(X, cr=self.cr, window=self.window , nsens=self.n_sens)
                Yd = Y
                Xd = np.expand_dims(Xd[:, :, :, self.i_sens], -1)
            else:
                X, Y = Library_load_and_split_data.load_preloaded_data(CR=self.cr, W=self.window, Noise=Noise, Noise_level=NL, datasetName = self.dataset)
                Xd = Library_load_and_split_data.reshape_data_keras(X, cr=self.cr, window=self.window , nsens=self.n_sens)
                Yd = Y
        return Xd,Yd

    def short_description(self):
        self.nas_logger = open(self.nas_saver_name+'.txt', 'a')

        hw_params = self.hw_measures()
        self.nas_logger.write('Net: len = ' + str(len(self.block_list)) + ', CR = ' + str(self.cr) +', n_params = ' + str(hw_params[0]) + ', max_tens = ' + str(hw_params[1]) +', flops = ' + str(hw_params[2]) + '\n')
        self.nas_logger.close()

        return True

    def dump(self):
        nas_logger = open(self.nas_saver_name+'.txt', 'a')
        nas_logger.write(' NET: ' +str(self.cr) + ' <= cr '+str(self.window) +' <= window ' +str(self.coeff) +' <= coeff '+str(self.tw)+ ' <= tw ' +str(self.n_sens) +' <= n_sens \n')
        nas_logger.close()

        nas_logger = open(self.nas_saver_name+'.txt', 'a')
        for i in range(len(self.block_list)):
            self.block_list[i].dump()
        nas_logger.close()

    def ins_keras_model(self, load_weigths = False):
        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(self.block_list[0].n_filters, [self.coeff,self.block_list[0].kernel_size], activation=self.block_list[0].activation, padding="valid",input_shape=[self.coeff,self.tw, self.n_sens]))
        if(self.block_list[0].is_pool == True):
            model.add(keras.layers.MaxPooling2D([1, 2]))
        if (self.block_list[0].is_dropout == True):
            model.add(keras.layers.Dropout(.2))
        if(len(self.block_list)>1):
            for i_blk in range(1, len(self.block_list)):
                model.add(keras.layers.Conv2D(self.block_list[i_blk].n_filters, [1,self.block_list[i_blk].kernel_size], activation=self.block_list[i_blk].activation, padding=self.block_list[i_blk].padding))
                if (self.block_list[i_blk].is_pool == True):
                    model.add(keras.layers.MaxPooling2D([1, 2]))
                if (self.block_list[i_blk].is_dropout == True):
                    model.add(keras.layers.Dropout(.2))

        model.add(keras.layers.GlobalAveragePooling2D())
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(2, activation="softmax"))

        if(load_weigths==True):
            for i in range(0, len(model.weights)-2, 2):
                if(self.block_list[int(i/2)].has_trained_weigths== True):
                    model.weights[i] = self.block_list[int(i/2)].trained_weights[0]
                    model.weights[i+1] = self.block_list[int(i/2)].trained_weights[1]
            if(self.trained_fully != None):
                model.weights[len(model.weights)-2] =self.trained_fully[0]
                model.weights[len(model.weights)-1] =self.trained_fully[1]

        #model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

    def train_routine(self, fold_train_ind, fold_test_ind, cached_data = False, Xd = [], Yd = []):
        res_cnn = []
        learning_rate_cb = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        early_stop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        n_epochs = 100
        multistart = 5
        batch_size = 256

        if(cached_data) == False:
            Xd,Yd = self.fetch_data()
        else:
            Xd = Xd
            Yd = Yd

        for i_fold in range(0, len(fold_train_ind)):
            X_train_full, y_train_full, X_test, y_test = Library_load_and_split_data.get_fold_split_keras(Xd, Yd,
                                                                                                          fold_train_ind,
                                                                                                          fold_test_ind,
                                                                                                          i_fold)
            X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25,
                                                              random_state=42)

            y_train_c = keras.utils.to_categorical((y_train + 1) / 2, num_classes=2)
            y_val_c = keras.utils.to_categorical((y_val + 1) / 2, num_classes=2)
            y_test_c = keras.utils.to_categorical((y_test + 1) / 2, num_classes=2)

            best_val = 0
            for i_mult in range(multistart):
                model = self.ins_keras_model()
                model.fit(X_train, y_train_c, epochs=n_epochs, batch_size=batch_size, use_multiprocessing=False,
                          validation_data=(X_val, y_val_c), callbacks=[learning_rate_cb, early_stop_cb])
                p_val = model.predict(X_val)
                cm_p_v, accuracy_p_v, prec_p_v, rec_p_v, f1_p_v, cm_n_v, accuracy_n_v, prec_n_v, rec_n_v, f1_n_v = Library_compute_stats.compute_descriptors(
                    y_val_c, p_val)
                if (best_val < (accuracy_p_v + prec_p_v + rec_p_v + f1_p_v )):
                    best_val = (accuracy_p_v + prec_p_v + rec_p_v + f1_p_v)
                    p_test = model.predict(X_test)
                    cm_p, accuracy_p, prec_p, rec_p, f1_p, cm_n, accuracy_n, prec_n, rec_n, f1_n = Library_compute_stats.compute_descriptors(
                        y_test_c, p_test)

                del model
                gc.collect()
                keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()

            print(cm_p, accuracy_p, prec_p, rec_p, f1_p, cm_n, accuracy_n, prec_n, rec_n, f1_n)
            res_cnn.append([cm_p, accuracy_p, prec_p, rec_p, f1_p, cm_n, accuracy_n, prec_n, rec_n, f1_n])
        return res_cnn

    def proxy_train_routine(self, is_dev, fold_train_ind, fold_test_ind, cached_data = False, Xd = [], Yd = []):
        res_cnn = []
        learning_rate_cb = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        early_stop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        n_epochs = 100
        batch_size = 256
        multistart = 5

        if(cached_data) == False:
            Xd,Yd = self.fetch_data()
        else:
            Xd = Xd
            Yd = Yd

        i_fold = 0
        X_train_full, y_train_full, X_test, y_test = Library_load_and_split_data.get_fold_split_keras(Xd, Yd,fold_train_ind,fold_test_ind,i_fold)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25,random_state=42)

        y_train_c = keras.utils.to_categorical((y_train + 1) / 2, num_classes=2)
        y_val_c = keras.utils.to_categorical((y_val + 1) / 2, num_classes=2)
        y_test_c = keras.utils.to_categorical((y_test + 1) / 2, num_classes=2)

        best_val = 0
        for i_mult in range(multistart):
            model = self.ins_keras_model()
            if(is_dev == False):
                model.fit(X_train, y_train_c, epochs=n_epochs, batch_size=batch_size, use_multiprocessing=False,
                      validation_data=(X_val, y_val_c), callbacks=[learning_rate_cb, early_stop_cb])

            p_val = model.predict(X_val)
            cm_p_v, accuracy_p_v, prec_p_v, rec_p_v, f1_p_v, cm_n_v, accuracy_n_v, prec_n_v, rec_n_v, f1_n_v = Library_compute_stats.compute_descriptors(
                y_val_c, p_val)
            if (best_val < (accuracy_p_v + prec_p_v + rec_p_v + f1_p_v)):
                best_val = (accuracy_p_v + prec_p_v + rec_p_v + f1_p_v)
                p_test = model.predict(X_test)
                cm_p, accuracy_p, prec_p, rec_p, f1_p, cm_n, accuracy_n, prec_n, rec_n, f1_n = Library_compute_stats.compute_descriptors(
                    y_test_c, p_test)

            del model
            gc.collect()
            keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()

        print(cm_p, accuracy_p, prec_p, rec_p, f1_p, cm_n, accuracy_n, prec_n, rec_n, f1_n)
        res_cnn.append([cm_p, accuracy_p, prec_p, rec_p, f1_p, cm_n, accuracy_n, prec_n, rec_n, f1_n])

        return res_cnn

    def hw_measures(self):
        model = self.ins_keras_model()
        n_params = model.count_params()

        #Max tensor propagated
        layers_outputs = [layer.output for layer in model.layers]
        max_tens = model._input_layers[0].input_shape[0][3]*model._input_layers[0].input_shape[0][2]*model._input_layers[0].input_shape[0][1]
        for i_outs in range(len(layers_outputs)):
            tmp = 1
            for i_dim in range(1,len(layers_outputs[i_outs].shape)):
                tmp = tmp*layers_outputs[i_outs].shape[i_dim]
            if(tmp>max_tens):
                max_tens = tmp

        flops = -7#get_flops(model, batch_size=1)

        flash_size = 4* n_params
        ram_size = 4* max_tens

        return  [flash_size, ram_size, flops]
