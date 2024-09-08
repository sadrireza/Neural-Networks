class Block:
    def __init__(self, n_filters= None, kernel_size = None, activation="relu", padding="same", is_pool = False, input_size = -1, is_dropout = False, is_childrebn = False, has_trained_weigths = False , nas_saver_name="NAS_logger" ):
        self.n_filters= n_filters
        self.kernel_size = kernel_size
        self.activation=activation
        self.padding=padding
        self.is_pool = is_pool
        self.input_size = input_size
        self.nas_saver_name = nas_saver_name

        if(is_pool == False):

            self.output_size = input_size
        else:
            self.output_size = input_size/2
        self.is_dropout = is_dropout
        self.is_children =is_childrebn
        self.has_trained_weigths = has_trained_weigths
        self.trained_weights = None

    def dump(self):
        nas_logger = open(self.nas_saver_name+'.txt', 'a')
        nas_logger.write(' Block: ' + str(self.n_filters) +' <= n_filters ' +str(self.kernel_size) +' <= kernel_size ' +str(self.activation) +' <= activation ' +str(self.padding) +' <= padding ' +str(self.is_pool) +' <= is_pool ' +str(self.input_size) +' <= input_size ' +str(self.output_size) +' <= output_size ' +str(self.is_dropout) +' <= is_dropout \n')
        nas_logger.close()
