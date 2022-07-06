from . import graph, coarsening
import tensorflow as tf, sklearn, scipy.sparse, numpy as np, os, time, collections, shutil
from scipy import linalg
import scipy.sparse as sp


class base_model(object):

    def __init__(self):
        self.regularizers = []

    def predict(self, data, labels=None, sess=None, types=30):
        loss = 0
        size = data.shape[0]
        predictions = np.empty((size, types))
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            batch_data = np.zeros((self.batch_size, data.shape[1]))
            tmp_data = data[begin:end, :]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()
                
            batch_data[:end - begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1}
            if labels is not None:
                batch_labels = np.zeros((self.batch_size, types))
                batch_labels[:end - begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss

            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)
            

            predictions[begin:end] = batch_pred[:end - begin]

        if labels is not None:
            return (predictions, loss * self.batch_size / size)
        else:
            return predictions

    def evaluate(self, data, labels, sess=None, types=30):
        t_process, t_wall = time.process_time(), time.time()
        predictions, loss = self.predict(data, labels, sess, types)
        print('loss:',loss)
        string = ('loss: {:.2e}').format(loss)
        if sess is None:
            string += ('\ntime: {:.0f}s (wall {:.0f}s)').format(time.process_time() - t_process, time.time() - t_wall)
        return (string, loss, predictions)

    def fit(self, train_data, train_labels, val_data, val_labels):
        t_process, t_wall = time.process_time(), time.time()
        sess = tf.Session(graph=self.graph)
        shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        print('get_path(summaries:',self._get_path('summaries'))
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)

        path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
               
        
        for step in range(1, num_steps + 1):
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]
            batch_data, batch_labels = train_data[idx, :], train_labels[idx, :]
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()
                
            feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_dropout: self.dropout}


            learning_rate, loss, predictions,values, i_value = sess.run([self.op_train, self.op_loss, self.op_prediction, self.op_value, self.op_i], feed_dict)








            print('steps:'+str(step)+';num_steps:'+str(num_steps)+';loss:'+str(loss))
            
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step/self.eval_frequency
                print(('step {} / {} (epoch {:.2f} / {}):').format(step, num_steps, epoch, self.num_epochs))
                print(('  learning_rate = {:.2e}').format(learning_rate))


                string, loss, predictions = self.evaluate(val_data, val_labels, sess)
                
                
                rightNum=0
                for temp_idx in range(val_labels.shape[0]):
                    if np.argmax(predictions[temp_idx])==np.argmax(val_labels[temp_idx]):
                        rightNum=rightNum+1       
                print('测试分类准确率为:%.2f',rightNum/val_labels.shape[0])
    

                losses.append(loss)
                print(('  validation {}').format(string))
                print(('  0310-gate3 time: {:.0f}s (wall {:.0f}s)').format(time.process_time() - t_process, time.time() - t_wall))




                self.op_saver.save(sess, path, global_step=step)
        writer.close()
        sess.close()
        t_step = (time.time() - t_wall) / num_steps
        return (losses, t_step)

    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    def build_graph(self, M_0=45, inputDepth=200, types=30, channel=1):

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('inputs'):
                self.ph_data = tf.placeholder(tf.int32, (self.batch_size, inputDepth), name='ph_data')
                self.ph_labels = tf.placeholder(tf.float32, (self.batch_size, types), name='ph_labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), name='ph_dropout')

            

            op_logits,op_value,op_i = self.inference(self.ph_data)

            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels, self.regularization)

            self.op_train = self.training(self.op_loss, self.learning_rate, self.decay_steps, self.decay_rate, self.momentum)              
            self.op_init = tf.global_variables_initializer()
            self.op_prediction=op_logits
            self.op_value=op_value
            self.op_i=op_i
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)
        self.graph.finalize()

    def inference(self, data):
        x, value, i = self._inference(data)
        return x, value, i
    
    def probabilities(self, logits):

        with tf.name_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, logits):

        with tf.name_scope('prediction'):
            prediction = tf.argmax(logits, axis=1)
            return prediction

    def loss(self, logits, labels, regularization):

        with tf.name_scope('loss'):
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)

            
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([regularization, loss])
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return (loss, loss_average)   
    

    def count_flops(self,graph):
        flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
        return flops.total_float_ops, params.total_parameters
    
    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):

        with tf.name_scope('training'):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            if momentum == 0:
                optimizer = tf.train.AdamOptimizer(learning_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            
            grads = optimizer.compute_gradients(loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            for grad, var in grads:           
                if grad is None:
                    print(('warning: {} has no gradient').format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            
            print('---training-')
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train
    
    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess=None):

        if sess is None:
            sess = tf.Session(graph=self.graph)
            print('self._get_path(checkpoints):', self._get_path('checkpoints'))
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            print('filename:', filename)
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        print('var.op.name:', var.op.name)
        tf.summary.histogram(var.op.name, var)
        return var

    def _definedWeight_variable(self, name, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable(name, shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        print(name+':', var.op.name)
        tf.summary.histogram(var.op.name, var)
        return var
    
    def _temporal1Weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('temporal1_weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        print('temporal1 var.op.name:', var.op.name)
        tf.summary.histogram(var.op.name, var)
        return var

    def _temporal2Weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('temporal2_weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        print('temporal2 var.op.name:', var.op.name)
        tf.summary.histogram(var.op.name, var)
        return var

    def _definedTuckerWeight_variable(self, name, shape, scope, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE): 
            var = tf.get_variable(name, shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        print(name+':', var.op.name)
        tf.summary.histogram(var.op.name, var)
        return var
    
    def _temporal1Cof_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('temporal1_cofs', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        print('temporal1_cofs var.op.name:', var.op.name)
        tf.summary.histogram(var.op.name, var)
        return var

    def cov_variable(self, name, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable(name, shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var
    
    def _temporal2Cof_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('temporal2_cofs', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        print('temporal2_cofs var.op.name:', var.op.name)
        tf.summary.histogram(var.op.name, var)
        return var
    
    def _spatialWeight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('spatial_weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        print('spatial var.op.name:', var.op.name)
        tf.summary.histogram(var.op.name, var)
        return var
    
    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var
    
    def defined_bias_variable(self, name, shape, scope, regularization=True):
        initial = tf.constant_initializer(0.1)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            var = tf.get_variable(name, shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

class cgcnn(base_model):
    def __init__(self, Cout_temporal, Cout_spatial, K_temporal, K_spatial, p, M, filter='mixedSubstructureFilter', brelu='b1relu', pool='mpool1', num_epochs=20, learning_rate=0.01, decay_rate=0.95, decay_steps=None, momentum=0.9, regularization=0, dropout=0.9, batch_size=100, eval_frequency=200, dir_name='', mapMatrixs=None, remapMatrixs=None, dymicW=None, Mask=None, originL=None,
                 scale_=None, mean_=None, sparse_AajaceMatrixList=None, AjacentMatrix=None, mainLenList=None):
        print('begin initial!')
        super().__init__()
        self.Cout_temporal, self.Cout_spatial, self.K_temporal, self.K_spatial, self.p, self.M, self.dymicW = (
         Cout_temporal, Cout_spatial, K_temporal, K_spatial, p, M, dymicW)
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)
        self.mapMatrixs = mapMatrixs
        self.remapMatrixs = remapMatrixs
        self.sparse_AajaceMatrixList=sparse_AajaceMatrixList
        self.Mask = Mask
        self.originL = originL
        self.scale_=scale_
        self.mean_=mean_
        self.AjacentMatrix=AjacentMatrix
        self.mainLenList=mainLenList
        self.build_graph(1515, inputDepth=1000)
        print('self.mean_=tf.convert_to_tensor(mean_)')
    
    def temporalSpatialBlock(self, x_input, spatialL, Cout_temporal, Cout_spatial, K_temporal, K_spatial, scope,  r=50):
        batch, N, M, Cin = x_input.get_shape()
        


        x_inputlist_spatialLayer=tf.split(x_input,M,2)
        x_outputList_spatialLayer=[]
        spatialKernel = self._definedTuckerWeight_variable('deconv2Kernel',[K_spatial, Cin, Cout_spatial], scope, regularization=False)
        for tempIdx in range(M):
            temp_input=tf.reshape(x_inputlist_spatialLayer[tempIdx],[batch, N, Cin])
            temp_output=self.chebyshev5ForBaseline(temp_input, spatialL, Cout_spatial, K_spatial, spatialKernel)
            x_outputList_spatialLayer.append(temp_output)
        x_output_spatialLayer=tf.stack(x_outputList_spatialLayer, axis=2)
        return x_output_spatialLayer

    def calculate_output_shape(self, x_input, filter_size_h, filter_size_w, stride_h, stride_w, num_outputs, outputShapeType=0, padding='SAME', data_format='NHWC'):

        if data_format == "NHWC":
            input_channel_size = x_input.get_shape().as_list()[3]
            input_size_h = x_input.get_shape().as_list()[1]
            input_size_w = x_input.get_shape().as_list()[2]
            stride_shape = [1, stride_h, stride_w, 1]
            if padding == 'VALID':

                output_size_w = (input_size_w - 1)*stride_w + filter_size_w
            elif padding == 'SAME':
                if input_size_w==2 or input_size_w==12:
                    output_size_w = (input_size_w - 1)*stride_w + 1
                else:
                    output_size_w = input_size_w*stride_w 
            else:
                raise ValueError("unknown padding")
    
            output_shape = tf.stack([tf.shape(x_input)[0], 
                                input_size_h, output_size_w, 
                                num_outputs])
        elif data_format == "NCHW":
            input_channel_size = x_input.get_shape().as_list()[1]
            input_size_h = x_input.get_shape().as_list()[2]
            input_size_w = x_input.get_shape().as_list()[3]
            stride_shape = [1, 1, stride_h, stride_w]
            if padding == 'VALID':

                output_size_w = (input_size_w - 1)*stride_w + filter_size_w
            elif padding == 'SAME':

                output_size_w = (input_size_w - 1)*stride_w + 1
            else:
                raise ValueError("unknown padding")
    
            output_shape = tf.stack([tf.shape(x_input)[0], 
                                    input_size_h, output_size_w, num_outputs])
        else:
            raise ValueError("unknown data_format")
    
        return output_shape

    def deTemporalSpatialBlock(self, x_input, spatialL, Cout_temporal, Cout_spatial, K_temporal, K_spatial, output_shapeType, scope, secondChannelNum):
        x_output=None;
        print('x_input.get_shape():',x_input.get_shape())
        batch, N, M, Cin = x_input.get_shape()
        temporalDeconvKernel1=self._definedTuckerWeight_variable('deconv1Kernel',[1, K_temporal, Cout_spatial, Cin], scope, regularization=False)
        print('**x_input:',x_input)
        
        strideHeight=1
        strideWidth=2
        out_shape=self.calculate_output_shape(x_input, 1, K_temporal, strideHeight, strideWidth, Cout_spatial, output_shapeType)
        print('out_shape:',out_shape)

        x_detemporal1=tf.nn.conv2d_transpose(x_input, 
                                 temporalDeconvKernel1, 
                                 output_shape=out_shape, 
                                 strides=[1,strideHeight,strideWidth,1], 
                                 padding="SAME", 
                                 data_format="NHWC", 
                                 name=None)

        batch, N, M, Cin = x_detemporal1.get_shape()
        

        print('x_detemporal1:',x_detemporal1)
        x_inputlist_spatialLayer=tf.split(x_detemporal1,M,2)
        x_outputList_spatialLayer=[]
        spatialKernel = self._definedTuckerWeight_variable('deconv2Kernel',[K_spatial, Cout_spatial, Cout_temporal], scope, regularization=False)
        for tempIdx in range(M):
            temp_input=tf.reshape(x_inputlist_spatialLayer[tempIdx],[batch, N, Cout_spatial])
            temp_output=self.chebyshev5ForBaseline(temp_input, spatialL, Cout_temporal, K_spatial, spatialKernel)
            x_outputList_spatialLayer.append(temp_output)
        x_output_spatialLayer=tf.stack(x_outputList_spatialLayer, axis=2)
        

        print('x_output_spatialLayer:',x_output_spatialLayer) 

        batch, N, M, Cin = x_output_spatialLayer.get_shape()
        out_shape=self.calculate_output_shape(x_output_spatialLayer, 1, K_temporal, strideHeight, strideWidth, secondChannelNum, output_shapeType)
        temporalDeconvKernel2=self._definedTuckerWeight_variable('deconv3Kernel',[1, K_temporal, secondChannelNum, Cin], scope, regularization=False)
        x_output=tf.nn.conv2d_transpose(x_output_spatialLayer, 
                                 temporalDeconvKernel2, 
                                 output_shape=out_shape, 
                                 strides=[1,strideHeight,strideWidth,1], 
                                 padding="SAME", 
                                 data_format="NHWC", 
                                 name=None)

        




        print('x_output:',x_output)
        return x_output, spatialL;
    
    def chebyshev5ForBaseline(self, x, LTensors, Fout, K, W):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)      
        x0=tf.identity(x)
        x = tf.expand_dims(x0, 0)
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)
            return tf.concat([x, x_], axis=0)

        if K > 1: 
            x1 = tf.matmul(LTensors, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.matmul(LTensors, x1) - x0
            x = concat(x, x2)
            x0, x1 = x1, x2

        return tf.tensordot(x, W,[[0,3],[0,1]])
    


    def chebyshev5(self, x, L, Fout, K, W):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        L = scipy.sparse.csr_matrix(L) 
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        x0 = tf.transpose(x, perm=[1, 2, 0])
        x0 = tf.reshape(x0, [M, Fin * N])
        x = tf.expand_dims(x0, 0)
        
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)
            return tf.concat([x, x_], axis=0)

        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0
            x = concat(x, x2)
            x0, x1 = x1, x2

        x = tf.reshape(x, [K, M, Fin, N])
        x = tf.transpose(x, perm=[3, 1, 2, 0])
        x = tf.reshape(x, [N * M, Fin * K])
        x = tf.matmul(x, W)
        return tf.reshape(x, [N, M, Fout]) 

    def b1relu(self, x, name, scope):

        batch, N, M, C = x.get_shape()
        b = self.defined_bias_variable(name, [1, int(N), int(M), int(C)], scope, regularization=False)
        return tf.nn.relu(x + b)
    
    def cov1_b1relu(self, x):

        batch, N, M, C = x.get_shape()
        b = self.defined_bias_variable('cov1_bias',[1, int(N), int(M), int(C)], regularization=False)
        return tf.nn.relu(x + b)
    
    def cov2_b1relu(self, x):

        batch, N, M, C = x.get_shape()
        b = self.defined_bias_variable('cov2_bias',[1, int(N), int(M), int(C)], regularization=False)
        return tf.nn.relu(x + b)

    def apool1(self, x, p):

        if p > 1:
            x = tf.expand_dims(x, 3)
            x = tf.nn.avg_pool(x, ksize=[1, p, 1, 1], strides=[1, p, 1, 1], padding='SAME')
            return tf.squeeze(x, [3])
        else:
            return x
        
    def prelu(self, _x, name, scope=None):

        with tf.variable_scope(name_or_scope=scope, default_name="prelu", reuse=tf.AUTO_REUSE):
            _alpha = tf.get_variable(name, shape=_x.get_shape()[-1],
                                     dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
            return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

    def fc(self, x, Mout, name, relu=True):

        N, Min = x.get_shape()
        scope='fc'
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE): 
            W = self._definedTuckerWeight_variable(name+'w', [Min, Mout], scope, regularization=True)
            b = self.defined_bias_variable(name+'bias', [Mout], scope, regularization=False)
        x = tf.matmul(x, W) + b
        if relu:
            return tf.nn.relu(x)
        return x

    def stackArrayToTensor(self, A, stackNums):
        aList = []
        A = A.astype(np.float32)
        for i in range(stackNums):
            aList.append(A.copy())

        x = tf.convert_to_tensor(aList[0], dtype=tf.float32)
        x = tf.identity(tf.expand_dims(x, 0), name='dynamicW')
        for i in range(1, stackNums):
            temp_x = tf.convert_to_tensor(aList[i])
            temp_x = tf.expand_dims(temp_x, 0)
            x = tf.concat([x, temp_x], axis=0)

        return x
    
    def scaleTimeT(self, x):
        batch, N, M, C = x.get_shape()
        scope='scaleTime'
        temporal1Kernel=self._definedTuckerWeight_variable('xInput1Kernel', [1, 4, C, C], scope, regularization=False)
        x=tf.nn.conv2d(x, 
                        temporal1Kernel,
                        strides=[1,1,1,1], 
                        padding='SAME')
        

        x = tf.nn.avg_pool(x,[1,1,4,1],[1,1,2,1],padding='VALID')
        
        temporal2Kernel=self._definedTuckerWeight_variable('xInput2Kernel', [1, 4, C, C], scope, regularization=False)
        x=tf.nn.conv2d(x, 
                        temporal2Kernel,
                        strides=[1,1,1,1], 
                        padding='SAME')
        
        x = tf.nn.avg_pool(x,[1,1,4,1],[1,1,2,1],padding='VALID')
        
        temporal3Kernel=self._definedTuckerWeight_variable('xInput3Kernel', [1, 4, C, C], scope, regularization=False)
        x=tf.nn.conv2d(x, 
                        temporal3Kernel,
                        strides=[1,1,1,1], 
                        padding='SAME')
        
        x = tf.nn.avg_pool(x,[1,1,4,1],[1,1,2,1],padding='VALID')
        return x
    
    def dot(self, x, y, sparse=False):

      if sparse:
          print('sparse x:',x)
          res = tf.sparse_tensor_dense_matmul(x, y)
      else:
          res = tf.matmul(x, y)
      return res
    
    def convert_sparse_matrix_to_sparse_tensor(self, coo):

      indices = np.transpose(np.array([coo.row, coo.col]))
      return tf.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)

    def gcnPooling(self, X, A, idx, scope, pooling_ratio = 0.8):




        


        batch, N, M, C = X.get_shape()
        X=tf.reshape(X, [batch*N, M*C])
        







        

        self.p = self._definedTuckerWeight_variable(name='p', 
                             shape=(M*C, 1),
                             scope=scope)
        

        

        p_norm = tf.nn.l2_normalize(self.p, axis=0)
        proj = tf.divide(self.p, p_norm)
        y = self.dot(X, proj, sparse=False)


        
        feat = []
        adj = []
        indices = []




        for g in range(batch):












          gg = tf.reshape(tf.where(tf.equal(idx, g)), [-1])


          X_g = tf.gather(X, indices=gg)

          A_g = tf.reshape(tf.gather(A, g, axis=0), [N,N])
          idx_g = tf.gather(idx, gg, axis=0)

          y_g = tf.gather(y, indices=gg, axis=0)
          



          if pooling_ratio < 1:
            to_keep = tf.cast(tf.math.ceil(int(N)*pooling_ratio), dtype=tf.int32)
          else:
            to_keep = N





          y_g_identi=tf.identity(y_g)
          y_g_identi=tf.reshape(y_g_identi, [-1])
          value, i = tf.nn.top_k(y_g_identi, to_keep)
          

          X_g = tf.gather(X_g, indices=i)
          A_g = tf.gather(tf.gather(A_g, i, axis=0),  i, axis=1)
          idx_g = tf.gather(idx_g, i, axis=0)
          
          y_tilde = tf.tanh(tf.gather(y_g, i, axis=0))
          X_tilde = tf.multiply(X_g, y_tilde)
          
          feat.append(X_tilde) 
          adj.append(A_g)
          indices.append(idx_g)
        

        X_stack = tf.concat(feat, axis=0)
        A_diag = tf.concat(adj, axis=0)
        idx_seq = tf.concat(indices, axis=-1)
        
        spatialL=tf.reshape(A_diag, [batch, to_keep, to_keep])
        spatialL=spatialL-tf.eye(to_keep)
        spatialL_degree=tf.abs(tf.reduce_sum(spatialL,axis=2,keepdims=True))
        spatialL=tf.eye(to_keep)+spatialL/tf.clip_by_value(spatialL_degree, 0.00000001,10000000)
        

        X_stack=tf.reshape(X_stack, [batch, -1, M, C])
        return X_stack, spatialL, idx_seq

 
    def redefine_A_2(self, x, A, idx=None):


        batch, N, M, C = x.get_shape()
        
        x = tf.reshape(x, [batch,N,M,2]) 
        inpus=tf.split(x,2,3)
        x_mid=tf.reshape(inpus[0],[batch,N,M])
        x_t_mid=tf.reshape(inpus[1],[batch,N,M])
        

        param1 = self._definedTuckerWeight_variable('param1',[M, 20], 'redefineA', regularization=False)
        param2 = self._definedTuckerWeight_variable('param2',[M, 20], 'redefineA', regularization=False)
        
        x_mid=tf.tensordot(x_mid, param1, axes=(2,0))
        x_t_mid=tf.tensordot(x_t_mid, param2, axes=(2,0))
        
        row1_mean=tf.reduce_mean(x_mid, axis=2, keep_dims=True)
        x1_mean=x_mid-row1_mean
        row1_squaresum=1/tf.sqrt(tf.clip_by_value(tf.reduce_sum((1/int(M-1))*tf.square(x1_mean), axis=2, keep_dims=True), 0.00000001,10000000))
        
        row2_mean=tf.reduce_mean(x_t_mid, axis=2, keep_dims=True)
        x2_mean=x_t_mid-row2_mean

        row2_squaresum_u=1/tf.sqrt(tf.clip_by_value(tf.reduce_sum((1/int(M-1))*tf.square(x2_mean), axis=2, keep_dims=True), 0.00000001,10000000))
        row2_squaresum=tf.transpose(row2_squaresum_u,[0,2,1])


        squaremiu=tf.matmul(row1_squaresum,tf.transpose(row1_squaresum,[0,2,1]))
        x1_mean_tp=tf.transpose(x1_mean, [0,2,1])
        similar=tf.matmul(x1_mean,x1_mean_tp)
        similar_uu=tf.multiply(similar, squaremiu)
        

        squaremiu=tf.matmul(row1_squaresum,row2_squaresum)
        x2_mean_tp=tf.transpose(x2_mean, [0,2,1])
        similar=tf.matmul(x1_mean,x2_mean_tp)
        similar_ud=tf.multiply(similar, squaremiu)


        x1_mean_tp=tf.transpose(x1_mean, [0,2,1])
        similar=tf.matmul(x2_mean,x1_mean_tp)
        similar_du=tf.multiply(similar, squaremiu)
        

        squaremiu=tf.matmul(row2_squaresum_u,row2_squaresum)
        x2_mean_tp=tf.transpose(x2_mean, [0,2,1])
        similar=tf.matmul(x2_mean,x2_mean_tp)
        similar_dd=tf.multiply(similar, squaremiu)

        lambdaP = self._definedTuckerWeight_variable('lamdaP',[5,1], 'redefineA', regularization=False)        

        new_ATensor=tf.concat([similar_uu,similar_ud,similar_du,similar_dd,A], axis=2)
        new_ATensor=tf.reshape(new_ATensor, [batch,N,N,5])

        new_A=tf.tensordot(new_ATensor, lambdaP, axes=(3,0))
        new_A=tf.reshape(new_A, [batch,N,N])


        
        new_A=tf.abs(new_A)
        A_degree=tf.clip_by_value(tf.reduce_sum(new_A,axis=2), 0.00000001,10000000)
        d = 1 / A_degree
        d=tf.matrix_diag(d)
        new_A=tf.matmul(d, new_A)
        
        one_tensor=tf.ones(shape=[batch,N,N], dtype="float32")
        eye_tensor=tf.eye(int(N),batch_shape=[batch],dtype=tf.float32)
        oe_tensor=one_tensor-eye_tensor

        new_A=tf.multiply(new_A,oe_tensor)
        return new_A
    
    def redefine_A_3(self, x, A, idx=None):


        batch, N, M, C = x.get_shape()
        
        x = tf.reshape(x, [batch,N,M*C]) 
        

        param1 = self._definedTuckerWeight_variable('param1',[1, 1, M*C], 'redefineA', regularization=False)
        param2 = self._definedTuckerWeight_variable('param2',[1, 1, M*C], 'redefineA', regularization=False)
        param1 = tf.tile(param1, multiples=[batch, N, 1])
        param2 = tf.tile(param2, multiples=[batch, N, 1])
        



        

        
        x_mid=tf.multiply(x, param1)
        x_t_mid=tf.multiply(x, param2)
        
        row1_mean=tf.reduce_mean(x_mid, axis=2, keep_dims=True)
        x1_mean=x_mid-row1_mean
        row1_squaresum=1/tf.sqrt(tf.clip_by_value(tf.reduce_sum((1/int(M-1))*tf.square(x1_mean), axis=2, keep_dims=True), 0.00000001,10000000))
        
        row2_mean=tf.reduce_mean(x_t_mid, axis=2, keep_dims=True)
        x2_mean=x_t_mid-row2_mean

        row2_squaresum_u=1/tf.sqrt(tf.clip_by_value(tf.reduce_sum((1/int(M-1))*tf.square(x2_mean), axis=2, keep_dims=True), 0.00000001,10000000))
        row2_squaresum=tf.transpose(row2_squaresum_u,[0,2,1])
        

        squaremiu=tf.matmul(row1_squaresum,row2_squaresum)
        x2_mean_tp=tf.transpose(x2_mean, [0,2,1])
        similar=tf.matmul(x1_mean,x2_mean_tp)
        similar_ud=tf.multiply(similar, squaremiu)

        new_A=tf.nn.relu(similar_ud)+A

        
        new_A=tf.abs(new_A)
        A_degree=tf.clip_by_value(tf.reduce_sum(new_A,axis=2), 0.00000001,10000000)
        d = 1 / A_degree
        d=tf.matrix_diag(d)
        new_A=tf.matmul(d, new_A)
        
        one_tensor=tf.ones(shape=[batch,N,N], dtype="float32")
        eye_tensor=tf.eye(int(N),batch_shape=[batch],dtype=tf.float32)
        oe_tensor=one_tensor-eye_tensor

        new_A=tf.multiply(new_A,oe_tensor)
        return new_A

    def matsubtract(self, x, param):
        batch, N, M = x.get_shape()
        nodeFeatureList = tf.split(x, N, 1)
        
        resultList = []



        param = tf.tile(param, multiples=[batch, 1])
            
        for j in range(N):
            for k in range(N):                   
                subtractRes = tf.abs(tf.subtract(nodeFeatureList[j], nodeFeatureList[k]))
                subtractRes = tf.reshape(subtractRes, [batch,M])
                print('subtractRes:',subtractRes)
                tempRes=tf.reduce_sum(tf.multiply(subtractRes, param),axis=1)
                resultList.append(tempRes)    
            
        res = tf.stack(resultList)
        res = tf.reshape(res, [N, N, batch])
        res = tf.transpose(res, [2,0,1])


        return res

    def redefine_A_4(self, x, A, idx=None):


        batch, N, M, C = x.get_shape()
        
        one_tensor=tf.ones(shape=[batch,N,N], dtype="float32")
        eye_tensor=tf.eye(int(N),batch_shape=[batch],dtype=tf.float32)
        oe_tensor=one_tensor-eye_tensor

        A=tf.abs(tf.multiply(A,oe_tensor))
        
        x = tf.reshape(x, [batch,N,M*C]) 
        


        param1 = self._definedTuckerWeight_variable('param1',[M*C, 20], 'redefineA', regularization=False)
        param2 = self._definedTuckerWeight_variable('param2',[M*C, 20], 'redefineA', regularization=False)
        
        x_mid=tf.tensordot(x, param1, axes=(2,0))
        x_t_mid=tf.tensordot(x, param2, axes=(2,0))
        
        row1_mean=tf.reduce_mean(x_mid, axis=2, keep_dims=True)
        x1_mean=x_mid-row1_mean
        row1_squaresum=1/tf.sqrt(tf.clip_by_value(tf.reduce_sum((1/int(M-1))*tf.square(x1_mean), axis=2, keep_dims=True), 0.00000001,10000000))
        
        row2_mean=tf.reduce_mean(x_t_mid, axis=2, keep_dims=True)
        x2_mean=x_t_mid-row2_mean

        row2_squaresum_u=1/tf.sqrt(tf.clip_by_value(tf.reduce_sum((1/int(M-1))*tf.square(x2_mean), axis=2, keep_dims=True), 0.00000001,10000000))
        row2_squaresum=tf.transpose(row2_squaresum_u,[0,2,1])

        

        squaremiu=tf.matmul(row1_squaresum,row2_squaresum)
        x2_mean_tp=tf.transpose(x2_mean, [0,2,1])
        similar=tf.matmul(x1_mean,x2_mean_tp)
        similar_ud=tf.multiply(similar, squaremiu)
        

        
        new_A=tf.nn.relu(similar_ud)+A
        new_A=tf.abs(new_A)
        A_degree=tf.clip_by_value(tf.reduce_sum(new_A,axis=2), 0.00000001,10000000)
        d = 1 / A_degree
        d=tf.matrix_diag(d)
        new_A=tf.matmul(d, new_A)
        
        one_tensor=tf.ones(shape=[batch,N,N], dtype="float32")
        eye_tensor=tf.eye(int(N),batch_shape=[batch],dtype=tf.float32)
        oe_tensor=one_tensor-eye_tensor

        new_A=eye_tensor-tf.multiply(new_A,oe_tensor)
        return new_A
    
    def redefine_A_5(self, x, A, idx=None):


        batch, N, M, C = x.get_shape()
        eye_tensor=tf.eye(int(N),batch_shape=[batch],dtype=tf.float32)
        A=tf.abs(A-eye_tensor)
        
        x = tf.reshape(x, [batch,N,M*C])        
        param = self._definedTuckerWeight_variable('param',[1,M*C], 'redefineA', regularization=False)


        
        similar=self.matsubtract(x,param)
        
        new_A=tf.nn.relu(similar)+A
        new_A=tf.abs(new_A)
        A_degree=tf.clip_by_value(tf.reduce_sum(new_A,axis=2), 0.00000001,10000000)
        d = 1 / A_degree
        d=tf.matrix_diag(d)
        new_A=tf.matmul(d, new_A)
        
        one_tensor=tf.ones(shape=[batch,N,N], dtype="float32")
        eye_tensor=tf.eye(int(N),batch_shape=[batch],dtype=tf.float32)
        oe_tensor=one_tensor-eye_tensor

        new_A=eye_tensor-tf.multiply(new_A,oe_tensor)
        return new_A,similar
    
    def redefine_A_6(self, x, A, name=None, idx=None):


        batch, N, M, C = x.get_shape()
        
        one_tensor=tf.ones(shape=[batch,N,N], dtype="float32")
        eye_tensor=tf.eye(int(N),batch_shape=[batch],dtype=tf.float32)
        oe_tensor=one_tensor-eye_tensor

        A=tf.abs(tf.multiply(A,oe_tensor))
        
        x = tf.reshape(x, [batch,N,M*C]) 
        


        param = self._definedTuckerWeight_variable('param1',[1, 1, M*C], name, regularization=False)
        param = tf.tile(param, multiples=[batch, N, 1])
        
        
        
        x_mid=tf.multiply(x, param)



              


        x_t=tf.transpose(x, [0,2,1])
        similar=tf.matmul(x_mid,x_t)

        similar=tf.multiply(similar,oe_tensor)
        
        new_A=tf.nn.relu(similar)+A
        new_A=tf.abs(new_A)
        A_degree=tf.clip_by_value(tf.reduce_sum(new_A,axis=2), 0.00000001,10000000)
        d = 1 / A_degree
        d=tf.matrix_diag(d)
        new_A=tf.matmul(d, new_A)
        
        one_tensor=tf.ones(shape=[batch,N,N], dtype="float32")
        eye_tensor=tf.eye(int(N),batch_shape=[batch],dtype=tf.float32)
        oe_tensor=one_tensor-eye_tensor

        new_A=eye_tensor-tf.multiply(new_A,oe_tensor)
        return new_A
    
    def gcnPooling_V2(self, X, A, idx, scope, pooling_ratio = 0.8):
        batch, N, M, C = X.get_shape()
        X=tf.reshape(X, [batch*N, M*C])
        
        feat = []
        adj = []
        indices = []
        
        value=None
        topk_idx=None 

        
        for g in range(batch):
          gg = tf.reshape(tf.where(tf.equal(idx, g)), [-1])
          X_g = tf.gather(X, indices=gg)
          A_g = tf.reshape(tf.gather(A, g, axis=0), [N,N])
          idx_g = tf.gather(idx, gg, axis=0)
          
          if pooling_ratio < 1:
            to_keep = tf.cast(tf.math.ceil(int(N)*pooling_ratio), dtype=tf.int32)
          else:
            to_keep = N

          A_degree=tf.clip_by_value(tf.reduce_sum(A_g,axis=1), 0.00000001,10000000)
          d = 1 / A_degree
          d = tf.diag(d)
          proj=tf.eye(int(N))-tf.matmul(d, A_g)
         
          p_score=tf.reduce_sum(tf.abs(tf.matmul(proj,X_g)),axis=1)
          value, topk_idx = tf.nn.top_k(p_score, to_keep)
          
          retainIdx, retainI=tf.nn.top_k(topk_idx, to_keep)
          X_g = tf.gather(X_g, retainIdx, axis=0)
          
          A_g = tf.gather(tf.gather(A_g, retainIdx, axis=0),  retainIdx, axis=1)
          idx_g = tf.gather(idx_g, retainIdx, axis=0)
          
          
          feat.append(X_g) 
          adj.append(A_g)
          indices.append(idx_g)
          

        

        X_stack = tf.concat(feat, axis=0)
        A_diag = tf.concat(adj, axis=0)
        idx_seq = tf.concat(indices, axis=-1)
        
        A_diag=tf.reshape(A_diag, [batch, to_keep, to_keep])
        X_stack=tf.reshape(X_stack, [batch, -1, M, C])
        return X_stack,A_diag,idx_seq
    
    def gcnPooling_V3(self, X, A, idx, scope, pooling_ratio = 0.8):
        batch, N, M, C = X.get_shape()
        X=tf.reshape(X, [batch, N, M*C])
        
        feat = []
        adj = []
        indices = []
        
        value=None
        topk_idx=None 

        
        for g in range(batch):

          X_g = tf.gather(X, g, axis=0)
          A_g = tf.gather(A, g, axis=0)
          idx_g = tf.gather(idx, g, axis=0)
          
          if pooling_ratio < 1:
            to_keep = tf.cast(tf.math.ceil(int(N)*pooling_ratio), dtype=tf.int32)
          else:
            to_keep = N

          A_degree=tf.clip_by_value(tf.reduce_sum(tf.eye(int(N))+A_g,axis=1), 0.00000001,10000000)
          d = 1 / A_degree
          d = tf.diag(d)
          proj=tf.eye(int(N))-tf.matmul(d, A_g) 
         
          p_score=tf.reduce_sum(tf.abs(tf.matmul(proj,X_g)),axis=1)
          value, topk_idx = tf.nn.top_k(p_score, to_keep)
          
          topk_idx=topk_idx*(-1)
          retainIdx, retainI=tf.nn.top_k(topk_idx, to_keep)
          retainIdx=tf.abs(retainIdx)
          X_g = tf.gather(X_g, retainIdx, axis=0)
          
          A_g = tf.gather(tf.gather(A_g, retainIdx, axis=0),  retainIdx, axis=1)


          idx_g = tf.gather(idx_g, retainIdx, axis=0)
          
          feat.append(X_g) 
          adj.append(A_g)
          indices.append(idx_g)
          


        X_stack = tf.concat(feat, axis=0)
        A_diag = tf.concat(adj, axis=0)
        idx_seq = tf.concat(indices, axis=-1)
        
        A_diag=tf.reshape(A_diag, [batch, to_keep, to_keep])
        X_stack=tf.reshape(X_stack, [batch, -1, M, C])
        idx_seq=tf.reshape(idx_seq, [batch, -1])
        return X_stack,A_diag,idx_seq
    
    def gcnPooling_V4(self, X, A, idx, scope, pooling_ratio = 0.8):
        batch, N, M, C = X.get_shape()
        X=tf.reshape(X, [batch, N, M*C])
        
        feat = []
        adj = []
        indices = []
        
        value=None
        topk_idx=None 

        
        for g in range(batch):

          X_g = tf.gather(X, g, axis=0)
          A_g = tf.gather(A, g, axis=0)
          idx_g = tf.gather(idx, g, axis=0)
          
          if pooling_ratio < 1:
            to_keep = tf.cast(tf.math.ceil(int(N)*pooling_ratio), dtype=tf.int32)
          else:
            to_keep = N

          A_degree=tf.clip_by_value(tf.reduce_sum(A_g,axis=1), 0.00000001,10000000)
          d = 1 / A_degree
          d = tf.diag(d)
          proj=tf.eye(int(N))-tf.matmul(d, A_g)
         
          p_score=tf.reduce_sum(tf.abs(tf.matmul(proj,X_g)),axis=1)
          value, topk_idx = tf.nn.top_k(p_score, to_keep)
          

          X_g = tf.gather(X_g, topk_idx, axis=0)
          
          A_g = tf.gather(tf.gather(A_g, topk_idx, axis=0),  topk_idx, axis=1)


          idx_g = tf.gather(idx_g, topk_idx, axis=0)
          
          feat.append(X_g) 
          adj.append(A_g)
          indices.append(idx_g)
          


        X_stack = tf.concat(feat, axis=0)
        A_diag = tf.concat(adj, axis=0)
        idx_seq = tf.concat(indices, axis=-1)
        
        A_diag=tf.reshape(A_diag, [batch, to_keep, to_keep])
        X_stack=tf.reshape(X_stack, [batch, -1, M, C])
        idx_seq=tf.reshape(idx_seq, [batch, -1])
        return X_stack,A_diag,idx_seq

    def gcnPooling_V5(self, X, A, idx, scope, pooling_ratio = 0.8):
        batch, N, M, C = X.get_shape()
        X=tf.reshape(X, [batch, N, M*C])
        
        feat = []
        adj = []
        indices = []
        
        value=None
        topk_idx=None 

        
        for g in range(batch):

          X_g = tf.gather(X, g, axis=0)
          A_g = tf.gather(A, g, axis=0)
          idx_g = tf.gather(idx, g, axis=0)
          
          if pooling_ratio < 1:
            to_keep = tf.cast(tf.math.ceil(int(N)*pooling_ratio), dtype=tf.int32)
          else:
            to_keep = N

          A_degree=tf.clip_by_value(tf.reduce_sum(tf.eye(int(N))-A_g,axis=1), 0.00000001,10000000)
          d = 1 / A_degree
          d = tf.diag(d)
          proj=tf.eye(int(N))-tf.matmul(d, tf.eye(int(N))-A_g) 
         
          p_score=tf.reduce_sum(tf.abs(tf.matmul(proj,X_g)),axis=1)
          value, topk_idx = tf.nn.top_k(p_score, to_keep)
          
          topk_idx=topk_idx*(-1)
          retainIdx, retainI=tf.nn.top_k(topk_idx, to_keep)
          retainIdx=tf.abs(retainIdx)
          X_g = tf.gather(X_g, retainIdx, axis=0)
          
          A_g = tf.gather(tf.gather(A_g, retainIdx, axis=0),  retainIdx, axis=1)


          idx_g = tf.gather(idx_g, retainIdx, axis=0)
          
          feat.append(X_g) 
          adj.append(A_g)
          indices.append(idx_g)
          


        X_stack = tf.concat(feat, axis=0)
        A_diag = tf.concat(adj, axis=0)
        idx_seq = tf.concat(indices, axis=-1)
        
        A_diag=tf.reshape(A_diag, [batch, to_keep, to_keep])
        X_stack=tf.reshape(X_stack, [batch, -1, M, C])
        idx_seq=tf.reshape(idx_seq, [batch, -1])
        return X_stack,A_diag,idx_seq
    
    def gCNConv(self, X, A, Cout_spatial, scope):
         batch, N, M, Cin = X.get_shape()
         
         convKernel1 = self._definedTuckerWeight_variable('convKernel1',[Cin, Cout_spatial], scope, regularization=False)

    
         x_inputlist_spatialLayer=tf.split(X, M, 2)
         x_outputList_spatialLayer=[]
         
         A=tf.eye(int(N))+A
         A_degree=tf.clip_by_value(tf.reduce_sum(A,axis=2), 0.00000001,10000000)
         d=1 / A_degree
         d=tf.matrix_diag(d)
         A=tf.matmul(d, A)

         
         for tempIdx in range(M):
             X=tf.reshape(x_inputlist_spatialLayer[tempIdx],[batch, N, Cin])
             res = self.dot(A, X, sparse=False)

             res=tf.reshape(res,[batch, N, Cin])
             res = tf.tensordot(res, convKernel1, axes=(2,0))

        


            


             res=tf.nn.relu(res)
             

             
             res=tf.reshape(res,[batch, N, Cout_spatial])
             x_outputList_spatialLayer.append(res)
         
         print('x_outputList_spatialLayer:',x_outputList_spatialLayer)
         x_output_spatialLayer=tf.stack(x_outputList_spatialLayer, axis=2)
         return x_output_spatialLayer


    def create_batch_elements(self, X, A):


      X_stack = np.vstack(X)
      A_diag = sp.block_diag(A)
      A_diag = self.convert_sparse_matrix_to_sparse_tensor(A_diag)
      A_diag = tf.sparse.reorder(A_diag)
      n_nodes = np.array([a.shape[0] for a in A])




      graph_idx = np.repeat(np.arange(len(n_nodes)), n_nodes)
      

      return X_stack, A_diag, graph_idx
    
    def avgpool1(self, x, idx):
        res = tf.math.segment_mean(x, idx) 
        return res
    
    def maxpool1(self, x, idx):
        res = tf.math.segment_max(x, idx)        
        return res
    def sumpool(self, inputs, ):
        nodes_feat = inputs[0]
        idx = inputs[1]
        num_seg = inputs[2]
        res = tf.math.unsorted_segment_sum(nodes_feat, idx, num_segments=num_seg)
        return res

    def gonv_new(self, x, new_A, spatialKernel):
        batch, N, M, Cin = x.get_shape()
        x_inputlist_spatialLayer=tf.split(x,M,2)
        x_outputList_spatialLayer=[]
        for tempIdx in range(M):
            temp_input=tf.reshape(x_inputlist_spatialLayer[tempIdx],[batch, N, Cin])
            temp_output=self.chebyshev5ForBaseline(temp_input, new_A, 2, 1, spatialKernel)
            x_outputList_spatialLayer.append(temp_output)
        x_output_spatialLayer=tf.stack(x_outputList_spatialLayer, axis=2)
        return x_output_spatialLayer
    
    def onehotLayer(self,x,region):
        batch, M = x.get_shape()
        
        x=tf.one_hot(x, region)
        print('x:',x)
        return x
    
    def getLocalAfromInput(x):
        localA=None
        return localA
        
    def _inference(self, x): 

        
        x=self.onehotLayer(x, 768)

        x=tf.transpose(x, [0,2,1])
        batch, N, M = x.get_shape()
        x=tf.reshape(x,[batch, N, M, 1])   
        
        originX=tf.identity(x)
        regionList=[]
        for i in range(batch):
            regionList.append(int(N))
                
        n_nodes = np.array(regionList)
        print('np.arange(len(n_nodes)):',np.arange(len(n_nodes)))
        
        graph_idx = np.repeat(np.arange(int(N)), int(batch))
        graph_idx=np.reshape(graph_idx,[N,batch])
        graph_idx=np.transpose(graph_idx)

        graph_idx=tf.convert_to_tensor(graph_idx)
        
        initialAjacency=self.AjacentMatrix


        
        originA=[]
        for i in range(batch):
            originA.append(initialAjacency)
        originA=tf.stack(originA, axis=0)
        print('originA 1:',originA)
        


        

        new_A=tf.abs(originA)
        A_degree=tf.clip_by_value(tf.reduce_sum(new_A,axis=2), 0.00000001,10000000)
        d = 1 / A_degree
        d=tf.matrix_diag(d)
        new_A=tf.matmul(d, new_A)
        new_A=tf.cast(new_A, tf.float32)
        
        one_tensor=tf.ones(shape=[batch,N,N], dtype="float32")
        eye_tensor=tf.eye(int(N),batch_shape=[batch],dtype=tf.float32)
        oe_tensor=one_tensor-eye_tensor

        new_A=eye_tensor-tf.multiply(new_A,oe_tensor)
        print('new_A:',new_A)
        
        scope='scaleTime'
        temporal1Kernel=self._definedTuckerWeight_variable('xInput1Kernel', [1, 5, 1, 3], scope, regularization=False)
        x=tf.nn.conv2d(originX, 
                        temporal1Kernel,
                        strides=[1,1,1,1], 
                        padding='SAME')



        x = tf.nn.max_pool(x,[1,1,4,1],[1,1,3,1],padding='VALID')
        
        print('x:',x)        
        temporal2Kernel=self._definedTuckerWeight_variable('xInput2Kernel', [1, 5, 3, 6], scope, regularization=False)
        x=tf.nn.conv2d(x, 
                        temporal2Kernel,
                        strides=[1,1,1,1], 
                        padding='SAME')
        

        x = tf.nn.max_pool(x,[1,1,4,1],[1,1,3,1],padding='VALID')
        
        print('x:',x)
        
        temporal3Kernel=self._definedTuckerWeight_variable('xInput3Kernel', [1, 5, 6, 3], scope, regularization=False)
        x=tf.nn.conv2d(x, 
                        temporal3Kernel,
                        strides=[1,1,1,1], 
                        padding='SAME')
        
        x0 = tf.nn.max_pool(x,[1,1,4,1],[1,1,3,1],padding='VALID')
        
        print('x0:',x0)

        




        
        print('1 gcnconv')



        A_diag0=self.redefine_A_6(x0, new_A, 'rede1')
        spatialKernel1 = self._definedTuckerWeight_variable('gcnKernel1',[1, 3, 3], ('encodeDecoderConv{}').format(1), regularization=False)
        x1=self.gonv_new(x0, A_diag0, spatialKernel1)     
        x1,A_diag1, idx_seq1=self.gcnPooling_V5(x1, A_diag0, graph_idx, ('gcnPooling{}').format(1), 0.6)
        
        print('2 gcnconv',x1)
        A_diag2=self.redefine_A_6(x1, A_diag1, 'rede2')

        spatialKernel2 = self._definedTuckerWeight_variable('gcnKernel2',[1, 3, 3], ('encodeDecoderConv{}').format(2), regularization=False)
        x2=self.gonv_new(x1, A_diag2, spatialKernel2)    
        x2, A_diag2, idx_seq2=self.gcnPooling_V5(x2, A_diag2, idx_seq1, ('gcnPooling{}').format(2), 0.6)

        
        print('3 gcnconv',x2)
        A_diag3=self.redefine_A_6(x2, A_diag2, 'rede3')
        spatialKernel3 = self._definedTuckerWeight_variable('gcnKernel3',[1, 3, 3], ('encodeDecoderConv{}').format(3), regularization=False)
        x3=self.gonv_new(x2, A_diag3, spatialKernel3)    
        x3, A_diag3, idx_seq3=self.gcnPooling_V5(x3, A_diag3, idx_seq2, ('gcnPooling{}').format(3), 0.6)
         
 
        x=tf.concat([x0,x1,x2,x3],axis=1)
        batch, N, M, C = x0.get_shape()

        idx = tf.concat([graph_idx, idx_seq1, idx_seq2, idx_seq3], axis=1)
        
        inputs=tf.split(x,batch,0)
        indexs=tf.split(idx,batch,0) 
        
        ressum_List=[]
        resmax_List=[]
        
        
        for i in range(batch):
            ressum_List.append(tf.math.unsorted_segment_sum(inputs[i], indexs[i], num_segments=N))
            resmax_List.append(tf.math.unsorted_segment_max(inputs[i], indexs[i], num_segments=N))
        
        res_sum=tf.stack(ressum_List, axis=0)
        res_max=tf.stack(resmax_List, axis=0)
        res_sum=tf.reshape(res_sum,[batch, N*M*C])
        res_max=tf.reshape(res_max,[batch, N*M*C])
        
        x0=tf.reshape(x0,[batch, N*M*C])
        
        
        idx=idx/768
        
        [x_mean0, x_varia0] = tf.nn.moments(x0, axes=0)
        [x_meanS, x_variaS] = tf.nn.moments(res_sum, axes=0)
        [x_meanM, x_variaM] = tf.nn.moments(res_max, axes=0)
        
        offset = 0
        scale = 0.1
        vari_epsl = 0.0001

        x0 = tf.nn.batch_normalization(x0, x_mean0, x_varia0, offset,scale,vari_epsl)
        res_sum = tf.nn.batch_normalization(res_sum, x_meanS, x_variaS, offset,scale,vari_epsl)
        res_max = tf.nn.batch_normalization(res_max, x_meanM, x_variaM, offset,scale,vari_epsl)
        
        idx=tf.cast(idx, tf.float32)
        
        x = tf.concat([x0, res_max, res_sum], axis=1)
        with tf.variable_scope('fc'):



            x = self.fc(x, 30, 'fc2', relu=False)
            x = tf.nn.softmax(x)
        return x,x,idx
    
    def add_k(self,inputs, k):
        return inputs, k+1
    
    def keep_k(self,inputs, k):
        return inputs,k
    
    def recoverTensor(self,inputs, idx, x0):
        batch,N,M,C=x0.get_shape()
        batch,N_1,M,C=inputs.get_shape()
        
        inputs=tf.reshape(inputs,[batch, N_1, M*C])
        zeros=tf.Variable(tf.zeros([M*C]))
        
        idx=tf.to_int32(idx)
        
        tempzero=tf.to_int32(tf.zeros([batch, int(N-N_1)]))
        idx= tf.concat([idx, tempzero], axis=1)
        
        outputList=[] 
        for i in range(batch): 
            k=0
            for j in range(N):
                tempValue,k = tf.cond(tf.equal(j,idx[i,k]), lambda: self.add_k(inputs[i,k,:],k), lambda: self.keep_k(zeros,k))
                outputList.append(tempValue)
                
            
        outputs=tf.stack(outputList, axis=0)
        outputs=tf.reshape(outputs, [batch, N*M*C])
        return outputs
