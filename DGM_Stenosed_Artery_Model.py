import tensorflow as tf
import numpy as np
class DGML:

    def __init__(self, nhl, ydim, xdim, initer, bias_initer, drate_tensor):

        self.initer = initer
        self.bias_initer = bias_initer
        self.nhl = nhl
        self.ydim = ydim
        self.xdim = xdim
        self.drate = drate_tensor
        self.Uz=tf.get_variable(name='Uz', shape=[ydim, nhl, xdim], dtype=tf.float64, trainable = True, initializer=self.initer)
        self.Wz=tf.get_variable(name='Wz', shape=[ydim, nhl, nhl] , dtype=tf.float64, trainable = True, initializer=self.initer)
        self.Bz=tf.get_variable(name='Bz', shape=[ydim, nhl, ]    , dtype=tf.float64, trainable = True, initializer=self.initer)

        self.Ug=tf.get_variable(name='Ug', shape=[ydim, nhl, xdim], dtype=tf.float64, trainable = True, initializer=self.initer)
        self.Wg=tf.get_variable(name='Wg', shape=[ydim, nhl, nhl] , dtype=tf.float64, trainable = True, initializer=self.initer)
        self.Bg=tf.get_variable(name='Bg', shape=[ydim, nhl, ]    , dtype=tf.float64, trainable = True, initializer=self.initer)

        self.Ur=tf.get_variable(name='Ur', shape=[ydim, nhl, xdim], dtype=tf.float64, trainable = True, initializer=self.initer)
        self.Wr=tf.get_variable(name='Wr', shape=[ydim, nhl, nhl] , dtype=tf.float64, trainable = True, initializer=self.initer)
        self.Br=tf.get_variable(name='Br', shape=[ydim, nhl, ]    , dtype=tf.float64, trainable = True, initializer=self.initer)

        self.Uh=tf.get_variable(name='Uh', shape=[ydim, nhl, xdim], dtype=tf.float64, trainable = True, initializer=self.initer)
        self.Wh=tf.get_variable(name='Wh', shape=[ydim, nhl, nhl] , dtype=tf.float64, trainable = True, initializer=self.initer)
        self.Bh=tf.get_variable(name='Bh', shape=[ydim, nhl, ]    , dtype=tf.float64, trainable = True, initializer=self.initer)


    def __call__(self, X, S):
        x = X
        S = S
        B = tf.shape(X)[0]
        def E(W):
            multiply = tf.concat([[B],tf.ones([tf.rank(W)], tf.int32)] ,0)
            shape = tf.concat([[B], tf.shape(W)], 0)
            return tf.reshape(tf.tile(tf.expand_dims(W, 0), multiply), shape)
        self.Z = tf.nn.dropout(tf.nn.relu(tf.einsum('bijk,bk->bij', E(self.Uz), x) + tf.einsum('bijk,bik->bij', E(self.Wz), S) + E(self.Bz), name='Z'),
                                seed=0.0, rate=self.drate)
        self.G = tf.nn.dropout(tf.nn.relu(tf.einsum('bijk,bk->bij', E(self.Ug), x) + tf.einsum('bijk,bik->bij', E(self.Wg), S) + E(self.Bg), name='G'),
                               seed=0.0, rate=self.drate)
        self.R = tf.nn.dropout(tf.nn.relu(tf.einsum('bijk,bk->bij', E(self.Ur), x) + tf.einsum('bijk,bik->bij', E(self.Wr), S) + E(self.Br), name='R'), 
                               seed=0.0, rate=self.drate)

        self.H = tf.add_n([tf.einsum( 'bijk,bk->bij', E(self.Uh), x), tf.einsum('bijk,bik->bij', E(self.Wh), tf.multiply(S, self.R)), E(self.Bh)], name='H')

        S_out = tf.add(tf.multiply(self.Z , S), tf.multiply( (1.0 - self.G) , self.H ), name='S')
        return S_out

class input_layer:
    def __init__(self, nhl, ydim, xdim, initer, bias_initer):
        self.initer = initer
        self.bias_initer = bias_initer
        with tf.variable_scope('Input_layer'):
            self.W1=tf.get_variable(name='W', shape=[ydim, nhl, xdim], dtype=tf.float64, trainable = True, initializer=self.initer)
            self.B1=tf.get_variable(name='B', shape=[ydim, nhl, ],          dtype=tf.float64, trainable = True, initializer=self.initer)

    def __call__(self, x):
        B = tf.shape(x)[0]
        def E(W):
            multiply = tf.concat([[B],tf.ones([tf.rank(W)], tf.int32)] ,0)
            shape = tf.concat([[B], tf.shape(W)], 0)
            return tf.reshape(tf.tile(tf.expand_dims(W, 0), multiply), shape)

        with tf.variable_scope('Input_layer_Ops'):
            return tf.nn.relu(tf.einsum('bijk,bk->bij', E(self.W1), x) + E(self.B1), name='S1')

class output_layer:
    def __init__(self, nhl, ydim, xdim, initer, bias_initer):
        self.initer = initer
        self.bias_initer = bias_initer
        with tf.variable_scope('Output_Layer'):
            self.W2=tf.get_variable(name='W', shape=[ydim, nhl], dtype=tf.float64, trainable = True, initializer=self.initer)
            self.B2=tf.get_variable(name='B', shape=[ydim,]   , dtype=tf.float64, trainable = True, initializer=self.initer)

    def __call__(self, S):
        B = tf.shape(S)[0]
        def E(W):
            multiply = tf.concat([[B],tf.ones([tf.rank(W)], tf.int32)] ,0)
            shape = tf.concat([[B], tf.shape(W)], 0)
            return tf.reshape(tf.tile(tf.expand_dims(W, 0), multiply), shape)

        with tf.variable_scope('Output_Layer_Ops'):
            return tf.add(tf.einsum('bij,bij->bi', E(self.W2) , S), self.B2, name='Y')

class Raw_DGM_Model:
    def __init__(self, nhl, ydim, xdim, n_layers, stenosis, drate_tensor): # nhl is the Number of Hidden Layers

        self.nhl = nhl
        self.xdim = xdim
        self.ydim = ydim
        self.n_layers = n_layers
        self.layers = []
#        self.initer=tf.contrib.layers.xavier_initializer(dtype=tf.float64)
        self.initer = tf.initializers.random_uniform(-0.001, 0.001, seed=0.0, dtype=tf.float64)
        self.bias_initer = tf.initializers.zeros()
        self.drate = drate_tensor
        self.eta = tf.cast(stenosis, tf.float64, name='Degree_of_Stenosis')

        with tf.variable_scope('Model_vars'):
            for i in range(self.n_layers):
                if i == 0:
                    self.layers.append(input_layer(self.nhl,
                                                   self.ydim,
                                                   self.xdim,
                                                   self.initer,
                                                   self.bias_initer))

                elif i == self.n_layers - 1:
                    self.layers.append(output_layer(self.nhl,
                                                    self.ydim,
                                                    self.xdim,
                                                    self.initer,
                                                    self.bias_initer))

                else:
                    with tf.variable_scope(f'DGM_Layer{i}'):
                        self.layers.append(DGML(self.nhl,
                                                self.ydim,
                                                self.xdim,
                                                self.initer,
                                                self.bias_initer,
                                                self.drate))
        print('Model Deloyed')
        
    def top_bound_y(self, X_Y):
        with tf.variable_scope('Stenosis_Trap'):
            X = X_Y[:, 0]
            Y = X_Y[:, 1]
            pi = tf.constant(np.pi, tf.float64)
            
            def cos_f(X_in):
                X_dash = (pi / 4.0) * (X_in - 3.0)
                f = -0.5 + tf.square(tf.cos(X_dash)) + tf.sqrt(1.0 - self.eta) * tf.square(tf.sin(X_dash)) 
                return f
 
#            def stenosis_(X_Y):
#                X = X_Y[0]
#                Y = X_Y[1]
                            
#                return tf.cond(pred=tf.stop_gradient(tf.squeeze(tf.logical_and(tf.greater(X, -5.0), tf.less(X, -1.0)))),
#                               true_fn=lambda: Y - cos_f(X),
#                               false_fn=lambda: Y - 0.5)
                
#            stenosis = tf.map_fn(stenosis_, X_Y, name='Stenosis_Trap')
            
#            closest_quadratic = Y - (0.00433 + 0.4392*X + 14.64*X**2.0) for x and y in metres, pure NS
#            closest_quadratic = Y - (4.33 + 0.4392*X + 0.01464*X**2) for x and y in mm pure NS
#            closest_quadratic = Y - (0.866125 + 0.43935*X + 0.073225*X**2.0)
#            out =  closest_quadratic + tf.stop_gradient(stenosis - closest_quadratic)
            constraint_exp =  tf.where(condition=tf.squeeze(tf.logical_and(tf.greater(X, -5.0), tf.less(X, -1.0))),
                                        x= cos_f(X),
                                        y= tf.fill(tf.shape(X), tf.constant(0.5, tf.float64)))

        return Y - constraint_exp
    
    def __call__(self, inp):
        B = tf.shape(inp)[0]
        def E(W):
            multiply = tf.concat([[B],tf.ones([tf.rank(W)], tf.int32)] ,0)
            shape = tf.concat([[B], tf.shape(W)], 0)
            return tf.reshape(tf.tile(tf.expand_dims(W, 0), multiply), shape)

        with tf.variable_scope('Model_Ops'):
            self.S = []
            self.x = tf.identity(inp, name='X' )
            self.B = B
            X = self.x[:, 0]
            Y = self.x[:, 1]

            for i, layer in enumerate(self.layers):
                if i == 0:
                    with tf.variable_scope(f'Input_Layer_Ops'):
                        self.S.append(layer(self.x, ))

                elif i == self.n_layers - 1:     
                    with tf.variable_scope(f'Output_Layer_Ops'):
                        self.S.append(layer(self.S[i - 1]))

                else :
                    with tf.variable_scope(f'DGML{i}_Ops'):
                        self.S.append(layer(self.x, self.S[i - 1]))

            trap_fn = self.top_bound_y(self.x)
            self.trap = tf.cast(tf.stack([tf.squeeze(trap_fn),
                                          tf.squeeze((Y + 0.5)*(X + 6.0)*trap_fn),
                                          tf.squeeze(6.0-X)], axis=1), tf.float64)

            self.UVP =  tf.multiply(self.S[-1], self.trap)
            u_val, v_val, p_val = tf.split(self.UVP, 3, axis=1)
#            self.output = tf.concat([u_val, v_val, tf.square(p_val)], axis=) # To limit P to be always positive
#        return [u_val, v_val, tf.square(p_val)]
        return [u_val, v_val, p_val]

    
#        return self.S[-1]
##
#def E(W):
#    multiply = tf.concat([[5],tf.ones([tf.rank(W)], tf.int32)] ,0)
#    shape = tf.concat([[5], tf.shape(W)], 0)
#    return tf.reshape(tf.tile(tf.expand_dims(W, 0), multiply), shape)

#with tf.device('/device:CPU:0'):
#a=Raw_DGM_Model(2, 3, 2, 5, 0.5)
##    check = a(tf.ones([5, 2], dtype=tf.float64))
##    d = tf.gradients(a(check), check)
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
##sess.run(check)
##
##
##print(sess.run(d))
### Cosine function
## Spline approx quadratic
## stenosis area limits check
## unconditioal upper trap value
## trap function multiplication
#check = a(tf.constant([[-2.952, 0.1905],
#                       [-2.952, 0.2368],
#                       [-2.952, 0.2894],
#                       [-2.952, 0.3421]], dtype=tf.float64))

