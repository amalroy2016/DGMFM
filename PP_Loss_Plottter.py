import numpy as np
import time, os, sys
import tensorflow as tf
from pyevtk.hl import pointsToVTK as p_vtk
import Models.DGM_Stenosed_Artery_Model as current_model

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def PDE_residue(points):
    # TODO Recasting is not a good solution. FInd a better way
    x = points[:32, 0] 
    y = points[:32, 1] 
    boundary_flags = points[:32, -1] 
    
    x.set_shape([32,])
    y.set_shape([32,])
    boundary_flags.set_shape([32,])
    
    x_y = tf.stack([x, y], axis=1)
    u, v, p = model(x_y)    
    u_x, u_y = tf.split(tf.gradients(u, [x, y], name='Du'), xdim)
    v_x, v_y = tf.split(tf.gradients(v, [x, y], name='Dv'), xdim)
    p_x, p_y = tf.split(tf.gradients(p, [x, y], name='Dp'), xdim)

    [u_xx], [u_yy] = [tf.gradients(u_x, x, name='D2u_Dx2'), tf.gradients(u_y, y, name='D2u_Dy2')]
    [v_xx], [v_yy] = [tf.gradients(v_x, x, name='D2v_Dx2'), tf.gradients(v_y, y, name='D2v_Dy2')]

    [u, v, p,
     u_x, u_y,
     v_x, v_y,
     p_x, p_y,
     u_xx, u_yy,
     v_xx, v_yy] = [tf.squeeze(Q) for Q in \
                                   [u, v, p,
                                    u_x, u_y,
                                    v_x, v_y,
                                    p_x, p_y,
                                    u_xx, u_yy,
                                    v_xx, v_yy]]
        
    with tf.name_scope('PDE_Residues'):

        rho = 1050.0
        R0 = 0.005
        U0 = 0.0986
        mu_inf = 0.00345
        mu_0 = 0.056

        y_dash = tf.clip_by_value(t=y + 0.5,
                                  clip_value_min=0.01,
                                  clip_value_max=1.1)
        
        zeta = mu_0 / mu_inf # Dimensionless
        T = R0 / U0
        alpha = R0 * U0 * rho / mu_inf

        A1 = u_x**2.0
        A2 = v_y**2.0
        A3 = ((v / y_dash)**2.0)
        A = A1 + A2 + A3

        B1 = v_x
        B2 = u_y
        B = (B1 + B2) ** 2.0
        lamda_dash = 3.313 / T
#        ndgamma = tf.sqrt( A + 0.5* B)

        ndgamma = tf.sqrt(2.0* A +  B)
        n = 0.3568
        chi = 1.0 + (zeta - 1.0) * ((1.0 + (lamda_dash * ndgamma)**2.0) ** ((n - 1.0)/2.0))
        chi_x, chi_y = tf.gradients(chi, [x, y], name='D_Chi')

        onebyalpha = 1.0 / alpha
        chi_by_alpha = chi / alpha
        
        continuity = v_y + (v / y_dash) +  u_x
        continuity = tf.identity(continuity, name='Continuity')

        x_residue = v * u_y +  u * u_x + p_x - \
                    \
                    chi_by_alpha*(u_xx + u_y/(y_dash) + u_yy) - \
                    \
                    onebyalpha * (chi_y * (v_x + u_y) + 2.0*chi_x*u_x)

        x_momentum = tf.identity(x_residue, name='X-Momentum')

        y_residue = v * v_y + p_y + u*v_x -\
                    \
                    chi_by_alpha * (v_yy + v_y/(y_dash) + v_xx - v/((y_dash)**2.0)) -\
                    \
                    onebyalpha * (chi_x * (v_x + u_y) + 2.0*chi_y*v_y)

        y_momentum = tf.identity(y_residue, name='Y-Momentum')
        batch_pde_residues = tf.square(tf.stack([continuity, x_momentum, y_momentum], axis=1))
      
        out = tf.concat([tf.expand_dims(tf.squeeze(x), axis=1),
                         tf.expand_dims(tf.squeeze(y), axis=1),
                         tf.expand_dims(u, axis=1),
                         tf.expand_dims(v, axis=1),
                         tf.expand_dims(p, axis=1),
                         batch_pde_residues], axis=1)
    return out


dev = 'device:CPU:0'

with tf.device(dev):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    x0 = tf.constant(-6.0, tf.float64)
    xL = tf.constant(6.0, tf.float64)
    y0 = tf.constant(-0.5, tf.float64)
    yL = tf.constant(0.5, tf.float64)

    Nx=100
    Ny=80

    x = tf.lin_space(x0, xL, Nx)
    y = tf.lin_space(y0, yL, Ny)
    X, Y = tf.meshgrid(x, y)
    points_stack = tf.reshape(tf.stack([X, Y], axis=2), shape=[-1, 2])
    
    eta=tf.constant(0.5, tf.float64)
    def cos_f(X_in):
        X_dash = (3.1415 / 4.0) * (X_in - 3.0)
        f = -0.5 + tf.square(tf.cos(X_dash)) + tf.sqrt(1.0 - eta) * tf.square(tf.sin(X_dash)) 
        return f
    mask = tf.logical_not(tf.logical_and(tf.greater(points_stack[:,1],
                                                    cos_f(points_stack[:,0])),
                                        tf.less(tf.abs(points_stack[:,0] + 3), 2)))
    masked_points_stack = tf.boolean_mask(points_stack, mask, axis=0)
    
    points_set = tf.data.Dataset.from_tensor_slices(masked_points_stack).shuffle(100).batch(32, True).make_initializable_iterator()
    points_set_initer = points_set.initializer
    points = points_set.get_next()
    # Hyper-Parameters
    xdim = 2 # x, y
    ydim = 3 # u, v, p
    nhl = 100
    n_dgml = 5
    stenosis = 0.5

    model = current_model.Raw_DGM_Model(nhl, ydim, xdim, n_dgml+2, stenosis, drate_tensor=tf.constant(0.0, tf.float64))
    
    reader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
                            name='Saver',
                            restore_sequentially=False,
                            max_to_keep=10,
                            keep_checkpoint_every_n_hours=0.2,
                            saver_def=None,
                            builder=None,
                            pad_step_number=True,
                            save_relative_paths=False,
                            filename='model_graph_save')

    batch_data_stack = PDE_residue(points)
    
    print('Generating Loss Graph Done.')
    

restore_latest = 'y' # input('Restore Latest?')
c = input('Enter time for continuous, ENTER for once: ')
while True:
    if c:
        time.sleep(int(c))
    try:
        if restore_latest=='y':
            state_file = tf.train.latest_checkpoint('./Training/Graph_Saves/')
            print(f'Visualizing {state_file}')
        else:
            print('Not Supported')
        reader.restore(sess, state_file)
        data=np.zeros([1, 8])
        sess.run(points_set_initer)
        while True:
            try:
                data_batch = sess.run(batch_data_stack)
                data = np.concatenate([data, data_batch], axis=0)
            except tf.errors.OutOfRangeError:
                break
        data = np.delete(data, [0], axis=0)
        z = np.zeros(shape=[data.shape[0]], dtype=np.float64)
        
        print('Processing data...', end='', flush=True)
        
        # Seperate and make contigious
        [x, y, u, v, p, continuity, x_momentum, y_momentum] = [np.squeeze(np.ascontiguousarray(i)) for i in np.split(data, 8, axis=1)]
        z = np.squeeze(np.ascontiguousarray(z))
        steps = state_file[state_file.find('l-')+2:]
        
        file = f'./Bucket/Plots/Velocity_Loss_Fields'
        wfile = p_vtk(file, x, y, z, data={'U':u,
                                           'V':v,
                                           'P':p,
                                           'Continuity':continuity,
                                           'X_Momentum':x_momentum,
                                           'Y_Momentum':y_momentum})
        time.sleep(5)
        os.rename(wfile, f'./Bucket/Plots/Velocity_Loss_Fields-{steps}.vtu')
        
        print(f'Field data Saved to "./Bucket/Losses/Velocity_Loss_Fields-{steps}.vtu"')
        if not c:
            break
    except KeyboardInterrupt:
        break
    
sess.close()
os._exit(1)

