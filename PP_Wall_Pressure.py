import tensorflow as tf
from pyevtk.hl import pointsToVTK as p_vtk
import Models.DGM_Stenosed_Artery_Model as current_model
import numpy as np
import time, os, sys
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MINLOG_LEVEL']='3'

dev = 'device:CPU:0'
batch_size = 32
def Wall_Pressure(record):
    global WSS,n,chi
    x = record[:, 0] 
    y = record[:, 1] 
    x_y = tf.stack([x, y], axis=1)
    u, v, p = model(x_y)
    u_x, u_y = tf.split(tf.gradients(u, [x, y], name='Du'), xdim)
    v_x, v_y = tf.split(tf.gradients(v, [x, y], name='Dv'), xdim)
    p_x, p_y = tf.split(tf.gradients(p, [x, y], name='Dp'), xdim)

    [u, v, p,
     u_x, u_y,
     v_x, v_y] = [tf.squeeze(Q) for Q in \
                                   [u, v, p,
                                    u_x, u_y,
                                    v_x, v_y]]
#    rho = 1050.0
    R0 = 0.005
    U0 = 0.0986
    mu_inf = 0.00345
    mu_0 = 0.056

    y_dash = tf.clip_by_value(t=y + 0.5,
                              clip_value_min=0.01,
                              clip_value_max=1.1)

    zeta = mu_0 / mu_inf # Dimensionless
#    alpha = R0 * U0 * rho / mu_inf
    T = R0 / U0
    lamda_dash = 3.313 / T
    
    A1 = u_x**2.0
    A2 = v_y**2.0
    A3 = ((v / y_dash)**2.0)
    A = A1 + A2 + A3

    B1 = v_x
    B2 = u_y
    B = (B1 + B2) ** 2.0
    
    ndgamma = tf.sqrt(2.0* A +  B)
    n = 0.3568
    chi = 1.0 + (zeta - 1.0) * (1.0 + (lamda_dash * ndgamma)**2.0) ** ((n - 1.0)/2.0)
    
    WSS = -1.0 * chi * mu_inf * u_y / T
    return tf.squeeze(tf.stack([p, WSS], axis=1))
    
with tf.device(dev):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    
    raw_wallpoints = np.genfromtxt('./Upper_Boundary_points.csv', delimiter=',', dtype='float')
    np.random.shuffle(raw_wallpoints)
    raw_wallpoints_x_y = raw_wallpoints[:320, [0, 1]]
    x=(raw_wallpoints_x_y[:, 0] - 15.0) / 5.0
    y=(raw_wallpoints_x_y[:, 1] - 2.5) / 5.0
    
    wallpoints_x_y = np.squeeze(np.stack([x, y], axis=1))
    wallpoints_x_y = wallpoints_x_y[wallpoints_x_y[:,0].argsort()]
    
    
    points_set = tf.data.Dataset.from_tensor_slices(wallpoints_x_y).batch(batch_size).make_initializable_iterator()
    points_set_initer = points_set.initializer
    points = points_set.get_next()
    # Hyper-Parameters
    xdim = 2 # x, y
    ydim = 3 # u, v, p
    nhl = 100
    n_dgml = 5
    stenosis = 0.5

    model = current_model.Raw_DGM_Model(nhl, ydim, xdim, n_dgml+2, stenosis, drate_tensor=tf.constant(0.0,tf.float64))
    
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

    batch_data_stack = Wall_Pressure(points)
    
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
        
        
        data=np.zeros([1, 2])
        sess.run(points_set_initer)
        
        
        while True:
            try:
                data_batch = sess.run(batch_data_stack)
                data = np.concatenate([data, data_batch], axis=0)
            except tf.errors.OutOfRangeError:
                break
        data = np.delete(data, [0], axis=0)
        
        print('Processing data...', end='', flush=True)
        
        # Seperate and make contigious
        x, y = [np.squeeze(np.ascontiguousarray(arr)) for arr in np.split(wallpoints_x_y, 2, axis=1)]
        WP, WSS = np.squeeze(np.ascontiguousarray(np.split(data, data.shape[1], axis=1)))
        steps = state_file[state_file.find('l-')+2:]
        
        file = f'./Bucket/Plots/Wall_Pressure'
        wfile = p_vtk(file, x, y, z=np.zeros(x.shape), data={'Wall_Pressure':WP,
                                                             'Wall Shear Stress':WSS})
        time.sleep(5)
        os.rename(wfile, f'./Bucket/Plots/Wall_Pressure-{steps}.vtu')
        
        print(f'Field data Saved to "./Bucket/Plots/Wall_Pressure-{steps}.vtu"')
        if not c: break
    except KeyboardInterrupt:
        break
    
sess.close()
os._exit(1)

