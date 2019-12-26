colab = False
laptop = True
gCLOUD = False
BLIND = False # NO SPECIFIC TIME LAG NOTED FOR 100BATCh SIZE
# if colab:
#     from google.colab import drive
#     from google.colab import files
#     import sys
#     drive.mount('/content/drive')
#     ground = "/content/drive/My Drive/Colab Notebooks/Work/DGM_Remake/N_Dimensional/training"
#     sys.path.append(ground)
#     %cd '/content/drive/My Drive/Colab Notebooks/Work/DGM_Remake/N_Dimensional/training'
#     !pip install tfmpl
#     !apt-get install python-tk

#     !pip install tensorboardcolab
#   #  !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
#   #  !unzip ngrok-stable-linux-amd64.zip
#     import tensorboardcolab as tbc
#     !mv Graph ./Graph_backedup/$(date +GMT%Y-%m-%d_%H:%M:%S)
#     !mkdir 'Graph'
#     tboard = tbc.TensorBoardColab(graph_path='./Graph')

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'figure.max_open_warning': 0})
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tfmpl
import time
import sys
import os
import datetime
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

"""#Specify Model"""

import Models.DGM_Stenosed_Artery_Model as current_model
np.set_printoptions(linewidth=200, threshold=sys.maxsize)

"""#DASHBOARD"""

print("\nOne Step At a time. We are closer every second!"
      "\n\n\t>>Execution STARTED!<<")
# Options
init_new = input('\nInitialize NEW MODEL?(y/n): ')
use_latest = input('\n\tRestore Latest?(y/n): ') if init_new!='y' else ''
use_default_schedule = input('Use default schedule file?')
# Placements
cpu = '/device:CPU:0'
gpu = '/device:GPU:0'
batch_host = cpu
model_host = cpu
GD_host = cpu
metric_host = cpu

if laptop or gCLOUD:
    datafile = './Training/dataset/DATA_MEAN_CENTERED.tfrecords'
    saver_save_file = './Training/Graph_Saves/Blood_Model'
    summaries_loc = './Training/summaries/'

    print('Datafile : {}'.format(datafile))

elif colab:
    datafile='./dataset/datafile.tfrecords'
    saver_save_file = '/content/drive/My Drive/Colab Notebooks/Work/' +\
                      'DGM_Remake/N_Dimensional/training/model_states/model_vars'
    summaries_loc = '/content/drive/My Drive/Colab Notebooks/Work/' +\
                    'DGM_Remake/N_Dimensional/training/summaries/'

    print('Datafile : {}'.format(datafile))

# Hyper-Parameters
xdim = 2 # x, y
ydim = 3 # u, v, p
nhl = 200
n_dgml = 5# Total number including input and output layers
act = input('\nEnter Summary Tag: ')
N_data_points = 10000
saveGraph=False
    
def gen_summary_tag(act, nhl, n_dgml, N_data_points, batch_size, learn_rate, wts):
    return  f'{act}_nhl{nhl}/{n_dgml}L/N{N_data_points}/b{batch_size}/LR{learn_rate},WTS{wts}'

def plot_model(grid=20, for_summary=True):
    with tf.variable_scope('Plotter'):
        lims = [X0, XL, Y0, YL] = [-6.0, 6.0, -0.5, 0.5]
        x0 = tf.constant(X0, name='x0', dtype=tf.float64)
        xL = tf.constant(XL, name='xL', dtype=tf.float64)
        y0 = tf.constant(Y0, name='y0', dtype=tf.float64)
        yL = tf.constant(YL, name='yL', dtype=tf.float64)

        x_space = tf.linspace(x0, xL, num=grid)
        y_space = tf.linspace(y0, yL, num=grid)
        X, Y = tf.meshgrid(x_space, y_space)
        universe = tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])], axis=1)

#        TODO: make the function so that model is run only once for one point
        U_V_P = model(universe)
#        U_V_P = tf.split(U_V_P, 3, axis=1)
        U, V, P = [tf.reshape(q, [grid, grid]) for q in U_V_P ]
        @tfmpl.figure_tensor
        def plotter(Z):
            def norm(z):
                normal = matplotlib.colors.Normalize()
                normal.autoscale(A=z)
                return normal
            fig = tfmpl.create_figure()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)

            extent = lims
            im = ax1.imshow(Z, origin='lower',
                           interpolation='bilinear',
                           norm=norm(Z),
                           extent=extent,
                           aspect='auto')

            ctr = ax2.contour(Z, levels=[i for i in np.arange(-1, 3, 0.1)],
                             origin='lower',
                             extent=extent)
            ax2.clabel(ctr, inline=1, fontsize=12)
            fig.colorbar(im)
            return fig

        if for_summary:
            # Wrapper is already applied at function definition
            U_image_tensor = plotter(U)
            V_image_tensor = plotter(V)
            P_image_tensor = plotter(P)

            return [U_image_tensor, V_image_tensor, P_image_tensor]

        else :
            print('Plotting..')
            fig_U = plt.figure('U Field')
            fig_V = plt.figure('V Field')
            fig_P = plt.figure('P Field')

            extent = lims
            U_ax1 = fig_U.add_subplot(211, label='U Field')
            U_ax2 = fig_U.add_subplot(212, label='U Field Coutour')
            V_ax = fig_V.add_subplot(111, label='V Field')
            P_ax = fig_P.add_subplot(111, label='P Field')

            U, V, P = sess.run([U, V, P])

            def norm(x):
                normal = matplotlib.colors.Normalize()
                normal.autoscale(A=x)
                return normal
            U_ax1.imshow(U, interpolation='bilinear', norm=norm(U), origin='lower',
                                                                   extent=extent,
                                                                   aspect='auto')
            V_ax.imshow(V, interpolation='bilinear', norm=norm(V), origin='lower',
                                                                   extent=extent,
                                                                   aspect='auto')
            P_ax.imshow(P, interpolation='bilinear', norm=norm(P), origin='lower',
                                                                   extent=extent,
                                                                   aspect='auto')
            CTR=U_ax2.contour(U, levels=10,
                             origin='lower',
                             extent=extent)
            U_ax2.clabel(CTR, inline=1, fontsize=12)
            plt.ion()
            plt.draw()
            plt.pause(0.0001)

            print(' >> Done << ')

@tfmpl.figure_tensor
def scatter_plot(X, Y):
    fig = tfmpl.create_figure()
    ax = fig.add_subplot(111)
    ax.scatter(X, Y)
    fig.tight_layout()
    return fig

def decode_and_iterate(datafile, batch_size_to_init):
    def decode_data(record):
        feature_key = {'X': tf.FixedLenFeature([12], tf.float32)}
        parsed = tf.parse_single_example(record, feature_key)
        parsed_record = tf.cast(parsed['X'], tf.float64)
        return parsed_record

    print('\tSetting pipeline... ', end='', flush=True)

    # Read TFRecord files and parsing
    dataset = tf.data.TFRecordDataset(datafile).take(N_data_points).cache().repeat(-1)
    parser = tf.data.experimental.map_and_batch(map_func=decode_data,
                                                batch_size=batch_size_to_init,
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)# )

    data_batch_iter = dataset.apply(parser).shuffle(64).prefetch(2*batch_size_to_init).make_initializable_iterator()
    if not BLIND:
        next_data_batch_ = data_batch_iter.get_next()

        tf.summary.image(name='Batch_Scatter',
                         tensor=scatter_plot(tf.slice(next_data_batch_, [0, 0], [-1, 1]),
                                             tf.slice(next_data_batch_, [0, 1], [-1, 1])))

        next_data_batch = tf.identity(next_data_batch_)

    else: next_data_batch = data_batch_iter.get_next()
    print('[Done.]')

    return next_data_batch, data_batch_iter
# %%
def feed_forward(record, MODE):
#    global u #, v, p, u_x, u_y, v_x, v_y, p_x, p_y, continuity,\
#    x_momentum, y_momentum, b_residues, b_residue_tensor, pde_residues, mask,\
#    truth_full_stack, pred_full_stack, pred_stack_M, truth_stack_M
    
    # TODO Recasting is not a good solution. FInd a better way
    x = record[:, 0] 
    y = record[:, 1] 
    boundary_flags = record[:, -1] 
#    x.set_shape([b_size, ])
#    y.set_shape([b_size, ])
    
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

    with tf.name_scope('Boundary_Residues'):
        
        truth_full_stack = tf.slice(record, [0, 2], [-1, 9])
        pred_full_stack = tf.stack([u, v, p, u_x, u_y, v_x, v_y, p_x, p_y], axis=1)
        mask = tf.logical_not(tf.is_nan(truth_full_stack))
        truth_locs = tf.where(mask)
        
        truth_stack_M = tf.boolean_mask(truth_full_stack, mask)
        pred_stack_M = tf.boolean_mask(pred_full_stack, mask)
        b_residues_stack = tf.squared_difference(pred_stack_M, truth_stack_M)
        
        b_residues_stack = tf.scatter_nd(truth_locs, 
                                      tf.squared_difference(pred_stack_M, truth_stack_M),
                                      shape=[b_size, 9])
        sumallbounds_residue_batch_stack = tf.reduce_sum(b_residues_stack, axis=1, keep_dims=True)
        
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
        alpha = R0 * U0 * rho / mu_inf
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
        chi_x, chi_y = tf.gradients(chi, [x, y], name='D_Chi')

        onebyalpha = 1.0 / alpha
        chi_by_alpha = chi / alpha
        
        continuity = v_y + (v / y_dash) +  u_x
        continuity = tf.identity(continuity, name='Continuity')

        x_residue = v * u_y + u * u_x + p_x - \
                    \
                    chi_by_alpha*(u_xx + u_y/(y_dash) + u_yy) - \
                    \
                    onebyalpha * (chi_y * (v_x + u_y) + 2.0*chi_x*u_x)
                    
        x_momentum = tf.identity(x_residue, name='X-Momentum')

        y_residue = v * v_y + u * v_x  + p_y - \
                    \
                    chi_by_alpha * (v_xx + v_y/(y_dash) + v_yy - v/((y_dash)**2.0)) -\
                    \
                    onebyalpha * (chi_x * (v_x + u_y) + 2.0*chi_y*v_y)
                    
        y_momentum = tf.identity(y_residue, name='Y-Momentum')
        
        batch_shape = [b_size, 3]

#        batch_full_pde_residues = tf.square(tf.stack([continuity,
#                                                      tf.zeros([b_size], tf.float64),
#                                                      tf.zeros([b_size], tf.float64)],
#                                                      axis=1))

#        
    batch_full_pde_residues = tf.square(tf.stack([continuity, x_momentum, y_momentum], axis=1))

    
    mask_momentum = tf.broadcast_to([True, False, False], batch_shape)
    continuity_res = tf.boolean_mask(tensor=batch_full_pde_residues,
                                                mask=mask_momentum)
    masked_batch_full_pde_residues = tf.scatter_nd(tf.where(mask_momentum),
                                                   continuity_res,
                                                   shape=batch_shape)
    
    batch_pde_residues = \
    tf.squeeze(tf.boolean_mask(tensor=tf.stack([batch_full_pde_residues,
                                                 masked_batch_full_pde_residues],axis=0),
                                mask=tf.cond(pred=tf.equal(MODE, 'NEWTONIAN'),
                                             true_fn= lambda:[False,
                                                              True],
                                             false_fn= lambda:[True,
                                                               False])))
    flagged_mini_batch_squared_losses_stack = \
    \
    tf.where(condition=tf.equal(boundary_flags, 1.0),
             x=tf.concat([tf.zeros(batch_shape,tf.float64),
                         sumallbounds_residue_batch_stack], axis=1),
             y=tf.concat([batch_pde_residues,
                         tf.zeros([batch_shape[0], 1], tf.float64)], axis=1))

    return flagged_mini_batch_squared_losses_stack

# %%
def safety_saver(train_fn):
    def train_safe(*args, **kwargs):
        try:
            training_start = time.time()
            train_fn(*args, **kwargs)
            print(f'Total Training Time : {time.time() - training_start}')

        except KeyboardInterrupt :
            print('\nTraining terminated. Saving before exit.. ')
            print(saver.save(sess, saver_save_file, global_step))
            writer.close()
#            sess.close()
            print('[Done]')
            print(f'Total Training Time : {datetime.timedelta(seconds=time.time() - training_start)}')
    return train_safe

def run_schedule(file='./Training/Schedule.csv'):
    from tabulate import tabulate
    schedule = np.genfromtxt(file,
                             delimiter=',',
                             dtype=None, 
                             skip_header=False, 
                             names=True)
    print('\nUsing Schedule ->\n')
    print(tabulate(schedule, schedule.dtype.names))

    current_step=sess.run(global_step)
    to_run = schedule[schedule['totalsteps']>current_step]#
    for i, slot_paras in enumerate(to_run):
        lr, W1, W2, W3, W4,\
        bs, dr, savat= list(slot_paras)[2:]
        
        N = slot_paras[1] - current_step if i==0  else slot_paras[0]
            
        train(n_batches=N, learn_rate=lr, saveat=savat, set_wts_as=[W1, W2, W3, W4],
              batch_size=bs, drop_out_rate=dr, training_mode='CARREAU')
        
    print('Schedule Completed')
def set_wts(W):
    W_ = sess.run(assign_wts, {new_wts:W})
    return f'New Weights set as {W_}'

@safety_saver
def train(n_batches, learn_rate, saveat, batch_size, drop_out_rate, training_mode, set_wts_as=[-1],):
    global saveGraph, writer
    print("Ctrl-C to cancel now", end='\r')
    time.sleep(1)
    if set_wts_as!=[-1]:
        set_wts(set_wts_as)
        
    # Create File Writer for graph ans summaries
#    weights = (np.array(weights)/sum(weights))*W_total
    weights = sess.run(wts)
    summary_tag = gen_summary_tag(act, nhl, n_dgml, N_data_points,
                                  batch_size, learn_rate, weights)
    global writer
    print('Summary Tag : "{}"'.format(summary_tag))

    writer = tf.summary.FileWriter(logdir=summaries_loc+summary_tag,
                                   graph=sess.graph if saveGraph else None,
                                   flush_secs=5,
                                   filename_suffix='DGM')
    saveGraph=False
    sess.run(batch_iter.initializer,feed_dict={b_size: batch_size})
#    L0 = np.squeeze(sess.run([mini_batch_means_of_squared_losses], feed_dict={b_size: b_size}))
#    L0 = np.squeeze(np.array([1e9, 1e9, 1e9, 1e-4]))

    print('\n')
    for i in range(n_batches):
        st = time.time()
        g_step = sess.run(global_step)
        print(f'\rStep:{g_step}   \t', end='' , flush=True)
        print(f'Batch {i+1}:{n_batches}. [b{batch_size}] ', end='', flush=True)
        
        feed_dict={b_size: batch_size,
                   l_rate: learn_rate,
                   drate: drop_out_rate,
                   MODE:training_mode}
        if g_step%saveat==0:
            fetches = [MODE, wts, mini_batch_mean_of_squared_losses, model_update, all_summaries]
        else:
            fetches = [MODE, wts, mini_batch_mean_of_squared_losses, model_update]
            
        fetched_vals = sess.run(fetches=fetches, feed_dict=feed_dict)
        
        m, wts_arr, loss  = fetched_vals[:3]
        print(f'\tmbatch_loss : {loss} \t [{str(m)[:4]}] wts:{wts_arr.round(3)}', end='', flush=True)
        print(f'\tTotal : {np.sum(loss).round(4)} ', end='', flush=True)
        if g_step%saveat==0:
            writer.add_summary(fetched_vals[-1], global_step=g_step)
            print(f'\t[logged]', end='', flush=True)
            saver.save(sess, saver_save_file, global_step)
            print(f'\t[saved]', end='', flush=True)
        print(f'\t[{time.time() - st:.3f} sec]')
        
    writer.close()
    saver.save(sess, saver_save_file, global_step)
#    sess.close()
    print(f'\t[saved]', end='', flush=True)
#    print(f'\nLearning complete on {n_batches}.')

# =============================================================================
# =============================================================================
#  SCRIPT
# =============================================================================
# =============================================================================
print('\n\nBuilding Graph')
with tf.device(batch_host), tf.name_scope('Data_Pipeline'):
    b_size = tf.placeholder(dtype=tf.int64, name='Batch_Size')
    next_batch, batch_iter = decode_and_iterate(datafile, b_size)
    print('\tDeploying Model and Training Ops')

with tf.device(model_host):
    drate = tf.placeholder_with_default(tf.constant(0.0,tf.float64), shape=[], name='Drop_Out_Rate') 
    model = current_model.Raw_DGM_Model(nhl, ydim, xdim, n_dgml+2, 0.5, drate_tensor=drate)
    global_step = tf.train.get_or_create_global_step()

with tf.device(GD_host):
    MODE = tf.placeholder_with_default(tf.constant('NEWTONIAN'), shape=None, name='Training_Mode')

    with tf.variable_scope('Model'):
        mini_batch_of_squared_losses = feed_forward(next_batch, MODE)
        mini_batch_sum_of_squared_losses = tf.reduce_sum(mini_batch_of_squared_losses, axis=0)#, name='Batch_Loss')
        
        mini_batch_mean_of_squared_losses = tf.divide(mini_batch_sum_of_squared_losses, tf.cast(b_size, tf.float64))
        
    with tf.variable_scope('Apply_Weights'):
        wts = tf.get_variable(name='Loss_weights',
                              initializer=tf.constant([1.0, 1.0, 1.0, 1.0], 
                              dtype=tf.float64),
                              trainable=False)
        new_wts = tf.placeholder(dtype=tf.float64, shape=[4])
        assign_wts = tf.assign(wts, new_wts)                      
        weighted_batch_mean_squared_loss_stack = tf.multiply(mini_batch_sum_of_squared_losses, wts,
                                                           name='Apply_Weights')
        weighted_sum_batch_mean_squared_loss = tf.reduce_sum(weighted_batch_mean_squared_loss_stack)

    l_rate = tf.placeholder(dtype=tf.float32, name='Learn_Rate')
    optimizer = tf.train.AdamOptimizer(l_rate, name='Model_Optimizer')
    model_update = optimizer.minimize(weighted_sum_batch_mean_squared_loss, global_step, name='Model_Update')
    
    with tf.name_scope('Logs'), tf.device(metric_host):
        solution_summaries = [tf.summary.image(f'{["U", "V", "P"][i]}', im) for i, im in enumerate(plot_model())]
        summary_names = ['continuity', 'X-mom', 'Y-mom', 'Bound']
        losses_summary = [tf.summary.scalar(name, tensor) for name, tensor in zip(summary_names,
                                                                                  [mini_batch_mean_of_squared_losses[0],
                                                                                   mini_batch_mean_of_squared_losses[1],
                                                                                   mini_batch_mean_of_squared_losses[2],
                                                                                   mini_batch_mean_of_squared_losses[3]])]
    all_summaries = tf.summary.merge_all()

saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
                        name='Saver',
                        restore_sequentially=False,
                        max_to_keep=10,
                        keep_checkpoint_every_n_hours=0.2,
                        saver_def=None,
                        builder=None,
                        pad_step_number=True,
                        save_relative_paths=False,
                        filename='model_graph_save')

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
if init_new!='y':
    if use_latest=='y':
        W_file = tf.train.latest_checkpoint(saver_save_file[:-11])
        saver.restore(sess, W_file)
        steps = int(W_file[W_file.find('l-')+2:])
        sess.run(tf.assign(global_step, steps))
        print(f'Model Loaded from "{saver_save_file[:-11]}"')
    else:
        saver.restore(sess, input('Enter file path'))

sess.graph.finalize()

if use_default_schedule=='y':
    run_schedule('./Training/Schedule.csv')
