import tkinter as tk
from tkinter import Scrollbar,Frame
from tkinter.ttk import Treeview
import tensorflow as tf
import numpy as np
import csv
import os
# import sys     
from keras.backend.tensorflow_backend import set_session
# from tkinter.filedialog import askdirectory
from tkinter import messagebox
from tkinter import filedialog

import aux1 as aux
import reader
import configs
import LNLSTM
import FSRNN

window = tk.Tk()
window.title('Ipv6异常流量检测系统')
window.geometry('900x500')#WxH
window.resizable(False,False)

frame1 = Frame(window)
frame1.place(x=5,y=5,width=880,height=110)
scrollbar1 = tk.Scrollbar(frame1)#滚动条
scrollbar1.pack(side=tk.RIGHT,fill=tk.Y)
show_text = tk.Text(frame1,bg='Gainsboro',height=8,width=122,yscrollcommand=scrollbar1.set)
show_text.place(x=0,y=0,anchor='nw')
scrollbar1.config(command=show_text.yview)

frame2 = Frame(window)
frame2.place(x=5,y=120,width=880,height=270)

scrollbar2 = tk.Scrollbar(frame2)#滚动条

scrollbar2.pack(side=tk.RIGHT,fill=tk.Y)

s0,s1,s2,s3,s4,s5 = 'No.','Source',"Destination","Protocol","Length","State"

tree = Treeview(frame2,show='headings',yscrollcommand=scrollbar2.set)#表格
tree["columns"] = (s0,s1,s2,s3,s4,s5)
tree.column(s0,width=150,anchor='center')
tree.column(s1,width=190,anchor='center')
tree.column(s2,width=190,anchor='center')
tree.column(s3,width=130,anchor='center')
tree.column(s4,width=100,anchor='center')
tree.column(s5,width=100,anchor='center')

tree.heading(s0,text=s0) #显示表头
tree.heading(s1,text=s1)
tree.heading(s2,text=s2)
tree.heading(s3,text=s3)
tree.heading(s4,text=s4)
tree.heading(s5,text=s5)

tree.pack(side=tk.LEFT,fill=tk.Y)#填充y方向
scrollbar2.config(command=tree.yview)

test_set = 0 
file_path = 0
test_filename='#'

def path_select():
    global file_path
    global test_filename
    file_path = filedialog.askopenfilename()
    #获取文件的绝对路径
    show_text.insert('end','\nload testing data：'+ file_path)
    test_filename = file_path

def path_analyze():
    global test_filename
    if test_filename == '#':
        tk.messagebox.showwarning(title='Warning',message='You have not select file to detect!\n'+
            "Please click 'load' to select the file!")
    else:
        show_text.insert('end','\nDetection over')
        test()

def return_col(file_name):
	with open(file_name,encoding='utf-8') as csvfile:
		reader = csv.reader(csvfile)
		row_num = ""
		for row in reader:
			row_num = row
		return len(row_num)

def load_data():
    global test_filename
    if test_filename =='#':
        tk.messagebox.showwarning(title='Warning',message='You have not select file to detect!\n'+
            'Please select the file to next step!')
    else:
        try:
            fr = open(test_filename)
        except OSError as reason:
            show_text.insert('end',"\nFalse! Please open a correct file!")
        lines = fr.readlines()
        line_nums = len(lines)
        para_num = return_col(test_filename)
        global test_set
        test_set = np.zeros((line_nums, para_num))  # Create a matrix of line_nums rows and para_num columns
        for i in range(line_nums):
            #line = lines[i].strip()
            line = lines[i].strip()
            test_set[i, :] = line.split(',')
            # test[:,0:5] = test_set[:,6,10]
            # test[:,5] = test_set[:,5]
        fr.close()
        # print("test is: ")
        # for i in length(test):
        # 	for j in length(test[0]):
        # 		print(test[i][j])
        
    return test_set


flags = tf.flags
flags.DEFINE_string("model", "ptb",
                    "A type of model. Check configs file to know which models are available.")
flags.DEFINE_string("data_path", 'data/',
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", 'models/',
                    "Model output directory.")

FLAGS = flags.FLAGS

class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        global batch_size
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1)

        #self.input_data, self.targets = reader.ptb_producer(
        #    data, batch_size, num_steps, name=name)
        self.input_data, self.targets = reader.kdd_iterator(data, batch_size, num_steps)

class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        F_size = config.cell_size
        S_size = config.hyper_size
        #vocab_size = config.vocab_size

        #emb_init = aux.orthogonal_initializer(1.0)
        #with tf.device("/cpu:0"):
        #    embedding = tf.get_variable(
        #        "embedding", [vocab_size, emb_size], initializer=emb_init, dtype=tf.float32)
        #    inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
        inputs = input_.input_data

        #First layer
        F_cells = [LNLSTM.LN_LSTMCell(F_size, use_zoneout=True, is_training=is_training,
                                      zoneout_keep_h=config.zoneout_h, zoneout_keep_c=config.zoneout_c)
                   for _ in range(config.fast_layers)]

        #Second layer
        S_cell  = LNLSTM.LN_LSTMCell(S_size, use_zoneout=True, is_training=is_training,
                                     zoneout_keep_h=config.zoneout_h, zoneout_keep_c=config.zoneout_c)

        FS_cell = FSRNN.FSRNNCell(F_cells, S_cell, config.keep_prob, is_training)

        self._initial_state = FS_cell.zero_state(batch_size, tf.float32)
        state = self._initial_state

        outputsF = []
        outputsS = []
        print('generating graph')
        #with tf.variable_scope("RNN"):
        #    for time_step in range(num_steps):
        #        if time_step > 0: tf.get_variable_scope().reuse_variables()
                #out, state = FS_cell(inputs[:, time_step, :], state)
        #        out, state = FS_cell(inputs[:, time_step, :], state)
        #        outputs.append(out)
        with tf.variable_scope("RNN"):
            #for time_step in range(batch_size):
             #   if time_step > 0: tf.get_variable_scope().reuse_variables()
            #out, state = FS_cell(inputs[:, time_step, :], state)
            outF, outS, state = FS_cell(inputs, state)
            outputsF.append(outF)
            outputsS.append(outS)
                #outputs.append(tf.squeeze(cell_output))

        print('graph generated')
        outputF = tf.reshape(tf.concat(axis=1, values=outputsF), [-1, F_size])
        #这里不确定
        outputS = tf.reshape(tf.concat(axis=1, values=outputsS), [-1, S_size])

        # Output layer and cross entropy loss

        out_init = aux.orthogonal_initializer(1.0)
        #out_init = tf.orthogonal_initializer(1.0)
        #softmax_w = tf.get_variable(
        #    "softmax_w", [F_size, vocab_size], initializer=out_init, dtype=tf.float32)
        #softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        softmax_w = tf.get_variable(
            "softmax_w", [F_size, 2], initializer=out_init, dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [2], dtype=tf.float32)
        logits1 = tf.matmul(outputF, softmax_w) + softmax_b

        softmax_w2 = tf.get_variable(
            "softmax_w2", [S_size, 2], initializer=out_init, dtype=tf.float32)
        softmax_b2 = tf.get_variable("softmax_b2", [2], dtype=tf.float32)
        logits2 = tf.matmul(outputS, softmax_w2) + softmax_b2

        logits = [tf.cond(tf.maximum(logits1[0,0], logits1[0,1]) > 0.6, lambda: logits1[0,:], lambda: logits1[0,:] + logits2[0,:])]

        for i in range(logits1.shape[0] - 1):
            logits = tf.cond(tf.maximum(logits1[i+1,0], logits1[i+1,1]) > 0.6, lambda: tf.concat([logits, [logits1[i+1,:]]],0),
             lambda: tf.concat([logits, [(logits1[i+1,:] + logits2[i+1,:])/2.0]], 0))

        #my loss
        y_new = tf.cond(logits[0,0] > logits[0,1], lambda: tf.constant([[1.0,0.0]]), lambda: tf.constant([[0.0,1.0]]))
        for i in range(logits.shape[0]-1):
            y_new = tf.cond(logits[i+1,0] > logits[i+1,1], lambda: tf.concat([y_new, tf.constant([[1.0,0.0]])], 0), 
                lambda: tf.concat([y_new, tf.constant([[0.0,1.0]])], 0))
        #y_new is the predicted result

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.to_float(input_.targets), logits=logits))

        self._cost = cost = loss

        self._final_state = state

        correct_prediction = tf.equal(y_new, tf.to_float(input_.targets))

        b_list = []
        for i in range(correct_prediction.shape[0]):
            b_list.append(correct_prediction[i,0] & correct_prediction[i,1])
        self._accuracy = tf.reduce_mean(tf.cast(b_list, tf.float32))
        self._y_new = y_new
        self._y_target = input_.targets

        if not is_training: return

        # Create the parameter update ops if training

        self._lr = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(cost, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N),
            config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def y_target(self):
        return self._y_target

    @property
    def y_new(self):
        return self._y_new

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""

    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
        "accuracy":model.accuracy,
        "y_new":model.y_new,
        "y_target":model.y_target
    }
    accuracys = 0.0
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    output_y = []
    for step in range(model.input.epoch_size):
        feed_dict = {}
        feed_dict[model.initial_state] = state

        vals = session.run(fetches, feed_dict)

        cost = vals["cost"]
        state = vals["final_state"]
        accuracy = vals["accuracy"]
        y_new = vals["y_new"]
        y_target = vals["y_target"]

        costs += cost
        accuracys += accuracy
        #iters += model.input.num_steps
        iters = iters + 1
        for i in range(model.input.batch_size):
            if y_new[i,0] == 0:
                output_y.append(1)
            else:
                output_y.append(0)

    return costs, accuracys / iters, output_y

def test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config       = configs.get_config(FLAGS.model)
    eval_config  = configs.get_config(FLAGS.model)
    eval_config.batch_size = 2

    # test_data = load_data()#获取测试数据
    test = load_data_string()
    test_data = np.zeros((len(test), 6))
    test_data[:,0:4] = test[:,6:10]
    test_data[:,5] = test[:,5]

    # print("test is: ")
    # for i in range(len(test_data)):
    # 	for j in range(len(test_data[0])):
    # 		print(test_data[i][j])

    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    test_input = PTBInput(config=eval_config, data=test_data)
    with tf.variable_scope("Model", reuse=tf.AUTO_REUSE, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)
    saver = tf.train.Saver(tf.trainable_variables())

    #saver = tf.train.import_meta_graph("models/model.ckpt.meta")
    with tf.Session() as sess:
        # config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        # set_session(tf.Session(config=config))
        # sess.run(tf.local_variables_initializer())
        show_text.insert('end',"\nLoading detection modle......")

        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # print(11111)
        #test_input = mains.PTBInput(config=mains.eval_config, data=test_data)
        #mtest = mains.PTBModel(is_training=False, config=mains.eval_config,input_=test_input)
        # saver.restore(session, FLAGS.save_path + 'model.ckpt')
        saver.restore(sess, FLAGS.save_path + 'model.ckpt')
        # print(2222)
        test_cost, test_accuracy, output_y_test = run_epoch(sess, mtest)

        clear_tree()
        anomaly_main = load_data_string()
        anomaly_test = []
        for i in range(len(output_y_test)):
            if output_y_test[i] == 1:
                anomaly_test.append(anomaly_main[i,:])

        # print("anomaly_test")
        # print(anomaly_test)

        for i in range(len(anomaly_test)):
	        if anomaly_test[i][-1] == b'1':# "1" is bad or error data
	            tree.insert('','end',value=[anomaly_test[i][j].decode('utf-8') for j in range(len(anomaly_test[i]))])


        # print(3333)
        print(test_cost)
        # Test Loss: " + str(test_cost) + "
        show_text.insert('end',"\n Test accuracy: " + str(test_accuracy))

        coord.request_stop()
        coord.join(threads)

def load_data_string():
    global test_filename
    if test_filename =='#':
        tk.messagebox.showwarning(title='Warning',message='You have not select file to detect!\n'+
            'Please select the file to next step!')
    else:
        try:
            file = open(test_filename,"r")
        except OSError as reason:
            show_text.insert('end',"\nFalse! Please open a correct file!")
        #file = open('/home/dl/lx/Fast-Slow-LSTM-master_ui/ipv6_test_orial.csv',"r")
        list_arr = file.readlines()
         
        lists = []
        for index,x in enumerate(list_arr):
            x = x.strip()
            x = x.strip('[]')
            x = x.split(",")
            lists.append(x)
        a = np.array(lists)
        a = a.astype('string_')
        # print (a)
        file.close()
    return a

def show_all_data():#显示所有数据
    result = load_data_string()
    if test_filename != '#':
        show_text.insert('end',"\nDisplay all testing data.")
    clear_tree()
    for i in range(len(result)):
        tree.insert('','end',value=[result[i][j].decode('utf-8') for j in range(len(result[i]))])

def show_anomaly_data():
    result = load_data_string()
    if test_filename != '#':
        show_text.insert('end',"\nDisplay anomaly testing data.")
    clear_tree()
    for i in range(len(result)):
        if result[i][-1] == b'1':
            tree.insert('','end',value=[result[i][j].decode('utf-8') for j in range(len(result[i]))])

def clear_text():
    show_text.delete(1.0,tk.END)

def clear_tree():
    x=tree.get_children()
    for item in x:
        tree.delete(item)

load_btn = tk.Button(window,command=path_select,width=10,text='Load')
dec_btn = tk.Button(window,command=path_analyze,width=10,text='Detection')
show_all_btn = tk.Button(window,command=show_all_data,width=12,text='Show all data')
show_anomaly_btn = tk.Button(window,command=show_anomaly_data,width=13,text='Show anomaly data')
clear_text_btn = tk.Button(window,command=clear_text,width=8,text='Clear Text')
clear_tree_btn = tk.Button(window,command=clear_tree,width=8,text='Clear List')


load_btn.place(x=30,y=430,anchor='nw')
dec_btn.place(x=180,y=430,anchor='nw')
show_all_btn.place(x=330,y=430,anchor='nw')
show_anomaly_btn.place(x=480,y=430,anchor='nw')
clear_text_btn.place(x=650,y=430,anchor='nw')
clear_tree_btn.place(x=780,y=430,anchor='nw')

window.mainloop()
