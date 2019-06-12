import tkinter as tk
import tensorflow as tf
import numpy as np
import mains
import os
from keras.backend.tensorflow_backend import set_session
# from tkinter.filedialog import askdirectory
from tkinter import messagebox
from tkinter import filedialog

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
window = tk.Tk()
window.title('CerNet for Detection')
window.geometry('400x460')

show_text = tk.Text(window,bg='gray',height=30,width=55)
show_text.place(x=5,y=5,anchor='nw')


file_path = 0
test_filename='#'
def path_select():
	global file_path
	file_path = filedialog.askopenfilename()
	#获取文件的绝对路径
	show_text.insert('end','\n'+file_path)

# path = tk.StringVar()
# def path_select():
# 	path_ = askdirectory()
# 	path.set(path_)
# 	show_text.insert('end',path)

def path_analyze():
	global test_filename
	test_filename = file_path
	if file_path == 0:
		tk.messagebox.showwarning(title='Warning',message='You have not select file to detect!\n'+
            "Please click 'load' to select the file!")
	else:
		test()
		print('after: ' + str(test_filename))



def load_data():
    if test_filename=='#':
        tk.messagebox.showwarning(title='Warning',message='You have not select file to detect!\n'+
            'Please select the file to next step!')
    else:
        fr = open(test_filename)
        show_text.insert('end','\nload testing data from' + test_filename)
        lines = fr.readlines()
        line_nums = len(lines)
        #line_nums = 20000
        #para_num = 25 #ae30
        para_num = 42 #UNSW-NB15
        test_set = np.zeros((line_nums, para_num+1))  # Create a matrix of line_nums rows and para_num columns
        for i in range(line_nums):
            #line = lines[i].strip()
            line = lines[i].strip()
            test_set[i, :] = line.split(',')
        fr.close()
    return test_set

def test():
    test_data = load_data()#获取测试数据
    saver = tf.train.import_meta_graph("models/model.ckpt.meta")
    with tf.Session() as sess:
        # config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        # set_session(tf.Session(config=config))
        # sess.run(tf.local_variables_initializer())
        show_text.insert('end',"\nLoading best weights...")
        initializer = tf.random_uniform_initializer(-0.1, 0.1)

        with tf.name_scope("Test"):
            global test_input
            test_input = mains.PTBInput(config=mains.eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                global mtest
                mtest = mains.PTBModel(is_training=False, config=mains.eval_config,
                                 input_=test_input)

        #test_input = mains.PTBInput(config=mains.eval_config, data=test_data)
        #mtest = mains.PTBModel(is_training=False, config=mains.eval_config,input_=test_input)
        # saver.restore(session, FLAGS.save_path + 'model.ckpt')
        saver = tf.train.Saver(tf.trainable_variables(),max_to_keep=7)

        sess.run(tf.local_variables_initializer())
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'models2/model.ckpt')#
        #sess.run(tf.local_variables_initializer())

        print(11111)
        
        test_cost, test_accuracy = mains.run_epoch(sess, mtest)#
        show_text.insert('end',"\nTest Loss: " + test_cost + ", Test accuracy: " + test_accuracy)
        print(2222)

load_btn = tk.Button(window,command=path_select,width=10,text='Load')
dec_btn = tk.Button(window,command=path_analyze,width=10,text='Detection')

load_btn.place(x=100,y=410,anchor='nw')
dec_btn.place(x=220,y=410,anchor='nw')

window.mainloop()


