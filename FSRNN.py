import tensorflow as tf

class FSRNNCell(tf.contrib.rnn.RNNCell):
    def __init__(self, fast_cells, slow_cell, keep_prob=1.0, training=True):
        """Initialize the basic Fast-Slow RNN.
            Args:
              fast_cells: A list of RNN cells that will be used for the fast RNN.
                The cells must be callable, implement zero_state() and all have the
                same hidden size, like for example tf.contrib.rnn.BasicLSTMCell.
              slow_cell: A single RNN cell for the slow RNN.
              keep_prob: Keep probability for the non recurrent dropout. Any kind of
                recurrent dropout should be implemented in the RNN cells.
              training: If False, no dropout is applied.
        """

        self.fast_layers = len(fast_cells)
        assert self.fast_layers >= 2, 'At least two fast layers are needed'
        self.fast_cells = fast_cells
        self.slow_cell = slow_cell
        self.keep_prob = keep_prob
        if not training: self.keep_prob = 1.0

    def __call__(self, inputs, state, scope='FS-RNN'):
        
        F_state = state[0]
        S_state = state[1]

        with tf.variable_scope(scope):

            inputs_size = int(inputs.get_shape().as_list()[0]/2)# inputs 的第一个维度除以2
            print("inputs_size")
            print(inputs_size)
            print(type(inputs_size))
            inputs1 = inputs[0:inputs_size,:]# 矩阵切片,第一维表示行的范围，第二维表示列
            inputs2 = inputs[inputs_size:inputs_size*2,:]
            print(inputs1.shape)
            print(inputs2.shape)
            inputs = tf.nn.dropout(inputs, self.keep_prob)
            inputs1 = tf.nn.dropout(inputs1, self.keep_prob)
            inputs2 = tf.nn.dropout(inputs2, self.keep_prob)

            with tf.variable_scope('First_0'):
                F_output, F_state = self.fast_cells[0](inputs1, F_state)
            F_output1_drop = tf.nn.dropout(F_output, self.keep_prob)

            with tf.variable_scope('First_1'):
                F_output, F_state = self.fast_cells[1](inputs2, F_state)
            F_output2_drop = tf.nn.dropout(F_output, self.keep_prob)

            with tf.variable_scope('Second'):
                S_output, S_state = self.slow_cell(inputs, S_state)
            S_output_drop = tf.nn.dropout(S_output, self.keep_prob)


            #for i in range(2, self.fast_layers):
            #    with tf.variable_scope('Fast_' + str(i)):
            #        # Input cannot be empty for many RNN cells
            #        F_output, F_state = self.fast_cells[i](F_output[:, 0:1] * 0.0, F_state)

            F_output_drop = tf.concat(axis=0, values=[F_output1_drop, F_output2_drop])
            print("output")
            print(F_output_drop)
            return F_output_drop, S_output_drop, (F_state, S_state)


    def zero_state(self, batch_size, dtype):
        F_state = self.fast_cells[0].zero_state(int(batch_size/2), dtype)
        S_state = self.slow_cell.zero_state(batch_size, dtype)

        return (F_state, S_state)