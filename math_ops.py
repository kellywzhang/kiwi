import tensorflow as tf

def cond_gather(cond, true_tensor, false_tensor):
    """
    The purpose of this function is to efficiently tf.gather from two
    tensors based on a given condition vector.
    If the condition is true, the vector from the true_tensor is gathered,
    otherwise the vector from the false_tensor is gathered.

    This function takes two tensors of the same rank and dimensions, and
    one vector that has the same dimension as the first dimension of the tensors.
    This function returns a tensor of the same rank and dimension as
    the true and false tensors.

    This version of this function will only gather along the first (0th)
    dimension; if needed, I can make an option to gather along a different axis.

    Example:
    true_tensor = tf.constant([[20,30,40,50,60],[2,3,4,5,0]]) # shape (2, 5)
    false_tensor = tf.constant([[2,3,4,5,0],[20,30,40,50,60]])
    cond = tf.constant([True, False])

    cond_gathered = math_ops.cond_gather(cond, true_tensor, false_tensor)
    cond_gathered = [[20 30 40 50 60], [20 30 40 50 60]]
    """
    # gather along the first dimension (usually batch_size)
    dim_0 = tf.shape(true_tensor)[0]

    # concatenate both true and false condition tensors
    # tf.concat_v2
    false_true_tensor = tf.concat(0, [false_tensor, true_tensor])

    # cast condition to int (typically a boolean) and multiply by dim_0
    # by adding int_cond to true indices, will gather those indices from true_tensor
    int_cond = tf.cast(cond, tf.int32) * dim_0
    int_indices = tf.range(0, dim_0, 1) + int_cond

    gathered = tf.gather(false_true_tensor, int_indices)

    return gathered
