'''
Forward and propagate operators for VNNs and cellular machines

Propagate functions act as learnable Green's functions for cells

@author Shakes
'''
import jax
import jax.numpy as jnp

#JIT compiled members for performance
from functools import partial

from convs import _fast_conv, _fast_conv_3D, _fast_conv_3D_compact, _filter

#propagate funcs
@partial(jax.jit, static_argnums=(2,3,4,10,11,12,13,14))
def _propagate_full_clp(input, coords, out_offset, out_size, depth, cells, weights, biases, states, filter_ffts, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. for all layers across entire universe
    """
    width = 0 #unused
    # length = input.shape[0]

    cells = cells.at[coords[:, 0], coords[:, 1]].set(input)

    #propagate
    for i in range(1, depth):
        cells = _forward_full(width, cells, weights, biases, states, filter_ffts, activate, norm, dn, window_strides) #not inplace
    cells = _forward_full(width, cells, weights, biases, states, filter_ffts, final_activate, norm, dn, window_strides) #final prop and activate

    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

@partial(jax.jit, static_argnums=(1,2,3,4,10,11,12,13,14))
def _propagate_mlp(input, offset, out_offset, out_size, depth, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. as a traditional MLP

    Uses column/last hyperplane based computation for efficiency and propagates up to provided depth and
    then one more for output layer.
    Last dim of filters is used as the initial starting step
    """
    zeros_tuple = (0,) * (len(offset)-1) #offset for entire hyper plane
    step = 1 #(filters.shape[-2]-1)//2 #-2 to take last dim butignore channel dim
    # print("forward step:", step)

    cells = jax.lax.dynamic_update_slice(cells, input, offset) #place input into cells

    #propagate
    index = offset[-1]+step #first col (or last hyper plane) next to input
    # print("forward index",index)
    for i in range(depth): #0 to depth-1, depth in total
        result_block = _forward_column(index+i, cells, weights, biases, states, filters, activate, norm, dn, window_strides)
        cells = jax.lax.dynamic_update_slice(cells, result_block, zeros_tuple+(index+i,)) #place back into col (or hyper plane)
    # print("forward index final",index+depth)
    result_block = _forward_column(index+depth, cells, weights, biases, states, filters, final_activate, norm, dn, window_strides)
    if not final_activate == None:
        result_block = final_activate(result_block) #force final activation, assume output layer
    cells = jax.lax.dynamic_update_slice(cells, result_block, zeros_tuple+(index+depth,)) #place back into final col (or hyper plane)

    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

@partial(jax.jit, static_argnums=(1,2,3,4,10,11,12,13,14))
def _propagate_mlp_row(input, offset, out_offset, out_size, depth, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. as a traditional MLP

    Uses row/second-last hyperplane based computation for efficiency and propagates up to provided depth and
    then one more for output layer.
    """
    zeros_tuple = (0,) * (len(offset)-2) #offset for entire hyper plane upto second last (row)

    cells = jax.lax.dynamic_update_slice(cells, input, offset) #place input into cells

    #propagate
    index = offset[-2]+1 #first row (or second-last hyper plane) next to input
    # print("forward index",index)
    for i in range(depth): #0 to depth-1, depth in total
        result_block = _forward_row(index+i, cells, weights, biases, states, filters, activate, norm, dn, window_strides)
        cells = jax.lax.dynamic_update_slice(cells, result_block, zeros_tuple+(index+i,)+(0,)) #place back into row (or hyper plane)
    # print("forward index final",index+depth)
    result_block = _forward_row(index+depth, cells, weights, biases, states, filters, final_activate, norm, dn, window_strides)
    if not final_activate == None:
        result_block = final_activate(result_block) #force final activation, assume output layer
    cells = jax.lax.dynamic_update_slice(cells, result_block, zeros_tuple+(index+depth,)+(0,)) #place back into final row (or hyper plane)

    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

@partial(jax.jit, static_argnums=(1,2,3,4,10,11,12,13,14))
def _propagate_tape(input, offset, out_offset, out_size, depth, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. as a Turing machine like tape.

    Uses 1D computations for efficiency and propagates up to provided depth.
    """
    zeros_tuple = (0,) * (len(offset)-2) #offset for entire hyper plane upto second last (row)

    cells = jax.lax.dynamic_update_slice(cells, input, offset) #place input into cells

    #propagate
    index = offset[-2] #first and only row, all computations to be done in 1D, no layers
    # print("forward index",index)
    for i in range(depth): #0 to depth-1, depth in total, i unused as there are no layers
        cells = _forward_row(index, cells, weights, biases, states, filters, activate, norm, dn, window_strides)
        # cells = jax.lax.dynamic_update_slice(cells, result_block, zeros_tuple+(index,)+(0,)) #place back into row (or hyper plane)
    # print("forward index final",index+depth)
    cells = _forward_row(index, cells, weights, biases, states, filters, final_activate, norm, dn, window_strides)
    if not final_activate == None:
        cells = final_activate(cells) #force final activation, assume output layer
    # cells = jax.lax.dynamic_update_slice(cells, result_block, zeros_tuple+(index,)+(0,)) #place back into final row (or hyper plane)

    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

@partial(jax.jit, static_argnums=(1,2,3,4,10,11,12,13,14))
def _propagate_tape_unrolled(input, offset, out_offset, out_size, depth, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. as a Turing machine like tape.

    Uses 1D computations for efficiency and propagates up to provided depth.
    Unlike _propagate_tape, this unrolled version returns the full history of states across the depth dimension.
    """
    zeros_tuple = (0,) * (len(offset)-2) #offset for entire hyper plane upto second last (row)

    cells = jax.lax.dynamic_update_slice(cells, input, offset) #place input into cells

    history = [cells]

    #propagate
    index = offset[-2] #first and only row, all computations to be done in 1D, no layers
    for i in range(depth): #0 to depth-1, depth in total, i unused as there are no layers
        cells = _forward_row(index, cells, weights, biases, states, filters, activate, norm, dn, window_strides)
        history.append(cells)

    cells = _forward_row(index, cells, weights, biases, states, filters, final_activate, norm, dn, window_strides)
    if not final_activate == None:
        cells = final_activate(cells) #force final activation, assume output layer
    history.append(cells)

    history_stacked = jnp.stack(history, axis=0)

    return history_stacked, jax.lax.dynamic_slice(cells, out_offset, out_size)

@partial(jax.jit, static_argnums=(1,2,3,4,10,11,12,13,14))
def _propagate_full_mlp(input, offset, out_offset, out_size, depth, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. as a traditional MLP
    """
    cells = jax.lax.dynamic_update_slice(cells, input, offset) #place input into cells

    #propagate
    for i in range(1, depth):
        cells = _forward_full(1, cells, weights, biases, states, filters, activate, norm, dn, window_strides) #not inplace
    cells = _forward_full(1, cells, weights, biases, states, filters, final_activate, norm, dn, window_strides) #final prop and activate
    if not final_activate == None:
        cells = final_activate(cells) #force final activation, assume output layer

    if isinstance(norm, list) and len(norm) == 5:
        return cells, jax.lax.dynamic_slice(cells, out_offset, out_size), norm[2], norm[3]
    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

@partial(jax.jit, static_argnums=(2,3,4,10,11,12,13,14))
def _propagate_full_mlp_list(inputs, offsets, out_offset, out_size, depth, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. as a traditional MLP, inputs are list of inputs, one output still
    """
    for input, offset in zip(inputs, offsets): #place inputs to cells
        cells = jax.lax.dynamic_update_slice(cells, input, offset)

    #propagate
    for i in range(1, depth):
        cells = _forward_full(1, cells, weights, biases, states, filters, activate, norm, dn, window_strides) #not inplace
    cells = _forward_full(1, cells, weights, biases, states, filters, final_activate, norm, dn, window_strides) #final prop and activate
    if not final_activate == None:
        cells = final_activate(cells) #force final activation, assume output layer

    if isinstance(norm, list) and len(norm) == 5:
        return cells, jax.lax.dynamic_slice(cells, out_offset, out_size), norm[2], norm[3]
    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

_static_argnums_mlp_blocks = (1,2,3,4,5,11,12,14,15,16,17)
@partial(jax.jit, static_argnums=_static_argnums_mlp_blocks)
def _propagate_mlp_blocks_fast(input, offset, out_offset, out_size, depth, blocks, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1), is_training=True, expand=False):
    """
    Propagate input across cellular net based on weights etc. as a traditional MLP

    Uses column/last hyperplane based computation for efficiency and propagates up to provided depth and
    then one more for output layer.
    Each block is done depth//blocks of the propagation assuming there are blocks filters for each block
    """
    zeros_tuple = (0,) * (len(offset)-1) #offset for entire hyper plane
    depth_block = depth//blocks #first plane is untouched
    update_offset = (filters[blocks,0].shape[-2]-1)//2 #used to account for result size with being filter size in this dim
    overlap = 1 #factor of kernel overlap
    step = (filters[blocks,0].shape[-2]-1)//2
    step //= overlap

    if isinstance(norm, list) and len(norm) == 5:
        ema_mean_arr = norm[2]
        ema_var_arr = norm[3]

    cells = jax.lax.dynamic_update_slice(cells, input, offset) #place input into cells

    if expand and blocks > 1:
        directions = filters.shape[-1]
        cells_expanded = jnp.repeat(cells[..., jnp.newaxis], directions, axis=-1)
    else:
        cells_expanded = None

    #propagate signals across cells using blocks of filters
    step_offset = input.shape[-1]
    for j in range(blocks):
        #propagate per block
        current_mode = "EXPAND" if (expand and blocks > 1) else "COMPRESS"
        index = offset[-1]+j*depth_block*overlap*step+step_offset #first col (or last hyper plane) next to input including its size
        for i in range(depth_block*overlap): #0 to depth-1, depth in total plus one overlap
            if expand and blocks > 1:
                result_block, ema_mean_arr, ema_var_arr = _forward_column_fast_compact(index+i*step, cells_expanded, weights, biases, states, filters[j,0], activate, norm, dn, window_strides, mode=current_mode, is_training=is_training, norm_index=j, depth_block=depth_block*overlap)
                if isinstance(norm, list) and len(norm) == 5:
                    norm = [norm[0], norm[1], ema_mean_arr, ema_var_arr, norm[4]]
                cells_expanded = jax.lax.dynamic_update_slice(cells_expanded, result_block, zeros_tuple+(index+i*step-update_offset, 0)) #place back into expanded col

                # Update persistent 1-channel cells state as well for visualization/consistency
                cells = jax.lax.dynamic_update_slice(cells, jnp.mean(result_block, axis=-1), zeros_tuple+(index+i*step-update_offset,))
            else:
                result_block, ema_mean_arr, ema_var_arr = _forward_column_fast_compact(index+i*step, cells, weights, biases, states, filters[j,0], activate, norm, dn, window_strides, mode=current_mode, is_training=is_training, norm_index=j, depth_block=depth_block*overlap)
                if isinstance(norm, list) and len(norm) == 5:
                    norm = [norm[0], norm[1], ema_mean_arr, ema_var_arr, norm[4]]
                cells = jax.lax.dynamic_update_slice(cells, result_block, zeros_tuple+(index+i*step-update_offset,)) #place back into col (or hyper plane)

    #final propagation using final filter, compress back into single channel
    target_cells = cells_expanded if (expand and blocks > 1) else cells
    result_block, ema_mean_arr, ema_var_arr = _forward_column_fast_compact(out_offset[-1], target_cells, weights, biases, states, filters[blocks,0], final_activate, norm, dn, window_strides, mode="COMPRESS", is_training=is_training, norm_index=blocks, depth_block=1)
    if isinstance(norm, list) and len(norm) == 5:
        norm = [norm[0], norm[1], ema_mean_arr, ema_var_arr, norm[4]]
    if not final_activate == None:
        result_block = final_activate(result_block) #force final activation, assume output layer
    cells = jax.lax.dynamic_update_slice(cells, result_block, zeros_tuple+(out_offset[-1]-update_offset,)) #place back into final col (or hyper plane)

    if isinstance(norm, list) and len(norm) == 5:
        return cells, jax.lax.dynamic_slice(cells, out_offset, out_size), norm[2], norm[3]
    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

# Memory-efficient version for Deep Learning training (Backward pass memory savings)
# Applying jax.checkpoint (remat) inside jax.jit ensures intermediate FFT arrays are discarded
# and recomputed during backpropagation, saving massive amounts of VRAM.
_propagate_mlp_blocks_fast_jit = jax.jit(
    jax.checkpoint(
        _propagate_mlp_blocks_fast,
        static_argnums=_static_argnums_mlp_blocks
    ),
    static_argnums=_static_argnums_mlp_blocks
)

def _propagate_mlp_blocks_fast_compact(input, offset, out_offset, out_size, depth, blocks, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1), is_training=True, expand=False):
    """Memory-efficient column forward operator for neural network training."""
    return _propagate_mlp_blocks_fast_jit(input, offset, out_offset, out_size, depth, blocks, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides, is_training, expand)

@partial(jax.jit, static_argnums=(1,2,3,4,5,11,12,13,14,15))
def _propagate_full_mlp_blocks(input, offset, out_offset, out_size, depth, blocks, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1)):
    """
    Propagate input across cellular net based on weights etc. as a traditional MLP
    """
    depth_block = depth//blocks
    # print("blocks", blocks, "depth_block", depth_block)

    # if norm: #with no definite 'layers', we need norm free approach, hence weights standardization
    #     # gamma for scaled weights standardization equivalent to batch norm
    #     # Calculate scaling factor for ReLU
    #     # gamma = utils.get_relu_gain_gamma([filters.shape[-2]])

    #     # Apply the standardization function to the weights
    #     mean_axes = tuple(i for i in range(len(filters.shape)-1)) #nD cells need norm'ing, ignore B and C
    #     # print("mean axes", mean_axes, "filters", filters.shape)
    #     # filters = jax.nn.standardize(filters, axis=mean_axes)
    #     filters = utils.normalize_l2(filters, axis=mean_axes) # unit norm

    #     # # Apply the standardization function to the weights
    #     # mean_axes = tuple(i for i in range(len(weights.shape))) #nD cells need norm'ing, ignore B and C
    #     # # print("mean axes", mean_axes, "filters", filters.shape)
    #     # weights = jax.nn.standardize(weights, axis=mean_axes)
    #     # weights = utils.normalize_l2(weights, axis=mean_axes) # unit norm

    cells = jax.lax.dynamic_update_slice(cells, input, offset) #place input into cells

    #propagate signals across cells using blocks of filters
    for j in range(blocks):
        #propagate per block
        for i in range(depth_block): #0 to depth-1, depth in total with no overlap required
            index = j*depth_block+i #unused atm
            # print("index", index, "cells shape", cells.shape, "filters", filters[j][jnp.newaxis,...].shape)
            cells = _forward_full(1, cells, weights, biases, states, filters[j], activate, norm, dn, window_strides, mode="EXPAND") #not inplace
    #final propagation using final filter, compress back into single channel
    cells = _forward_full(1, cells, weights, biases, states, filters[blocks], final_activate, norm, dn, window_strides, mode="COMPRESS") #final prop and activate
    # print("index", index, "cells shape", cells.shape)
    if not final_activate == None:
        cells = final_activate(cells) #force final activation, assume output layer

    if isinstance(norm, list) and len(norm) == 5:
        return cells, jax.lax.dynamic_slice(cells, out_offset, out_size), norm[2], norm[3]
    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

static_argnums_mlp_blocks = (1,2,3,4,5,11,12,14,15,16,17)
@partial(jax.jit, static_argnums=static_argnums_mlp_blocks)
def _propagate_full_mlp_blocks_fast(input, offset, out_offset, out_size, depth, blocks, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1), is_training=True, expand=False):
    """
    Propagate input across cellular net based on weights etc. as a traditional MLP
    """
    depth_block = depth//blocks
    # print("blocks", blocks, "depth_block", depth_block)

    # if norm: #with no definite 'layers', we need norm free approach, hence weights standardization
    #     # gamma for scaled weights standardization equivalent to batch norm
    #     # Calculate scaling factor for ReLU
    #     # gamma = utils.get_relu_gain_gamma([filters.shape[-2]])

    #     # Apply the standardization function to the weights
    #     mean_axes = tuple(i for i in range(len(filters.shape)-1)) #nD cells need norm'ing, ignore B and C
    #     # print("mean axes", mean_axes, "filters", filters.shape)
    #     filters = jax.nn.standardize(filters, axis=mean_axes)
    #     # filters = utils.normalize_l2(filters, axis=mean_axes) # unit norm

    #     # # Apply the standardization function to the weights
    #     # mean_axes = tuple(i for i in range(len(weights.shape))) #nD cells need norm'ing, ignore B and C
    #     # # print("mean axes", mean_axes, "filters", filters.shape)
    #     # weights = jax.nn.standardize(weights, axis=mean_axes)
    #     # weights = utils.normalize_l2(weights, axis=mean_axes) # unit norm

    cells = jax.lax.dynamic_update_slice(cells, input, offset) #place input into cells

    #propagate signals across cells using blocks of filters
    if isinstance(norm, list) and len(norm) == 5:
        ema_mean_arr = norm[2]
        ema_var_arr = norm[3]

    for j in range(blocks):
        #propagate per block
        current_mode = "EXPAND" if (expand and blocks > 1) else "COMPRESS"
        for i in range(depth_block): #0 to depth-1, depth in total with no overlap required
            index = j*depth_block+i #unused atm
            cells, ema_mean_arr, ema_var_arr = _forward_full_fast_compact(index, cells, weights, biases, states, filters[j,0], activate, norm, dn, window_strides, mode=current_mode, is_training=is_training) #not inplace
            if isinstance(norm, list) and len(norm) == 5:
                norm = [norm[0], norm[1], ema_mean_arr, ema_var_arr, norm[4]]

    #final propagation using final filter, compress back into single channel
    cells, ema_mean_arr, ema_var_arr = _forward_full_fast_compact(depth, cells, weights, biases, states, filters[blocks,0], final_activate, norm, dn, window_strides, mode="COMPRESS", is_training=is_training) #final prop and activate
    if isinstance(norm, list) and len(norm) == 5:
        norm = [norm[0], norm[1], ema_mean_arr, ema_var_arr, norm[4]]
    # print("index", index, "cells shape", cells.shape)
    if not final_activate == None:
        cells = final_activate(cells) #force final activation, assume output layer

    if isinstance(norm, list) and len(norm) == 5:
        return cells, jax.lax.dynamic_slice(cells, out_offset, out_size), norm[2], norm[3]
    return cells, jax.lax.dynamic_slice(cells, out_offset, out_size)

# Memory-efficient version for Deep Learning training (Backward pass memory savings)
# Applying jax.checkpoint (remat) inside jax.jit ensures intermediate FFT arrays are discarded
# and recomputed during backpropagation, saving massive amounts of VRAM.
_propagate_full_mlp_blocks_fast_jit = jax.jit(
    jax.checkpoint(
        _propagate_full_mlp_blocks_fast,
        static_argnums=static_argnums_mlp_blocks
    ),
    static_argnums=static_argnums_mlp_blocks
)

def _propagate_full_mlp_blocks_fast_compact(input, offset, out_offset, out_size, depth, blocks, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides=(1,1), is_training=True, expand=False):
    """Memory-efficient column forward operator for neural network training."""
    return _propagate_full_mlp_blocks_fast_jit(input, offset, out_offset, out_size, depth, blocks, cells, weights, biases, states, filters, activate, final_activate, norm, dn, window_strides, is_training, expand)

# forward step operators
@partial(jax.jit, static_argnums=(5,6,7))
def _forward(cells, weights, biases, states, nethood, activate, norm, dn):
    """
    Compute the forward pass by computing each neuron depending on its state.
    """
    #get weighted sum of neighbors using convolution
    cells = jax.lax.conv_general_dilated(cells[jnp.newaxis,:,:,jnp.newaxis], # lhs = NHWC image tensor
                                        nethood,  # rhs = OHWI conv kernel tensor
                                        (1,1),  # window strides
                                        'SAME', # padding mode
                                        (1,1),  # lhs/image dilation
                                        (1,1),  # rhs/kernel dilation
                                        dn)     # dimension_numbers = lhs, rhs, out dimension permutation
    cells = cells[0,:,:,0] # NHWC

    if norm:
        mean_axes = tuple(i for i in range(len(dn.lhs_spec)-2)) #nD cells need norm'ing, ignore B and C
        # print("mean axes", mean_axes)
        #norm, note no B in result due to vectorization
        cells = jax.nn.standardize(cells, axis=mean_axes)

    #norm states to 0-1 range
    states = jax.nn.sigmoid(states)
    # sub_states = (sub_states - sub_states.min()) / (sub_states.max() - sub_states.min())

    #check states
    #if passive apply activation, else signal state so weight
    cells = jnp.where(jnp.logical_and(states>=STATE_SIGNAL,states<STATE_ACTIVE), cells*weights, cells) #weight if signal
    if activate == None:
        cells = jnp.where(states>=STATE_ACTIVE, cells+biases, cells) #activate if passive, more accurate
    else:
        cells = jnp.where(states>=STATE_ACTIVE, activate(cells+biases), cells) #activate if passive, more accurate
    # cells = jnp.where(states==2, activate(cells+biases), cells) #activate if passive, faster

    return cells

@partial(jax.jit, static_argnums=(0,6,7,8,9,10))
def _forward_column(index, cells, weights, biases, states, filters, activate, norm, dn, window_strides=(1,1), mode="COMPRESS"):
    """
    Compute the forward pass by computing each neuron depending on its state for a subregion.

    This version computes the forward WRT a central row/column (provided by index) with the nethood applied to it
    according to the width of the nethood. For a nethood of width 3, that means central row/column
    plus one before and one after it.

    All operations are done on the sub-region defined by width of nethood and central index

    Supports row/column in 2D and second-last/last hyperplane in nD

    Assumes filters are odd sized @todo generalise
    """
    filter_shape = filters.shape[1:-1] #shape is (1,rows,cols,...,1), remove first and last dims
    width = filter_shape[-1] # last dim, column in 2D
    height_pad_shape = tuple(map(lambda x: (x-1)//2, filter_shape[:-1])) # half of other dims of filter
    height_pad_shape = height_pad_shape + (0,) + (0,) #padding for each dim plus nth dim and add filter dim
    height_pad = tuple([(pad_value, pad_value) for pad_value in height_pad_shape]) #pad before and after
    width_pad = max(1, (width-1)//2) # half of last dim of filter
    # print("index",index,"width_pad",width_pad, "height_pad_shape", height_pad_shape,"height_pad",height_pad)

    #selects hyper plane (n-1)D at last dim
    sub_weights = weights[...,index-width_pad:index-width_pad+width]
    sub_biases = biases[...,index-width_pad:index-width_pad+width]
    sub_states = states[...,index-width_pad:index-width_pad+width]
    # print("sub_states",sub_states.shape)

    #process only width of kernel and restore to nD if needed
    groups_count = 1
    # groups_count = filters.shape[-1]//cells.shape[-1]
    # print("groups_count", groups_count, "cells shape", cells.shape)
    if len(cells.shape) == len(filter_shape): #if 3D
        #extract sub array of filter width to conv
        # print("cells",cells.shape)
        result_block = cells[...,index-width_pad:index-width_pad+width]
        result_block = result_block[...,jnp.newaxis] # add dim to support filter channel
        #~ groups_count = 1
    else:
        #extract sub array of filter width to conv, and preserve the channel dim
        result_block = cells[...,index-width_pad:index-width_pad+width,:]
        #~ groups_count = filters.shape[-1]

    sub_weights = sub_weights[...,jnp.newaxis] #promotes to column vec (2D) or back to nD and support filter channel
    sub_biases = sub_biases[...,jnp.newaxis] #promotes to column vec (2D) or back to nD and support filter channel
    sub_states = sub_states[...,jnp.newaxis] #promotes to column vec (2D) or back to nD and support filter channel

    # print("result_block",result_block.shape, "filters", filters.shape)
    #valid mode approach in all dims, reduced complexity by deliberately squishing last hyperplane as well to current index
    #handles boundaries better?
    result_block = _fast_conv(result_block, filters, dn, window_strides, padding_mode="SAME", feature_group_count=groups_count)
    #Fast Fourier conv
    # result_block = _fast_conv_3D(result_block, filters, pad_volume=False, normalize=True, apply_antialias=False)
    # print("result_block conv",result_block.shape)
    # result_block = jnp.pad(result_block, height_pad) #pad rows to undo valid conv in rows
    #Same mode approach
    # result_block = _fast_conv(result_block, filters, dn, padding_mode="SAME")
    # result_block = result_block[:,(width_pad+1)-1].reshape(-1,1) #reshape promotes to column vec, -1 for 0 indexing
    # print("result_block padded",result_block.shape)

    if norm: # or isinstance(norm, list):
        mean_axes = tuple(i for i in range(len(dn.lhs_spec)-2)) #nD cells need norm'ing, ignore B and C
        # print("mean axes", mean_axes)
        #norm, note no B in result due to vectorization
        result_block = jax.nn.standardize(result_block, axis=mean_axes)
        # result_block = utils.iqr_median_normalize(result_block, axis=mean_axes)
        # mean_axes = tuple(i for i in range(len(dn.lhs_spec)-3)) #nD cells need norm'ing, ignore B and C
        # gamma for scaled weights standardization equivalent to batch norm
        # Calculate scaling factor for ReLU
        # gamma = utils.get_relu_gain_gamma([sub_weights.shape[-3]])
        # sub_weights = jax.nn.standardize(sub_weights, axis=-3)
        # if isinstance(norm, list): #assume layer norm wanted
        #     scale = norm[0][index] #usually learnt per layer
        #     shift = norm[1][index] #usually learnt per layer
        #     # print("layer norm:", scale, shift)
        #     result_block = result_block * scale + shift #scale and shift distribution

    #~ result_block = jnp.pad(result_block, height_pad) #pad rows to undo valid conv in rows
    #we do this after norm to ensure padding does not skew the norm after valid approach

    #norm states to 0-1 range
    sub_states = jax.nn.sigmoid(sub_states)
    #~ sub_states = jax.nn.hard_sigmoid(sub_states)
    #~ sub_states = jax.nn.leaky_relu(sub_states)
    #~ sub_states = utils.min_max_normalize(sub_states)

    #check states
    #if passive apply activation, else signal state so weight, let the grad descent decide
    #the operation is akin to adversarial states
    # print("result_block",result_block.shape, "sub_weights", sub_weights.shape)
    if activate == None:
        result_block = result_block*sub_weights*(1.0-sub_states) + (result_block+sub_biases)*sub_states
    else:
        result_block = result_block*sub_weights*(1.0-sub_states) + activate(result_block+sub_biases)*sub_states

    if mode.lower() == "compress":
        return jnp.mean(result_block, axis=-1) # averaged across filters, back to 1 channel dim
    elif mode.lower() == "expand":
        return result_block
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'COMPRESS' or 'EXPAND'.")

_static_argnums_column = (0,6,8,9,10,11)
@partial(jax.jit, static_argnums=_static_argnums_column)
def _forward_column_fast(index, cells, weights, biases, states, filters, activate, norm, dn, window_strides=(1,1), mode="COMPRESS", is_training=True, norm_index=0, depth_block=1):
    """
    Compute the forward pass by computing each neuron depending on its state for a subregion.

    This version computes the forward WRT a central row/column (provided by index) with the nethood applied to it
    according to the width of the nethood. For a nethood of width 3, that means central row/column
    plus one before and one after it.

    All operations are done on the sub-region defined by width of nethood and central index

    Supports row/column in 2D and second-last/last hyperplane in nD

    Assumes filters are odd sized @todo generalise and doesn't have leading axis
    """
    filter_shape = filters.shape[:-1] #shape is (rows,cols,...,1), remove last dims
    width = filter_shape[-1] # last dim, column in 2D
    height_pad_shape = tuple(map(lambda x: (x-1)//2, filter_shape[:-1])) # half of other dims of filter
    height_pad_shape = height_pad_shape + (0,) + (0,) #padding for each dim plus nth dim and add filter dim
    height_pad = tuple([(pad_value, pad_value) for pad_value in height_pad_shape]) #pad before and after
    width_pad = max(1, (width-1)//2) # half of last dim of filter
    # print("index",index,"width_pad",width_pad, "height_pad_shape", height_pad_shape,"height_pad",height_pad)

    #selects hyper plane (n-1)D at last dim
    sub_weights = weights[...,index-width_pad:index-width_pad+width]
    sub_biases = biases[...,index-width_pad:index-width_pad+width]
    sub_states = states[...,index-width_pad:index-width_pad+width]
    # print("sub_states",sub_states.shape)

    #process only width of kernel and restore to nD if needed
    if len(cells.shape) == len(filter_shape): #if 3D
        #extract sub array of filter width to conv
        # print("cells",cells.shape)
        result_block = cells[...,index-width_pad:index-width_pad+width]
        result_block = result_block[...,jnp.newaxis] # add dim to support filter channel
    else:
        #extract sub array of filter width to conv, and preserve the channel dim
        result_block = cells[...,index-width_pad:index-width_pad+width,:]

    #restore to nD
    sub_weights = sub_weights[...,jnp.newaxis] #promotes to column vec (2D) or back to nD and support filter channel
    sub_biases = sub_biases[...,jnp.newaxis] #promotes to column vec (2D) or back to nD and support filter channel
    sub_states = sub_states[...,jnp.newaxis] #promotes to column vec (2D) or back to nD and support filter channel

    # print("result_block",result_block.shape, "filters", filters.shape)
    #Fast Fourier conv
    # result_block = _fast_conv_3D(result_block, filters, padding_mode="VALID", normalize=True, apply_antialias=False)
    result_block = _fast_conv_3D_compact(result_block, filters, padding_mode="CIRCULAR", normalize=True, apply_antialias=False)
    # print("result_block conv",result_block.shape)

    ema_mean_arr = None
    ema_var_arr = None
    if isinstance(norm, list) and len(norm) == 5:
        scale, shift, ema_mean_arr, ema_var_arr, _ = norm
        if is_training:
            # Batch norm over batch dimension using collective ops
            mean_axes = tuple(i for i in range(len(dn.lhs_spec)-2)) # spatial axes
            spatial_mean = jnp.mean(result_block, axis=mean_axes, keepdims=True)
            batch_mean = jax.lax.pmean(spatial_mean, axis_name='batch')

            spatial_var = jnp.mean((result_block - batch_mean)**2, axis=mean_axes, keepdims=True)
            batch_var = jax.lax.pmean(spatial_var, axis_name='batch')

            result_block = (result_block - batch_mean) / jnp.sqrt(batch_var + 1e-5)

            # Update EMA
            update_rate = 0.1 / depth_block
            new_ema_mean = (1.0 - update_rate) * ema_mean_arr[norm_index] + update_rate * jnp.squeeze(batch_mean)
            new_ema_var = (1.0 - update_rate) * ema_var_arr[norm_index] + update_rate * jnp.squeeze(batch_var)
            ema_mean_arr = ema_mean_arr.at[norm_index].set(new_ema_mean)
            ema_var_arr = ema_var_arr.at[norm_index].set(new_ema_var)
        else:
            # Inference mode
            mean_val = ema_mean_arr[norm_index]
            var_val = ema_var_arr[norm_index]
            for _ in range(len(result_block.shape) - 1):
                mean_val = jnp.expand_dims(mean_val, axis=0)
                var_val = jnp.expand_dims(var_val, axis=0)
            result_block = (result_block - mean_val) / jnp.sqrt(var_val + 1e-5)

        result_block = result_block * scale[norm_index] + shift[norm_index]
    elif norm:
        # Fallback to older standardize (though rarely used now)
        mean_axes = tuple(i for i in range(len(dn.lhs_spec)-2))
        result_block = jax.nn.standardize(result_block, axis=mean_axes)
        if isinstance(norm, list) and len(norm) == 2:
            scale = norm[0][norm_index]
            shift = norm[1][norm_index]
            result_block = result_block * scale + shift

    #norm states to 0-1 range
    sub_states = jax.nn.sigmoid(sub_states)
    #~ sub_states = jax.nn.hard_sigmoid(sub_states)
    #~ sub_states = jax.nn.leaky_relu(sub_states)
    #~ sub_states = utils.min_max_normalize(sub_states)

    #check states
    #if passive apply activation, else signal state so weight, let the grad descent decide
    #the operation is akin to adversarial states
    # print("result_block",result_block.shape, "sub_weights", sub_weights.shape)
    if activate == None:
        result_block = result_block*sub_weights*(1.0-sub_states) + (result_block+sub_biases)*sub_states
    else:
        result_block = result_block*sub_weights*(1.0-sub_states) + activate(result_block+sub_biases)*sub_states

    if mode.lower() == "compress":
        return jnp.mean(result_block, axis=-1), ema_mean_arr, ema_var_arr # averaged across filters, back to 1 channel dim
    elif mode.lower() == "expand":
        return result_block, ema_mean_arr, ema_var_arr
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'COMPRESS' or 'EXPAND'.")

# Memory-efficient version for Deep Learning training (Backward pass memory savings)
# Applying jax.checkpoint (remat) inside jax.jit ensures intermediate FFT arrays are discarded
# and recomputed during backpropagation, saving massive amounts of VRAM.
_forward_column_fast_jit = jax.jit(
    jax.checkpoint(
        _forward_column_fast,
        static_argnums=_static_argnums_column
    ),
    static_argnums=_static_argnums_column
)

def _forward_column_fast_compact(index, cells, weights, biases, states, filters, activate, norm, dn, window_strides=(1,1), mode="COMPRESS", is_training=True, norm_index=0, depth_block=1):
    """Memory-efficient column forward operator for neural network training."""
    return _forward_column_fast_jit(index, cells, weights, biases, states, filters, activate, norm, dn, window_strides, mode, is_training, norm_index, depth_block)

@partial(jax.jit, static_argnums=(0,6,7,8,9))
def _forward_row(index, cells, weights, biases, states, filters, activate, norm, dn, window_strides=(1,1)):
    """
    Compute the forward pass by computing each neuron depending on its state for a subregion.

    This version computes the forward WRT a central row/column (provided by index) with the nethood applied to it
    according to the width of the nethood. For a nethood of width 3, that means central row/column
    plus one before and one after it.

    All operations are done on the sub-region defined by width of nethood and central index

    Supports row/column in 2D and second-last/last hyperplane in nD

    Assumes filters are odd sized @todo generalise
    """
    filter_shape = filters.shape[1:-1] #shape is (1,rows,cols,...,1), remove first and last dims
    width = filter_shape[-2] # second-last dim, row in 2D
    height_pad_shape = tuple(map(lambda x: (x-1)//2, filter_shape[:-2])) # half of other dims of filter, no padding in second last dim
    height_pad_shape = height_pad_shape + (0,) + ((filter_shape[-1]-1)//2,) + (0,) #padding for each dim and add filter dim
    height_pad = tuple([(pad_value, pad_value) for pad_value in height_pad_shape]) #pad before and after
    width_pad = (width-1)//2 # half of last dim of filter
    # print("index",index,"width_pad",width_pad, "height_pad_shape", height_pad_shape,"height_pad",height_pad)

    #selects hyper plane (n-1)D at second-last dim
    # sub_cells = cells[:,index-1:index-1+width]
    sub_weights = weights[...,index,:]
    sub_biases = biases[...,index,:]
    sub_states = states[...,index,:]
    # print("sub_states",sub_states.shape)

    #extract sub array of filter width to conv
    # print("cells",cells.shape)
    result_block = cells[...,index-width_pad:index-width_pad+width,:]
    # print("result_block",result_block.shape)

    #restore to nD
    result_block = result_block[...,jnp.newaxis] # add dim to support filter channel
    sub_weights = sub_weights[...,jnp.newaxis,:,jnp.newaxis] #promotes to row vec (2D) or back to nD and support filter channel
    sub_biases = sub_biases[...,jnp.newaxis,:,jnp.newaxis] #promotes to row vec (2D) or back to nD and support filter channel
    sub_states = sub_states[...,jnp.newaxis,:,jnp.newaxis] #promotes to row vec (2D) or back to nD and support filter channel

    # if norm:
    #     # gamma for scaled weights standardization equivalent to batch norm
    #     # Calculate scaling factor for ReLU
    #     # gamma = utils.get_relu_gain_gamma([filters.shape[-2]])

    #     # Apply the standardization function to the weights
    #     mean_axes = tuple(i for i in range(len(filters.shape)-1)) #nD cells need norm'ing, ignore B and C
    #     # print("mean axes", mean_axes, "filters", filters.shape)
    #     filters = jax.nn.standardize(filters, axis=mean_axes)

    # print("result_block",result_block.shape, "filters", filters.shape)
    #valid mode approach, reduced complexity, handles boundaries better?
    result_block = _fast_conv(result_block, filters, dn, window_strides, padding_mode="VALID")
    #Fast Fourier conv
    # result_block = _fast_conv_3D(result_block, filters, pad_volume=False, normalize=True, apply_antialias=False)
    # result_block = jnp.pad(result_block, height_pad) #pad rows to undo valid conv in rows
    #Same mode approach
    # result_block = _fast_conv(result_block, filters, dn, padding_mode="SAME")
    # result_block = result_block[:,(width_pad+1)-1].reshape(-1,1) #reshape promotes to column vec, -1 for 0 indexing
    # print("result_block padded",result_block.shape)

    if norm or isinstance(norm, list):
        mean_axes = tuple(i for i in range(len(dn.lhs_spec)-2)) #nD cells need norm'ing, ignore B and C
        # print("mean axes", mean_axes)
        #norm, note no B in result due to vectorization
        result_block = jax.nn.standardize(result_block, axis=mean_axes)
        # result_block = utils.iqr_median_normalize(result_block, axis=mean_axes)
        # mean_axes = tuple(i for i in range(len(dn.lhs_spec)-3)) #nD cells need norm'ing, ignore B and C
        # gamma for scaled weights standardization equivalent to batch norm
        # Calculate scaling factor for ReLU
        # gamma = utils.get_relu_gain_gamma([sub_weights.shape[-3]])
        # sub_weights = jax.nn.standardize(sub_weights, axis=-3)
        if isinstance(norm, list): #assume layer norm wanted
            scale = norm[0][index] #usually learnt per layer
            shift = norm[1][index] #usually learnt per layer
            # print("layer norm:", scale, shift)
            result_block = result_block * scale + shift #scale and shift distribution

    result_block = jnp.pad(result_block, height_pad) #pad rows to undo valid conv in rows
    #we do this after norm to ensure padding does not skew the norm after valid approach

    #norm states to 0-1 range
    sub_states = jax.nn.sigmoid(sub_states)
    # sub_states = (sub_states - sub_states.min()) / (sub_states.max() - sub_states.min())

    #check states
    #if passive apply activation, else signal state so weight, let the grad descent decide
    #the operation is akin to adversarial states
    # print("result_block",result_block.shape, "sub_weights", sub_weights.shape)
    if activate == None:
        result_block = result_block*sub_weights*(1.0-sub_states) + (result_block+sub_biases)*sub_states
    else:
        result_block = result_block*sub_weights*(1.0-sub_states) + activate(result_block+sub_biases)*sub_states

    return jnp.mean(result_block, axis=-1) # averaged across filters, back to 1 channel dim

@partial(jax.jit, static_argnums=(0,6,7,8,9,10))
def _forward_full(width, cells, weights, biases, states, filters, activate, norm, dn, window_strides=(1,1), mode="COMPRESS"):
    """
    Compute the forward pass by computing each neuron depending on its state for a subregion of width.
    Assumes subregion width is centred within universe WRT columns and width is even.

    This version computes the forward across the full array with the nethood applied to it
    The result is a full 2D convolution with the nethood, which can be expensive.
    Multiple nethoods or filters are supported, returns the mean result of these filters

    All operations are done on the entire universe.

    Assumes filters are odd sized @todo generalise
    """
    # centre = cells.shape[1]//2
    # padding = filters.shape[1]-1
    filter_shape = filters.shape[1:-1] #shape is (1,rows,cols,...,1), remove first and last dims
    height_pad_shape = tuple(map(lambda x: (x-1)//2, filter_shape)) # half of all dims of filter
    height_pad_shape = height_pad_shape + (0,) #padding for each dim plus filter dim
    height_pad = tuple([(pad_value, pad_value) for pad_value in height_pad_shape]) #pad before and after
    # print("height_pad_shape", height_pad_shape,"height_pad",height_pad)

    # sub_weights = weights[:,centre-width//2-padding:centre+width//2+padding]
    # sub_biases = biases[:,centre-width//2-padding:centre+width//2+padding]
    # sub_states = states[:,centre-width//2-padding:centre+width//2+padding]

    #broadcasting used later for mul, can be memory intensive for large cell arrays, see compact version
    groups_count = 1
    # groups_count = filters.shape[-1]
    if len(cells.shape) == len(filter_shape): #if 3D
        cells = cells[...,jnp.newaxis] #add features/channels dim
        # groups_count = 1
    weights = weights[...,jnp.newaxis]
    biases = biases[...,jnp.newaxis]
    states = states[...,jnp.newaxis]

    #conv to eval neighbourhood
    # padding VALID mode approach, reduced complexity, handles boundaries better?
    result = _fast_conv(cells, filters, dn, window_strides, padding_mode="VALID", feature_group_count=groups_count)
    # result = jnp.pad(result, height_pad) #pad rows to undo valid conv in rows and cols
    # result = _fast_conv(cells[:,centre-width//2-padding:centre+width//2+padding], filters, dn)
    # padding SAME mode approach
    # result = _fast_conv(cells, filters, dn, window_strides) #produces result channel dims matching filters channel dims
    # print("Conv Result Shape:",result.shape)

    if norm:
        mean_axes = tuple(i for i in range(len(dn.lhs_spec)-2)) #nD cells need norm'ing, ignore B and C
        # print("mean axes", mean_axes, "result", result)
        #norm, note no B in result due to vectorization
        result = jax.nn.standardize(result, axis=mean_axes)
        # filters = utils.normalize_l2(filters, axis=mean_axes) # unit norm

    result = jnp.pad(result, height_pad) #pad rows to undo valid conv in rows and cols
    #we do this after norm to ensure padding does not skew the norm after valid approach

    #norm states to 0-1 range
    states = jax.nn.sigmoid(states)
    #~ states = jax.nn.hard_sigmoid(states)
    #~ states = jax.nn.gelu(states)
    #~ states = utils.min_max_normalize(states)

    #check states
    #if passive apply activation, else signal state so weight, let the grad descent decide
    #the operation is akin to adversarial states
    if activate == None:
        result = result*weights*(1.0-states) + (result+biases)*states
    else:
        result = result*weights*(1.0-states) + activate(result+biases)*states

    if mode.lower() == "compress":
        return jnp.mean(result, axis=-1) # averaged across filters, back to 1 channel dim
    elif mode.lower() == "expand":
        return result
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'COMPRESS' or 'EXPAND'.")

# Define the complete list of static argument indices so JAX knows not to trace them as dynamic arrays
_static_argnums_full = (0,6,8,9,10,11)
@partial(jax.jit, static_argnums=_static_argnums_full)
def _forward_full_fast(width, cells, weights, biases, states, filters, activate, norm, dn, window_strides=(1,1), mode="COMPRESS", is_training=True):
    """
    Compute the forward pass by computing each neuron depending on its state for a subregion of width.
    Assumes subregion width is centred within universe WRT columns and width is even.

    This version computes the forward across the full array with the nethood applied to it
    The result is a full 2D convolution with the nethood, which can be expensive.
    Multiple nethoods or filters are supported, returns the mean result of these filters

    FFT based convolution version with memory constraints on back propagation

    All operations are done on the entire universe.

    Assumes filters are odd sized @todo generalise and doesn't have leading axis
    """
    # centre = cells.shape[1]//2
    # padding = filters.shape[1]-1
    filter_shape = filters.shape[:-1] #shape is (rows,cols,...,1), remove last dims
    # height_pad_shape = tuple(map(lambda x: (x-1)//2, filter_shape)) # half of all dims of filter
    # height_pad_shape = height_pad_shape + (0,) #padding for each dim plus filter dim
    # height_pad = tuple([(pad_value, pad_value) for pad_value in height_pad_shape]) #pad before and after
    # print("height_pad_shape", height_pad_shape,"height_pad",height_pad)

    # sub_weights = weights[:,centre-width//2-padding:centre+width//2+padding]
    # sub_biases = biases[:,centre-width//2-padding:centre+width//2+padding]
    # sub_states = states[:,centre-width//2-padding:centre+width//2+padding]

    #broadcasting used later for mul, can be memory intensive for large cell arrays, see compact version
    if len(cells.shape) == len(filter_shape): #if 3D
        cells = cells[...,jnp.newaxis] #add features/channels dim
    weights = weights[...,jnp.newaxis]
    biases = biases[...,jnp.newaxis]
    states = states[...,jnp.newaxis]

    #conv to eval neighbourhood
    #Fast Fourier conv
    # result = _fast_conv_3D(cells, filters, pad_volume=False, normalize=True, apply_antialias=False)
    result = _fast_conv_3D_compact(cells, filters, padding_mode="CIRCULAR", normalize=True, apply_antialias=False)
    # result = _fast_conv_3D_compact(cells, filters, padding_mode="VALID", normalize=True, apply_antialias=False)

    ema_mean_arr = None
    ema_var_arr = None
    if isinstance(norm, list) and len(norm) == 5:
        scale, shift, ema_mean_arr, ema_var_arr, _ = norm
        if is_training:
            mean_axes = tuple(i for i in range(len(dn.lhs_spec)-2)) # spatial axes
            spatial_mean = jnp.mean(result, axis=mean_axes, keepdims=True)
            batch_mean = jax.lax.pmean(spatial_mean, axis_name='batch')

            spatial_var = jnp.mean((result - batch_mean)**2, axis=mean_axes, keepdims=True)
            batch_var = jax.lax.pmean(spatial_var, axis_name='batch')

            result = (result - batch_mean) / jnp.sqrt(batch_var + 1e-5)

            new_ema_mean = 0.9 * ema_mean_arr[width] + 0.1 * jnp.squeeze(batch_mean)
            new_ema_var = 0.9 * ema_var_arr[width] + 0.1 * jnp.squeeze(batch_var)
            ema_mean_arr = ema_mean_arr.at[width].set(new_ema_mean)
            ema_var_arr = ema_var_arr.at[width].set(new_ema_var)
        else:
            mean_val = ema_mean_arr[width]
            var_val = ema_var_arr[width]
            for _ in range(len(result.shape) - 1):
                mean_val = jnp.expand_dims(mean_val, axis=0)
                var_val = jnp.expand_dims(var_val, axis=0)
            result = (result - mean_val) / jnp.sqrt(var_val + 1e-5)

        result = result * scale[width] + shift[width]
    elif bool(norm) or isinstance(norm, list):
        if isinstance(norm, list): #assume layer norm wanted
            scale = norm[0][width] #usually learnt per layer
            shift = norm[1][width] #usually learnt per layer
            result = result * scale + shift

    # result = jnp.pad(result, height_pad) #pad rows to undo valid conv in rows and cols
    #we do this after norm to ensure padding does not skew the norm after valid approach

    #norm states to 0-1 range
    states = jax.nn.sigmoid(states)
    #~ states = jax.nn.hard_sigmoid(states)
    #~ states = jax.nn.leaky_relu(states)
    #~ states = utils.min_max_normalize(states)

    #check states
    #if passive apply activation, else signal state so weight, let the grad descent decide
    #the operation is akin to adversarial states
    if activate == None:
        result = result*weights*(1.0-states) + (result+biases)*states
    else:
        result = result*weights*(1.0-states) + activate(result+biases)*states

    if mode.lower() == "compress":
        return jnp.mean(result, axis=-1), ema_mean_arr, ema_var_arr # averaged across filters, back to 1 channel dim
    elif mode.lower() == "expand":
        return result, ema_mean_arr, ema_var_arr
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'COMPRESS' or 'EXPAND'.")

# Memory-efficient version for Deep Learning training (Backward pass memory savings)
# Applying jax.checkpoint (remat) inside jax.jit ensures intermediate FFT arrays are discarded
# and recomputed during backpropagation, saving massive amounts of VRAM.
_forward_full_fast_jit = jax.jit(
    jax.checkpoint(
        _forward_full_fast,
        static_argnums=_static_argnums_full
    ),
    static_argnums=_static_argnums_full
)

def _forward_full_fast_compact(width, cells, weights, biases, states, filters, activate, norm, dn, window_strides=(1,1), mode="COMPRESS", is_training=True):
    """Memory-efficient full forward operator for neural network training."""
    return _forward_full_fast_jit(width, cells, weights, biases, states, filters, activate, norm, dn, window_strides, mode, is_training)
