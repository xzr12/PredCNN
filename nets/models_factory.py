from nets import predcnn


networks_map = {
                'predcnn': predcnn.predcnn,
                'predcnn_stride': predcnn.predcnn_stride,
               }


def construct_model(name, images, params, mask_true, num_hidden, filter_size, seq_length, input_length):
    '''
    Returns a sequence of generated frames
    Args:
        name: [predrnn_pp]
        params: dict for extra parameters of some models
        mask_true: for schedualed sampling.
        num_hidden: number of units in a lstm layer.
        filter_size: for convolutions inside lstm.
        seq_length: including ins and outs.
        input_length: for inputs.
    Returns:
        gen_images: a seq of frames.
        loss: [l2 / l1+l2].
    Raises:
        ValueError: If network `name` is not recognized.
    '''
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    func = networks_map[name]
    return func(images, params, mask_true, num_hidden, filter_size, seq_length, input_length)
