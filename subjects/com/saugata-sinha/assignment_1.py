def layer_computation(layer_type, layer_params, input_dim):
    """
    Compute architectural details for a neural network layer.

    Args:
        layer_type: str - 'conv' for convolutional, 'dense' for fully connected
        layer_params: dict - parameters for the layer
            For conv: {'h': height, 'w': width, 'stride': stride,
                      'padding': padding, 'num_filters': number of filters}
            For dense: {'nodes': number of nodes}
        input_dim: tuple - (channels, height, width) for conv,
                           (nodes,) for dense

    Returns:
        dict: {
            'output_dim': output dimensions,
            'trainable_params': number of trainable parameters,
            'dot_products': number of dot products,
            'total_ops': total number of operations (products + sums)
        }
    """
    result = {}

    if layer_type == 'conv':
        # Unpack parameters
        C_in = input_dim[0]
        H_in = input_dim[1]
        W_in = input_dim[2]
        F_h = layer_params['h']
        F_w = layer_params['w']
        stride = layer_params['stride']
        padding = layer_params['padding']
        num_filters = layer_params['num_filters']

        # Calculate output dimensions
        H_out = ((H_in - F_h + 2 * padding) // stride) + 1
        W_out = ((W_in - F_w + 2 * padding) // stride) + 1
        result['output_dim'] = (num_filters, H_out, W_out)

        # Number of trainable parameters
        # (weights + bias) for each filter
        result['trainable_params'] = num_filters * (C_in * F_h * F_w + 1)

        # Number of dot products (one per output pixel per filter)
        result['dot_products'] = num_filters * H_out * W_out

        # Total operations: each dot product is F_h*F_w*C_in multiplications
        # and (F_h*F_w*C_in - 1) additions
        ops_per_dot = F_h * F_w * C_in * 2 - 1
        result['total_ops'] = result['dot_products'] * ops_per_dot

    elif layer_type == 'dense':
        # Unpack parameters
        nodes_in = input_dim[0]
        nodes_out = layer_params['nodes']

        # Output dimension
        result['output_dim'] = (nodes_out,)

        # Number of trainable parameters (weights + bias)
        result['trainable_params'] = nodes_in * nodes_out + nodes_out

        # Number of dot products (one per output node)
        result['dot_products'] = nodes_out

        # Total operations: each dot product is nodes_in multiplications
        # and (nodes_in - 1) additions
        ops_per_dot = nodes_in * 2 - 1
        result['total_ops'] = result['dot_products'] * ops_per_dot

    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

    return result


def analyze_googlenet():
    # Input dimensions (RGB image)
    current_dim = (3, 224, 224)
    total_params = 0
    total_ops = 0

    # Layer definitions for GoogleNet
    layers = [
        # Initial convolutional layers
        {'type': 'conv', 'params': {'h': 7, 'w': 7, 'stride': 2, 'padding': 3, 'num_filters': 64}},
        {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 64}},
        {'type': 'conv', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 192}},

        # Inception modules (simplified - actual GoogleNet has more complex structure)
        # Inception 3a
        {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 64}},
        {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 96}},
        {'type': 'conv', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 128}},
        {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 16}},
        {'type': 'conv', 'params': {'h': 5, 'w': 5, 'stride': 1, 'padding': 2, 'num_filters': 32}},

        # Inception 3b (simplified)
        {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 128}},
        {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 128}},
        {'type': 'conv', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 192}},
        {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 32}},
        {'type': 'conv', 'params': {'h': 5, 'w': 5, 'stride': 1, 'padding': 2, 'num_filters': 96}},

        # More layers would follow in a complete analysis...

        # Final layers (simplified)
        {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 1024}},
        {'type': 'dense', 'params': {'nodes': 1000}}  # Assuming 1000-class classification
    ]

    print("GoogleNet Architecture Analysis:")
    print("=" * 50)
    print(f"Input dimension: {current_dim}")
    print("-" * 50)

    for i, layer in enumerate(layers):
        result = layer_computation(layer['type'], layer['params'], current_dim)
        print(f"Layer {i + 1} ({layer['type']}):")
        print(f"  Output dimension: {result['output_dim']}")
        print(f"  Trainable parameters: {result['trainable_params']:,}")
        print(f"  Dot products: {result['dot_products']:,}")
        print(f"  Total operations: {result['total_ops']:,}")
        print("-" * 30)

        total_params += result['trainable_params']
        total_ops += result['total_ops']
        current_dim = result['output_dim']

    print("=" * 50)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Total operations: {total_ops:,}")


# Run the analysis
analyze_googlenet()