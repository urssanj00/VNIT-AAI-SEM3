from computation_load import ComputationLoad
def inception_module(input_dim, module_params):
    """
    Compute architectural details for an Inception module.

    Args:
        input_dim: tuple - (channels, height, width)
        module_params: dict - parameters for each branch
            {
                'branches': [
                    {'type': 'conv', 'params': {...}},  # 1x1 branch
                    {'type': 'conv', 'params': {...}},  # 3x3 branch
                    {'type': 'conv', 'params': {...}},  # 5x5 branch
                    {'type': 'pool', 'params': {...}}   # pooling branch
                ]
            }

    Returns:
        dict: same structure as layer_computation
    """
    branch_results = []

    for branch in module_params['branches']:
        if branch['type'] == 'pool':
            # For pooling branch, we need to add a 1x1 conv after pooling
            # First do pooling (no trainable params)
            pool_params = branch['params']
            H_in = input_dim[1]
            W_in = input_dim[2]
            F_h = pool_params['h']
            F_w = pool_params['w']
            stride = pool_params['stride']
            padding = pool_params['padding']

            H_out = ((H_in - F_h + 2 * padding) // stride) + 1
            W_out = ((W_in - F_w + 2 * padding) // stride) + 1
            pool_output_dim = (input_dim[0], H_out, W_out)

            # Then do 1x1 conv
            conv_params = {
                'h': 1, 'w': 1, 'stride': 1,
                'padding': 0, 'num_filters': pool_params['num_filters']
            }
            branch_result = ComputationLoad.layer_computation('conv', conv_params, pool_output_dim)
        else:
            branch_result = ComputationLoad.layer_computation(branch['type'], branch['params'], input_dim)

        branch_results.append(branch_result)

    # Concatenate all branches along the channel dimension
    total_filters = sum(br['output_dim'][0] for br in branch_results)
    output_dim = (total_filters,) + branch_results[0]['output_dim'][1:]

    # Sum all operations and parameters
    total_params = sum(br['trainable_params'] for br in branch_results)
    total_ops = sum(br['total_ops'] for br in branch_results)

    return {
        'output_dim': output_dim,
        'trainable_params': total_params,
        'dot_products': sum(br['dot_products'] for br in branch_results),
        'total_ops': total_ops
    }

def analyze_googlenet():
    current_dim = (3, 224, 224)  # Input RGB image
    total_params = 0
    total_ops = 0

    def pool_output(dim, k, s, p):
        c, h, w = dim
        h_out = ((h - k + 2 * p) // s) + 1
        w_out = ((w - k + 2 * p) // s) + 1
        return (c, h_out, w_out)

    layers = [
        {'type': 'conv', 'params': {'h': 7, 'w': 7, 'stride': 2, 'padding': 3, 'num_filters': 64}},  # 1
        {'type': 'pool', 'params': {'h': 3, 'w': 3, 'stride': 2, 'padding': 1}},                    # 2 - MaxPool
        {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 64}}, # 3
        {'type': 'conv', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 192}},# 4
        {'type': 'pool', 'params': {'h': 3, 'w': 3, 'stride': 2, 'padding': 1}},                    # 5 - MaxPool

        # Inception modules
        {'type': 'inception', 'params': {  # 6 - 3a
            'branches': [
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 64}},
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 96}},
                {'type': 'conv', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 128}},
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 16}},
                {'type': 'conv', 'params': {'h': 5, 'w': 5, 'stride': 1, 'padding': 2, 'num_filters': 32}},
                {'type': 'pool', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 32}}
            ]
        }},
        {'type': 'inception', 'params': {  # 7 - 3b
            'branches': [
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 128}},
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 128}},
                {'type': 'conv', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 192}},
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 32}},
                {'type': 'conv', 'params': {'h': 5, 'w': 5, 'stride': 1, 'padding': 2, 'num_filters': 96}},
                {'type': 'pool', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 64}}
            ]
        }},
        {'type': 'pool', 'params': {'h': 3, 'w': 3, 'stride': 2, 'padding': 1}},  # 8 - MaxPool

        {'type': 'inception', 'params': {  # 9 - 4a
            'branches': [
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 192}},
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 96}},
                {'type': 'conv', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 208}},
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 16}},
                {'type': 'conv', 'params': {'h': 5, 'w': 5, 'stride': 1, 'padding': 2, 'num_filters': 48}},
                {'type': 'pool', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 64}}
            ]
        }},
        {'type': 'inception', 'params': {  # 10 - 4b
            'branches': [
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 160}},
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 112}},
                {'type': 'conv', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 224}},
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 24}},
                {'type': 'conv', 'params': {'h': 5, 'w': 5, 'stride': 1, 'padding': 2, 'num_filters': 64}},
                {'type': 'pool', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 64}}
            ]
        }},
        {'type': 'inception', 'params': {  # 11 - 4c
            'branches': [
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 128}},
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 128}},
                {'type': 'conv', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 256}},
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 24}},
                {'type': 'conv', 'params': {'h': 5, 'w': 5, 'stride': 1, 'padding': 2, 'num_filters': 64}},
                {'type': 'pool', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 64}}
            ]
        }},
        {'type': 'inception', 'params': {  # 12 - 4d
            'branches': [
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 112}},
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 144}},
                {'type': 'conv', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 288}},
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 32}},
                {'type': 'conv', 'params': {'h': 5, 'w': 5, 'stride': 1, 'padding': 2, 'num_filters': 64}},
                {'type': 'pool', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 64}}
            ]
        }},
        {'type': 'inception', 'params': {  # 13 - 4e
            'branches': [
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 256}},
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 160}},
                {'type': 'conv', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 320}},
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 32}},
                {'type': 'conv', 'params': {'h': 5, 'w': 5, 'stride': 1, 'padding': 2, 'num_filters': 128}},
                {'type': 'pool', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 128}}
            ]
        }},
        {'type': 'pool', 'params': {'h': 3, 'w': 3, 'stride': 2, 'padding': 1}},  # 14 - MaxPool

        {'type': 'inception', 'params': {  # 15 - 5a
            'branches': [
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 256}},
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 160}},
                {'type': 'conv', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 320}},
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 32}},
                {'type': 'conv', 'params': {'h': 5, 'w': 5, 'stride': 1, 'padding': 2, 'num_filters': 128}},
                {'type': 'pool', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 128}}
            ]
        }},
        {'type': 'inception', 'params': {  # 16 - 5b
            'branches': [
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 384}},
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 192}},
                {'type': 'conv', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 384}},
                {'type': 'conv', 'params': {'h': 1, 'w': 1, 'stride': 1, 'padding': 0, 'num_filters': 48}},
                {'type': 'conv', 'params': {'h': 5, 'w': 5, 'stride': 1, 'padding': 2, 'num_filters': 128}},
                {'type': 'pool', 'params': {'h': 3, 'w': 3, 'stride': 1, 'padding': 1, 'num_filters': 128}}
            ]
        }},
        {'type': 'pool', 'params': {'h': 7, 'w': 7, 'stride': 1, 'padding': 0}},  # 17 - AvgPool
        {'type': 'dense', 'params': {'nodes': 1000}}  # 18 - Fully connected classifier
    ]

    print("GoogleNet Architecture Analysis (All 22 Layers):")
    print("=" * 70)
    print(f"Input dimension: {current_dim}")
    print("-" * 70)

    for i, layer in enumerate(layers):
        if layer['type'] == 'inception':
            result = inception_module(current_dim, layer['params'])
            print(f"Layer {i + 1} (Inception module):")
        elif layer['type'] == 'pool':
            result = {'output_dim': pool_output(current_dim, layer['params']['h'], layer['params']['stride'], layer['params']['padding']),
                      'trainable_params': 0, 'dot_products': 0, 'total_ops': 0}
            print(f"Layer {i + 1} (Pooling layer):")
        else:
            result = ComputationLoad.layer_computation(layer['type'], layer['params'], current_dim)
            print(f"Layer {i + 1} ({layer['type']}):")

        print(f"  Output dimension: {result['output_dim']}")
        print(f"  Trainable parameters: {result['trainable_params']:,}")
        print(f"  Total operations: {result['total_ops']:,}")
        print("-" * 50)

        total_params += result['trainable_params']
        total_ops += result['total_ops']
        current_dim = result['output_dim']

    print("=" * 70)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Total operations: {total_ops:,}")


# Run the analysis
analyze_googlenet()