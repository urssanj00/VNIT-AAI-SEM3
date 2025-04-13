
class ComputationLoad:

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
                branch_result = layer_computation('conv', conv_params, pool_output_dim)
            else:
                branch_result = layer_computation(branch['type'], branch['params'], input_dim)

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
