# usage_example.py

from computation_load import ComputationLoad

def main():
    # Example 1: Convolutional Layer
    conv_params = {
        'h': 3,
        'w': 3,
        'stride': 1,
        'padding': 1,
        'num_filters': 64
    }
    conv_input_dim = (3, 224, 224)
    conv_result = ComputationLoad.layer_computation('conv', conv_params, conv_input_dim)
    print(f"  Convolutional Layer Result")
    print(f"  Output dimension: {conv_result['output_dim']}")
    print(f"  Trainable parameters: {conv_result['trainable_params']:,}")
    print(f"  Total operations: {conv_result['total_ops']:,}")

    print("="*80)
    # Example 2: Dense (Fully Connected) Layer
    dense_params = {'nodes': 1000}
    dense_input_dim = (4096,)
    dense_result = ComputationLoad.layer_computation('dense', dense_params, dense_input_dim)
    print(f"  Dense Layer Result:")
    print(f"  Output dimension: {conv_result['output_dim']}")
    print(f"  Trainable parameters: {conv_result['trainable_params']:,}")
    print(f"  Total operations: {conv_result['total_ops']:,}")

if __name__ == "__main__":
    main()
