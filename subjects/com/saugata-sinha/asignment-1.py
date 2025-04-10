class LayerAnalyzer:
    def __init__(self, layer_type, input_shape, num_filters=None, filter_size=None, padding=0, stride=1):
        self.layer_type = layer_type.lower()
        self.input_shape = input_shape  # (Height, Width, Channels)
        self.num_filters = num_filters
        self.filter_size = filter_size  # (Height, Width)
        self.padding = padding
        self.stride = stride

    def compute_output_dimensions(self):
        if self.layer_type == "conv":
            h_in, w_in, c_in = self.input_shape
            f_h, f_w = self.filter_size
            h_out = ((h_in - f_h + 2 * self.padding) // self.stride) + 1
            w_out = ((w_in - f_w + 2 * self.padding) // self.stride) + 1
            return (h_out, w_out, self.num_filters)
        elif self.layer_type == "dense":
            return (self.num_filters,)  # Assuming a fully connected layer
        else:
            return self.input_shape  # For pooling, dimensions remain unchanged except spatial reduction

    def compute_trainable_params(self):
        if self.layer_type == "conv":
            return (self.filter_size[0] * self.filter_size[1] * self.input_shape[2] + 1) * self.num_filters  # Includes bias
        elif self.layer_type == "dense":
            return (self.input_shape[0] + 1) * self.num_filters  # Fully connected layer, includes bias
        else:
            return 0  # Pooling layers have no trainable parameters

    def compute_computations(self):
        if self.layer_type == "conv":
            h_out, w_out, _ = self.compute_output_dimensions()
            ops_per_filter = self.filter_size[0] * self.filter_size[1] * self.input_shape[2]  # Multiplications per filter
            total_products = ops_per_filter * h_out * w_out * self.num_filters
            total_sums = total_products - (h_out * w_out * self.num_filters)  # Sum follows multiplications
            return total_products, total_sums
        elif self.layer_type == "dense":
            total_products = self.input_shape[0] * self.num_filters
            total_sums = total_products - self.num_filters
            return total_products, total_sums
        else:
            return 0, 0  # Pooling layers only perform reduction operations

    def analyze_layer(self):
        output_dim = self.compute_output_dimensions()
        trainable_params = self.compute_trainable_params()
        num_products, num_sums = self.compute_computations()
        return {
            "Output Dimensions": output_dim,
            "Trainable Parameters": trainable_params,
            "Computational Cost": {"Products": num_products, "Sums": num_sums}
        }

# Example usage
if __name__ == "__main__":
    layer = LayerAnalyzer(layer_type="conv", input_shape=(32, 32, 3), num_filters=64, filter_size=(3, 3), padding=1, stride=1)
    results = layer.analyze_layer()
    print(results)
