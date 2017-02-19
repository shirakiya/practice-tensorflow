import os
from tensorflow.python import pywrap_tensorflow


def print_tensors_in_checkpoint_file(file_name):
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print('tensor_name: ', '"{}"'.format(key))
        print('tensor: ', reader.get_tensor(key))


if __name__ == '__main__':
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'ckpt/model.ckpt-99')
    print_tensors_in_checkpoint_file(checkpoint_path)
