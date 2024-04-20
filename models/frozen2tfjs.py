# tensorflowjs_converter \
    # --input_format=tf_frozen_model \
    # --output_node_names='feature_fusion/concat_3,feature_fusion/Conv_7/Sigmoid:0' \
    # frozen_east_text_detection.pb \
    # web_model

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

# Load the frozen model
graph = load_graph('frozen_east_text_detection.pb')

# Get all the operations (nodes) in the graph
ops = graph.get_operations()

# Print the names of all nodes to inspect them
for op in ops:
    print(op.name)
