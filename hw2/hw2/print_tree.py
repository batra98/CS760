import graphviz


def print_tree(node, depth=0):
  indent = "  " * depth

  if node.label is not None:
      print(f"{indent}Predict: {node.label}")
      return

  print(f"{indent}Feature {node.feature} >= {node.threshold}")

  print(f"{indent}---> True branch:")
  print_tree(node.left, depth + 1)

  print(f"{indent}---> False branch:")
  print_tree(node.right, depth + 1)




def visualize_tree(node, feature_names=None):
    def add_node(graph, node, node_id):
        
        if node.label is not None:
            label = f"Predict: {node.label}"
            graph.node(str(node_id), label, shape='ellipse', style='filled', color='lightblue')
        else:
            feature_name = f"Feature {node.feature}" if feature_names is None else feature_names[node.feature]
            label = f"{feature_name} >= {node.threshold}"
            graph.node(str(node_id), label, shape='box')

            left_id = node_id * 2 + 1
            graph.edge(str(node_id), str(left_id), "True")
            add_node(graph, node.left, left_id)

            right_id = node_id * 2 + 2
            graph.edge(str(node_id), str(right_id), "False")
            add_node(graph, node.right, right_id)

    graph = graphviz.Digraph()

    add_node(graph, node, 0)

    return graph


