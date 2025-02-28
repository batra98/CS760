def count_nodes(root):  
  if root is None:
    return 0
  if root.label is not None:
    return 1
  else:
    return 1 + count_nodes(root.left) + count_nodes(root.right)


