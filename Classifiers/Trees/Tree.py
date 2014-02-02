__author__ = 'simon.hughes'

class Node(object):
    def __init__(self, attribute_name, attribute_val, parentnode, fn_attribute_val_extractor):
        """
            attribute_name              :   str
                                                attribute name (on which node splits)
            attribute_value             :   any
                                                value of the node
            parentnode                  :   Node
                                                parent node
            fn_attribute_val_extractor  :   fn - (xs, attribute_name, parent_attribute_name, parent_attribute_val) => attribute_values
                                                A function that takes a list of examples,
                                                an attribute name, parent_attribute name and value,
                                                 and extracts a list of values


        """
        self.attribute_name = attribute_name
        self.attribute_val  = attribute_val
        self.parentnode = parentnode
        self.fn_attribute_val_extractor = fn_attribute_val_extractor
        self.dct_children = {}

    def add_child(self, child_node):
        self.dct_children[child_node.attribute_val] = child_node

    def eval(self, x):
        value = self.fn_attribute_val_extractor([x], self.attribute_name, self.parentnode.attribute_name, self.attribute_val)[0]
        return self.dct_children[value].eval(x)

    def __to_str__(self, level):
        items = self.dct_children.items()
        indent = "\n" + ("\t" * (level + 1))
        if len(items) == 0:
            str_children = ""
        else:
            if type(self.dct_children.keys()[0]) == bool:
                items = items[::-1]
            str_children = indent.join(map(lambda (k, v): str(k) + ":" + v.__to_str__(level+1) + "", items))

        return "Node[%s](%s%s%s)" % (self.attribute_name, indent, str_children, "\n" + ("\t" * (level)))

    def __repr__(self):
        return self.__to_str__(1)

class Leaf(object):
    """
    Leaf node. Stores the value at the leaf of a tree
    """
    def __init__(self, attribute_val, eval_val):
        """
            val     :   any type

            create a leaf node with value val:
        """
        self.attribute_val = attribute_val
        self.eval_val = eval_val

    def eval(self, x):
        return self.eval_val

    def __to_str__(self, level):
        return "Leaf: %s" % str(self.eval_val)

    def __repr__(self):
        return self.__to_str__(0)