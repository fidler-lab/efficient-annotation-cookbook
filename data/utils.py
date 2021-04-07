import os
import xmltodict
from . import REPO_DIR


class Node():
    def __init__(self, node, depth=0, parent=None):
        self._node = node
        self.depth = depth
        self.wnid = node['@wnid']
        self.gloss = node['@gloss']
        self.name = node['@words']
        self.parent = parent
        self._set_children_node()

    def _set_children_node(self):
        self.children = []
        if 'synset' in self._node:
            children = self._node['synset']
            if not isinstance(children, list):
                children = [children]
            for child in children:
                child = Node(child, depth=self.depth+1, parent=self)
                self.children.append(child)

        self.num_children = len(self.children)

    def is_leaf(self):
        return self.num_children == 0

    def __repr__(self):
        return self.name


class Tree():
    def __init__(self, root):
        self.root = root
        self.leaf_nodes = self.find_nodes(lambda node: node.is_leaf())
        self.max_depth = max([n.depth for n in self.leaf_nodes])

    def find_nodes(self, filter):
        nodes = []
        to_expand = [self.root]
        while len(to_expand) > 0:
            node = to_expand.pop()
            if filter(node):
                nodes.append(node)
            to_expand.extend(node.children)
        return nodes


class ImageNetStruc():
    structure_path = os.path.join(REPO_DIR, 'data/structure_released.xml')

    def __init__(self):
            
        # >>> ImageNet Structure
        with open(self.structure_path) as f:
            xml = f.read()
            struct = xmltodict.parse(xml)
        root = struct['ImageNetStructure']['synset']
        root_node = Node(root)
        tree = Tree(root_node)
        self.tree = tree

        # Info
        wnid_to_name = {}
        to_expand = [tree.root]
        while len(to_expand) > 0:
            node = to_expand.pop(0)
            wnid_to_name.update({node.wnid: node.name})
            to_expand.extend(node.children)

        self.wnid_to_name = wnid_to_name

    def register_nodes(self, class_wnid):
        nodes_of_interest = []
        for wnid in class_wnid:
            nodes = self.tree.find_nodes(lambda node: node.wnid == wnid)
            # Each wnid might correspond to multiple nodes in the tree
            nodes_of_interest.append(nodes[0])
        self.nodes_of_interest = nodes_of_interest
