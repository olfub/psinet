import graphviz
import torch


class DTNode:
    def __init__(self, name="", level=0):
        self._is_leaf = True
        self.name = name
        self.childs = []
        self.level = level

    def score(self):
        pass

    def create_child_node(self, name=""):
        child = DTNode(name, self.level + 1)
        self.childs.append(child)
        return child

    def print(self):
        print(
            ("  " * self.level)
            + f"[{self.name}|{self._class}|{self.decision_feature}|{self.scores}]"
        )
        for child in self.childs:
            child.print()

    def is_leaf(self):
        return len(self.childs) == 0

    def predict(self, sample):
        if self.is_leaf():
            return self._class

        feature_idx = self.decision_feature
        return self.childs[int(sample[feature_idx].item())].predict(sample)

    def _default_content_str(self):
        if self.scores is None:
            score_str = "None"
        else:
            score_str = [f"{score:.3f}" for score in self.scores]
        return f"{self.name}\n{self.decision_feature}\n{score_str}"

    def dot(
        self,
        dot,
        prefix="",
        name_delimiter="_",
        node_args=None,
        edge_args=None,
        content_func=None,
    ):
        if node_args is None:
            node_args = {}
        if edge_args is None:
            edge_args = {}
        node_name = prefix + self.name

        if content_func is None:
            content = self._default_content_str()
        else:
            content = content_func(self)
        dot.node(node_name, content, **node_args)

        child_names = [
            child.dot(
                dot,
                node_name + name_delimiter,
                name_delimiter,
                node_args,
                edge_args,
                content_func,
            )
            for child in self.childs
        ]
        for child_name in child_names:
            dot.edge(node_name, child_name, **edge_args)
        return node_name

    def clean(self):
        self.scores = None
        self.mask = None
        self.active_var_names = None

        for child in self.childs:
            child.clean()

    def to_torch(self, to_torch=True, cuda=True):
        if to_torch:
            if not isinstance(self._class, torch.Tensor) and self._class is not None:
                self._class = torch.tensor(self._class)
                if cuda:
                    self._class = self._class.cuda()
        else:
            if isinstance(self._class, torch.Tensor):
                self._class = self._class.detach().cpu().numpy()

        for child in self.childs:
            child.to_torch(to_torch, cuda)

    def prune(self):
        if self.is_leaf():
            return self._class

        is_mixed = False  # node contains more than one class
        c_classes = None
        for child in self.childs:
            c_class = child.prune()
            if not is_mixed:
                if c_class is None:
                    is_mixed = True
                else:
                    if c_classes is None:
                        c_classes = c_class
                    else:
                        if not torch.equal(c_class, c_classes):
                            is_mixed = True
        if is_mixed:
            return None
        else:
            self.childs.clear()
            # node.scores = None
            self.decision_feature = None
            self._class = c_classes

            return c_classes


class DecisionTree:
    def __init__(self, root_node_name=""):
        self._root = DTNode(root_node_name)

    def get_root(self):
        return self._root

    def print(self):
        self._root.print()

    def predict(self, sample):
        return self._root.predict(sample)

    def dot(self, graph_args, node_args, edge_args, content_func=None):
        dot = graphviz.Digraph(**graph_args)
        self.get_root().dot(
            dot, node_args=node_args, edge_args=edge_args, content_func=content_func
        )
        return dot

    def clean(self):
        """
        remove all statistics and helper vars from tree construction from the nodes
        """
        self.get_root().clean()

    def to_torch(self, to_torch=False, cuda=True):
        self.get_root().to_torch(to_torch, cuda)

    def prune(self):
        self.get_root().prune()

    def node_stats(self):
        return self._node_stats(self.get_root())

    def _node_stats(self, node):
        if node.is_leaf():
            return [0, 1]
        else:
            stats = [1, 0]
            for child in node.childs:
                n, l = self._node_stats(child)
                stats[0] += n
                stats[1] += l
            return stats
