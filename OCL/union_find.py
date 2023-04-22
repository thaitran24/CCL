class UnionFindArray():
    def __init__(self):
        self.P = []
        self.label = 0

    def make_label(self):
        """
        Create new label when when there is no neighbor object.
        """
        root = self.label
        self.label += 1
        self.P.append(root)
        return root
    
    def find_root(self, i):
        """
        Find root value of node `i`.
        """
        root = i
        while self.P[root] < root:
            root = self.P[root]
        return root
    
    def set_root(self, i, root):
        """
        Make all nodes on the path from node `i` to its root to point directly to `root`.
        This reduces the height of the tree, hence make `find_root` faster. 
        """
        while self.P[i] < i:
            j = self.P[i]
            self.P[i] = root
            i = j
        self.P[i] = root
    
    def union(self, i, j):
        """
        Combining two trees containing node `i` and `j`.
        This method always select the root with the smaller label 
        as the root of the combined tree.
        """
        if i != j:
            root = self.find_root(i)
            rootj = self.find_root(j)
            if root > rootj:
                root = rootj
            self.set_root(j, root)
            self.set_root(i, root)

    def find(self, i):
        """
        Find the root node of the tree containing node `i`
        """
        root = self.find_root(i)
        self.set_root(i, root)
        return root

    def flatten(self):
        """
        Flatten the tree, guarantee that each node in a tree points directly to its 
        root for better finding label equivalence.
        """
        for i in range(1, len(self.P)):
            self.P[i] = self.P[self.P[i]]

    def flattenL(self):
        k = 1
        for i in range(1, len(self.P)):
            if self.P[i] < i:
                self.P[i] = self.P[self.P[i]]
            else:
                self.P[i] = k
                k += 1