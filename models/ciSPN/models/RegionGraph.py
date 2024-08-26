import numpy as np


class RegionGraph:
    def __init__(
        self,
        region,
        num_permutations=1,
        num_splits=2,
        max_depth=2,
        rng_seed=12345,
        verbose=True,
    ):
        assert num_splits == 2, "Hardcoded. Assumed for several functions ..."

        self.verbose = verbose

        if type(region) is int:
            region = list(range(region))

        self.childs = []
        self.region = region.copy()  # sorted(region)
        self.is_leaf = False

        self._rng = np.random.default_rng(rng_seed)
        self._permutations = []

        if len(region) <= 1:
            # FIXME we can, but need special handling (and it doesn't really make sense)
            raise ValueError("Can't build a tree from zero [TODO or one] variables")

        for _ in range(num_permutations):
            # permutating here and splitting afterwards is equal to randomly selecting
            # variables from the ordered regions.
            region_perm = self._get_new_permutation(region)
            if self.verbose:
                print(f"Region: {region_perm}")
            self.childs.append(
                RegionGraphNode(region_perm, num_splits=num_splits, max_depth=max_depth)
            )

    def _get_new_permutation(self, region):
        # TODO we could be a bit smarter here ...
        retries = 10
        for _ in range(retries):
            region_perm = list(self._rng.permutation(region))
            if region_perm not in self._permutations:
                break
        self._permutations.append(region_perm.copy())
        return region_perm


class RegionGraphNode:
    def __init__(self, region, num_splits, max_depth):
        assert num_splits == 2, "Hardcoded. Also assumed for several functions ..."

        self.childs = []
        self.region = region.copy()  # sorted(region)
        self.is_leaf = False

        if max_depth == 0:
            self.is_leaf = True
        else:
            if len(region) == 0:
                raise ValueError("Region is empty")
            elif len(region) == 1:
                self.is_leaf = True
            else:
                # assumes num_splits == 2
                split = len(region) // 2
                # print(region, "->", region[:split], region[split:])
                self.childs.append(
                    RegionGraphNode(
                        region[:split], num_splits=num_splits, max_depth=max_depth - 1
                    )
                )
                self.childs.append(
                    RegionGraphNode(
                        region[split:], num_splits=num_splits, max_depth=max_depth - 1
                    )
                )

    def __repr__(self):
        return f"{{region {self.region}}}"
