import math

import torch

from .RegionGraph import RegionGraph

NINF = -float("inf")


class SPNParamProvider_old:
    def __init__(self, batch_size, num_leaf_variables):
        self.batch_size = batch_size
        self.num_leaf_variables = num_leaf_variables

        self.reset()

        self.nn = None

    def set_nn(self, nn):
        self.nn = nn

    def estimate_parameters(self, y):
        sum_weights, leaf_weights = self.nn.forward(y)
        self.set_params(sum_weights, leaf_weights)

    def reset(self):
        self.sum_params_used = 0
        self.leaf_params_used = [0] * self.num_leaf_variables

        self.sum_params = None
        self.leaf_params = None

    def generate_sum_index(self, num_inputs, num_sums):
        # each sum get num_inputs: num_inputs*num_sums parameters
        # use access_sum_parameters to access data
        num_vars = num_inputs * num_sums

        idx_range_tuple = (
            self.sum_params_used,
            self.sum_params_used + num_vars,
            num_inputs,
            num_sums,
        )
        self.sum_params_used += num_vars

        return idx_range_tuple

    def generate_leaf_index(self, scope, num_parameters):
        # creates tuple of indices: (scope, start_idcs, end_idcs)
        # use access_leaf_parameters to access data

        # for performance reasons (torch has no gather_nd function ...)
        # we repeated collect slices
        start_idcs = [self.leaf_params_used[var] for var in scope]
        end_idcs = [start_idx + num_parameters for start_idx in start_idcs]

        for var in scope:
            self.leaf_params_used[var] += num_parameters

        return scope, start_idcs, end_idcs

    def set_params(self, sum_params, leaf_params):
        self.sum_params = sum_params
        self.leaf_params = leaf_params

    def access_sum_parameters(self, index):
        idx_low, idx_high, num_inputs, num_sums = index

        params = self.sum_params[:, idx_low:idx_high]
        return torch.reshape(params, [-1, num_inputs, num_sums])

    def access_leaf_parameters(self, index):
        # NOTE if it crashes here the number of passed leaf weights for the SPN was probably too low.

        scope = index[0]
        start_idcs = index[1]
        end_idcs = index[2]
        if len(scope) == 1:
            params_new = self.leaf_params[
                :, scope[0], start_idcs[0] : end_idcs[0]
            ].unsqueeze(1)
        else:
            scope_params = []
            for var, start_idx, end_idx in zip(scope, start_idcs, end_idcs):
                scope_params.append(
                    self.leaf_params[:, var, start_idx:end_idx].unsqueeze(1)
                )
            params_new = torch.cat(scope_params, 1)

        return params_new

    def num_params(self):
        return sum(self.leaf_params_used), self.sum_params_used

    def num_params_from_sum_index(self, idx):
        return idx[1] - idx[0]

    def num_params_from_leaf_index(self, idx):
        return sum([end - start for start, end in zip(idx[1], idx[2])])


class SPNFlatParamProvider:
    def __init__(self):
        self.reset()

        self.nn = None

    def set_nn(self, nn):
        self.nn = nn

    def estimate_parameters(self, x):
        sum_weights, leaf_weights = self.nn.forward(x)
        self.set_params(sum_weights, leaf_weights)

    def reset(self):
        self.sum_params_used = 0
        self.leaf_params_used = 0

        self.sum_params = None
        self.leaf_params = None

    def generate_sum_index(self, num_inputs, num_sums):
        # each sum get num_inputs: num_inputs*num_sums parameters
        # use access_sum_parameters to access data
        num_vars = num_inputs * num_sums

        idx_range_tuple = (
            self.sum_params_used,
            self.sum_params_used + num_vars,
            num_inputs,
            num_sums,
        )
        self.sum_params_used += num_vars

        return idx_range_tuple

    def generate_leaf_index(self, scope, num_parameters):
        assert (
            len(scope) == 1
        )  # we need to do some reshaping in the access function otherwise
        num_vars = len(scope) * num_parameters
        idx_range_tuple = (
            self.leaf_params_used,
            self.leaf_params_used + num_vars,
            len(scope),
            num_parameters,
        )
        self.leaf_params_used += num_vars

        return idx_range_tuple

    def set_params(self, sum_params, leaf_params):
        self.sum_params = sum_params
        self.leaf_params = leaf_params

    def access_sum_parameters(self, index):
        idx_low, idx_high, num_inputs, num_sums = index

        params = self.sum_params[:, idx_low:idx_high]
        return torch.reshape(params, [-1, num_inputs, num_sums])

    def access_leaf_parameters(self, index):
        idx_low, idx_high, num_inputs, num_params = index

        params = self.leaf_params[:, idx_low:idx_high]
        return torch.reshape(params, [-1, num_inputs, num_params])

    def num_params(self):
        return self.leaf_params_used, self.sum_params_used

    def num_params_from_sum_index(self, idx):
        return idx[1] - idx[0]

    def num_params_from_leaf_index(self, idx):
        return idx[1] - idx[0]


class NodeParameterization:
    def __init__(
        self,
        param_provider: SPNFlatParamProvider,
        num_total_variables,
        num_gauss=4,
        num_sums=4,
    ):
        self.param_provider = param_provider

        self.process_config = CiSPNProcessConfig()

        # used for sampling
        self.num_total_variables = num_total_variables

        self.num_gauss = num_gauss
        self.num_sums = num_sums

        self.gauss_min_var = 0.1
        self.gauss_max_var = 2


class CiSPNNode:
    def __init__(self, region, is_leaf, childs=None):
        if childs is None:
            childs = []

        self.region = region
        self.is_leaf = is_leaf
        self.childs = childs

        self.size = -1
        self.num_outputs = -1

    def traverse(self, fn):
        fn(self)
        for child in self.childs:
            child.traverse(fn)


class CiSPNSumNode(CiSPNNode):
    def __init__(
        self, region, childs, num_sums, node_parameterization: NodeParameterization
    ):
        super().__init__(region, is_leaf=False, childs=childs)

        self.process_config = node_parameterization.process_config
        self._max_mask = None
        self.param_provider = node_parameterization.param_provider

        num_inputs = sum([child.num_outputs for child in childs])
        self._param_idxs = self.param_provider.generate_sum_index(num_inputs, num_sums)

        self.node_child_map = []  # maps which argmax belongs to which child
        self.node_child_offset = (
            []
        )  # maps which offset is needed to map own argmax to child output
        offset = 0
        for idx, child in enumerate(self.childs):
            self.node_child_map.extend([idx] * child.num_outputs)
            self.node_child_offset.extend([offset] * child.num_outputs)
            offset += child.num_outputs

        self.num_outputs = num_sums

    def forward(self, x):
        # sum nodes compute weighted sums out of all inputs
        inputs = torch.cat([child.forward(x) for child in self.childs], dim=1)

        params = self.param_provider.access_sum_parameters(self._param_idxs)

        # normalize weights
        weights = torch.log_softmax(params, 1)

        # multiplication in log space is addition:
        child_values = torch.unsqueeze(inputs, -1) + weights

        if self.process_config.record_argmax:
            self._max_mask = torch.argmax(child_values, dim=1)

        if self.process_config.apply_mask:
            # apply point-wise mask
            log_zero = (
                torch.ones_like(child_values) * NINF
            )  # zero out parameters in log space

            # build a mask task contains ones at max value positions and zero everywhere else
            oh_mask = torch.nn.functional.one_hot(
                self._max_mask, num_classes=child_values.shape[1]
            ).transpose(1, 2)

            child_values = torch.where(oh_mask.bool(), child_values, log_zero)

        sums = torch.logsumexp(child_values, dim=1)
        # print("sum", sums.shape)
        return sums
    
    def bounded_integral(self, y_lower, y_upper):
        # sum nodes compute weighted sums out of all inputs
        inputs = torch.cat([child.bounded_integral(y_lower, y_upper) for child in self.childs], dim=1)

        params = self.param_provider.access_sum_parameters(self._param_idxs)

        # normalize weights
        weights = torch.log_softmax(params, 1)

        # multiplication in log space is addition:
        child_values = torch.unsqueeze(inputs, -1) + weights

        if self.process_config.record_argmax:
            self._max_mask = torch.argmax(child_values, dim=1)

        if self.process_config.apply_mask:
            # apply point-wise mask
            log_zero = (
                torch.ones_like(child_values) * NINF
            )  # zero out parameters in log space

            # build a mask task contains ones at max value positions and zero everywhere else
            oh_mask = torch.nn.functional.one_hot(
                self._max_mask, num_classes=child_values.shape[1]
            ).transpose(1, 2)

            child_values = torch.where(oh_mask.bool(), child_values, log_zero)

        sums = torch.logsumexp(child_values, dim=1)
        # print("sum", sums.shape)
        return sums

    def num_params(self):
        return self.param_provider.num_params_from_sum_index(self._param_idxs)

    def reconstruct(self, node_num, case_num, recons):
        my_max_idx = self._max_mask[case_num, node_num]
        # find the vector that output the max_idx
        # for inp_vector in self.inputs:
        #    if my_max_idx < inp_vector.size:
        #        return inp_vector.reconstruct(my_max_idx, case_num, sample)
        #    my_max_idx -= inp_vector.size
        child_idx = self.node_child_map[my_max_idx]
        child_idx_offset = self.node_child_offset[my_max_idx]
        # reconstruction = self.childs[child_idx].reconstruct(my_max_idx - child_idx_offset, case_num, recons)
        # return reconstruction

        # just calling reconstruct. The result is stored in recons
        self.childs[child_idx].reconstruct(
            my_max_idx - child_idx_offset, case_num, recons
        )


class CiSPNProductNode(CiSPNNode):
    def __init__(self, region, childs, node_parameterization: NodeParameterization):
        assert len(childs) == 2, "Hardcoded. Assumed for forward/infer functions."
        super().__init__(region, is_leaf=False, childs=childs)

        self.num_outputs = math.prod([child.num_outputs for child in childs])

    def forward(self, x):
        inputs = [child.forward(x) for child in self.childs]

        # FIXME assumes 2 split of region. HARD CODED!
        dists1 = inputs[0]
        dists2 = inputs[1]

        # we compute a product for every possible combination of inputs:
        # we take outer products, thus expand in different dims
        # TODO for multiple splits this should be a n-dim outer product (this is the point
        #  that is hardcoded and currently does not allow more than two splits ...)
        dists1_expand = torch.unsqueeze(dists1, 1)
        dists2_expand = torch.unsqueeze(dists2, 2)

        # product == sum in log-domain
        prods = dists1_expand + dists2_expand
        # flatten out the outer product
        prods = torch.reshape(prods, [dists1.shape[0], -1])
        # print("prod", prods.shape)
        return prods
    
    def bounded_integral(self, x_lower, x_upper):
        inputs = [child.bounded_integral(x_lower, x_upper) for child in self.childs]

        # FIXME assumes 2 split of region. HARD CODED!
        dists1 = inputs[0]
        dists2 = inputs[1]

        # we compute a product for every possible combination of inputs:
        # we take outer products, thus expand in different dims
        # TODO for multiple splits this should be a n-dim outer product (this is the point
        #  that is hardcoded and currently does not allow more than two splits ...)
        dists1_expand = torch.unsqueeze(dists1, 1)
        dists2_expand = torch.unsqueeze(dists2, 2)

        # product == sum in log-domain
        prods = dists1_expand + dists2_expand
        # flatten out the outer product
        prods = torch.reshape(prods, [dists1.shape[0], -1])
        # print("prod", prods.shape)
        return prods

    def num_params(self):
        return 0

    def reconstruct(self, node_num, case_num, recons):
        # reconstruct results from the n-th product, of the c-th sample
        # FIXME assumes 2 split of region. HARD CODED!
        row_num = node_num // self.childs[0].num_outputs
        col_num = node_num % self.childs[0].num_outputs
        # result1 = self.childs[0].reconstruct(col_num, case_num, recons)
        # result2 = self.childs[1].reconstruct(row_num, case_num, recons)
        # return result1 + result2

        # we don't need to add results since the leaf nodes already add up values in recons
        self.childs[0].reconstruct(col_num, case_num, recons)
        self.childs[1].reconstruct(row_num, case_num, recons)


class CiSPNGaussNode(CiSPNNode):
    def __init__(self, region, node_parameterization: NodeParameterization):
        super().__init__(region, is_leaf=True, childs=None)

        self.node_parameterization = node_parameterization
        self.process_config = node_parameterization.process_config
        self.num_gaussians = node_parameterization.num_gauss

        self.gauss_min_var = node_parameterization.gauss_min_var
        self.gauss_max_var = node_parameterization.gauss_max_var

        self.means_idx = self.node_parameterization.param_provider.generate_leaf_index(
            self.region, self.node_parameterization.num_gauss
        )

        if self.gauss_min_var < self.gauss_max_var:
            self.sigma_params_idx = (
                self.node_parameterization.param_provider.generate_leaf_index(
                    self.region, self.node_parameterization.num_gauss
                )
            )
        else:
            self.sigma_params_idx = None

        # parameters are set during forward pass
        self.dist = torch.distributions.Normal(0.0, 1.0)

        self.num_outputs = node_parameterization.num_gauss

    def forward(self, x):
        self.means = self.node_parameterization.param_provider.access_leaf_parameters(
            self.means_idx
        )

        if self.gauss_min_var < self.gauss_max_var:
            sigma_params = (
                self.node_parameterization.param_provider.access_leaf_parameters(
                    self.sigma_params_idx
                )
            )
            sigma = self.gauss_min_var + torch.sigmoid(sigma_params) * (
                self.gauss_max_var - self.gauss_min_var
            )
        else:
            sigma = 1.0

        self.dist.loc = self.means
        self.dist.scale = torch.sqrt(sigma)

        local_inputs = x[:, self.region]  # select scope from data
        local_inputs = torch.unsqueeze(local_inputs, -1)
        gauss_log_pdf_single = self.dist.log_prob(local_inputs)

        # marginalized[v] = 0 -> variable v weights are kept in the forward pass
        # marginalized[v] = 1 -> variable v weights are zeroed out in the forward pass
        if self.process_config.marginalized is not None:
            # marginalized = torch.clip(marginalized, 0.0, 1.0)
            local_marginalized = torch.unsqueeze(
                self.process_config.marginalized[:, self.region], -1
            )

            # setting a value zero in log space = multiplying by one = marginalizing it out
            gauss_log_pdf_single = gauss_log_pdf_single * (1 - local_marginalized)

        gauss_log_pdf = torch.sum(gauss_log_pdf_single, dim=1)
        # print("gauss", gauss_log_pdf.shape)
        return gauss_log_pdf
    
    def bounded_integral(self, x_lower, x_upper):
        self.means = self.node_parameterization.param_provider.access_leaf_parameters(
            self.means_idx
        )

        if self.gauss_min_var < self.gauss_max_var:
            sigma_params = (
                self.node_parameterization.param_provider.access_leaf_parameters(
                    self.sigma_params_idx
                )
            )
            sigma = self.gauss_min_var + torch.sigmoid(sigma_params) * (
                self.gauss_max_var - self.gauss_min_var
            )
        else:
            sigma = 1.0

        self.dist.loc = self.means
        self.dist.scale = torch.sqrt(sigma)

        # select scope from data
        if x_lower is not None:
            local_inputs_lower = x_lower[:, self.region]
            local_inputs_lower = torch.unsqueeze(local_inputs_lower, -1)
        if x_upper is not None:
            local_inputs_upper = x_upper[:, self.region]  
            local_inputs_upper = torch.unsqueeze(local_inputs_upper, -1)

        # TODO test that shapes are correct
        if x_lower is None and x_upper is None:
            cdfs = self.dist.loc
        elif x_lower is None:
            cdfs = self.dist.cdf(local_inputs_upper)
        elif x_upper is None:
            cdfs = 1 - self.dist.cdf(local_inputs_lower)
        else:
            cdfs = self.dist.cdf(local_inputs_upper) - self.dist.cdf(local_inputs_lower)
        gauss_log_pdf_single = torch.log(cdfs)

        # marginalized[v] = 0 -> variable v weights are kept in the forward pass
        # marginalized[v] = 1 -> variable v weights are zeroed out in the forward pass
        if self.process_config.marginalized is not None:
            # marginalized = torch.clip(marginalized, 0.0, 1.0)
            local_marginalized = torch.unsqueeze(
                self.process_config.marginalized[:, self.region], -1
            )

            # setting a value zero in log space = multiplying by one = marginalizing it out
            gauss_log_pdf_single = gauss_log_pdf_single * (1 - local_marginalized)

        gauss_log_pdf = torch.sum(gauss_log_pdf_single, dim=1)
        # print("gauss", gauss_log_pdf.shape)
        return gauss_log_pdf

    def num_params(self):
        sum = self.node_parameterization.param_provider.num_params_from_leaf_index(
            self.means_idx
        )
        if self.sigma_params_idx is not None:
            sum += self.node_parameterization.param_provider.num_params_from_leaf_index(
                self.sigma_params_idx
            )
        return sum

    def reconstruct(self, node_num, case_num, recons):
        my_sample = self.means[case_num, ...]  # means.shape = [1000, 1, 4]
        my_sample = my_sample[:, node_num]
        # full_sample = torch.zeros((self.node_parameterization.num_total_variables,), device=my_sample.device)
        # full_sample[self.region] = my_sample
        # return full_sample

        # we add our value since, there might be already another value (= doing 'inplace summation' instead of doing
        # it explicitly in the product node)
        recons[case_num, self.region] += my_sample


class CiSPNProcessConfig:
    def __init__(self):
        # None or [{0,1},...]
        # 0 -> Keep variable
        # 1 -> Marginalize / "Ignore" variable
        self.marginalized = None

        # records argmax in the forward pass
        self.record_argmax = False

        # applies a recorded argmax to the values in the forward pass
        self.apply_mask = False


class CiSPN(torch.nn.Module):
    def __init__(
        self, region_graph: RegionGraph, node_parameterization: NodeParameterization
    ):
        super().__init__()
        assert len(region_graph.region) == node_parameterization.num_total_variables

        self.nn = None

        self.node_parameterization = node_parameterization
        self.node_parameterization.param_provider.reset()
        self.process_config = node_parameterization.process_config

        # construct root node, and attach child trees
        childs = []
        for child_region in region_graph.childs:
            childs.append(self._from_graph(child_region, node_parameterization))
        self.root = CiSPNSumNode(
            region_graph.region,
            childs,
            num_sums=1,
            node_parameterization=node_parameterization,
        )

        # collect leaf nodes
        self.leafs = []

        def collect_leafs(node):
            if node.is_leaf:
                self.leafs.append(node)

        self.root.traverse(collect_leafs)

    def _from_graph(self, region_node, node_parameterization):
        region = region_node.region
        childs = []

        if region_node.is_leaf:
            childs.append(CiSPNGaussNode(region, node_parameterization))
        else:
            for child_region in region_node.childs:
                if child_region.is_leaf:
                    childs.append(
                        CiSPNGaussNode(child_region.region, node_parameterization)
                    )
                else:
                    childs.append(
                        CiSPNSumNode(
                            child_region.region,
                            [self._from_graph(child_region, node_parameterization)],
                            node_parameterization.num_sums,
                            node_parameterization,
                        )
                    )

        return CiSPNProductNode(region, childs, node_parameterization)

    def forward(self, x, y, marginalized=None):
        # x = condition data, y = var data
        self.process_config.marginalized = marginalized
        self.node_parameterization.param_provider.set_nn(self.nn)
        self.node_parameterization.param_provider.estimate_parameters(x)
        result = self.root.forward(y)
        self.node_parameterization.param_provider.set_nn(None)
        self.node_parameterization.param_provider.set_params(None, None)
        self.process_config.marginalized = None
        return result
    
    def forward_integral(self, x, y_lower, y_upper, marginalized=None):
        # x = condition data, y = var data
        self.process_config.marginalized = marginalized
        self.node_parameterization.param_provider.set_nn(self.nn)
        self.node_parameterization.param_provider.estimate_parameters(x)
        result = self.root.bounded_integral(y_lower, y_upper)
        self.node_parameterization.param_provider.set_nn(None)
        self.node_parameterization.param_provider.set_params(None, None)
        self.process_config.marginalized = None
        return result

    def predict(self, conditions, targets, marginalized=None):
        return self.reconstruct_batch(conditions, targets, marginalized)

    def reconstruct_batch(self, conditions, targets, marginalized=None):
        # this also updates max_child_idx for SumVectors
        # and makes sure leaf_vector.means is set
        self.process_config.record_argmax = True
        self.forward(conditions, targets, marginalized=marginalized)
        self.process_config.record_argmax = False

        batch_size = targets.shape[0]

        # FIXME write directly into torch tensor? We know the size ...
        recons = torch.clone(targets) * (1 - marginalized)
        # recons = [None]*batch_size
        for i in range(batch_size):
            # recons[i] = self.reconstruct(i, recons)
            self.reconstruct(i, recons)
        # recons = torch.stack(recons, dim=0)
        return recons

    def reconstruct(self, case_num, recons):
        return self.root.reconstruct(0, case_num, recons)

    def num_parameters(self):
        # FIXME deprecated use param_provider directly
        return self.node_parameterization.param_provider.num_params()

    def set_nn(self, nn):
        self.nn = nn

    def print_structure_info(self):
        counts = {
            "Product": 0,
            "Sum": 0,
            "Gaussian": 0,
        }

        def record_nodes(node):
            if isinstance(node, CiSPNProductNode):
                counts["Product"] += 1
            elif isinstance(node, CiSPNSumNode):
                counts["Sum"] += 1
            elif isinstance(node, CiSPNGaussNode):
                counts["Gaussian"] += 1
            else:
                raise ValueError("Unknown node instance")

        self.root.traverse(record_nodes)

        print("== SPN Statistics ==")
        print("Splits:", len(self.root.childs))
        print("Nodes:", counts)

        (
            num_leaf_params,
            num_sum_params,
        ) = self.node_parameterization.param_provider.num_params()
        print(f"{num_leaf_params} leaf parameters. {num_sum_params} sum parameters.")
        print("== Statistics End ==")
