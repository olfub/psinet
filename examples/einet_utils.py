import torch

from src import EinsumNetwork, Graph
from src.nns import MLP

def init_spn(device, obs_shape, int_shape, args):
    """
    Build a SPN (implemented as an einsum network). The structure is either
    the same as proposed in https://arxiv.org/pdf/1202.3732.pdf (referred to as
    poon-domingos) or a binary tree.

    In case of poon-domingos the image is split into smaller hypercubes (i.e. a set of
    neighbored pixels) where each pixel is a random variable. These hypercubes are split further
    until we operate on pixel-level. The spplitting is done randomly. For more information
    refer to the link above.
    """
    depth = args.depth
    while True:
        try:
            graph = Graph.random_binary_trees(
                num_var=obs_shape, depth=depth, num_repetitions=args.num_repetitions
            )
            break
        except:
            depth -= 1

    exponential_family = EinsumNetwork.NormalArray
    exponential_family_args = {"min_var": args.min_var, "max_var": args.max_var}

    einsum_args = EinsumNetwork.Args(
        num_var=obs_shape,
        num_dims=args.num_dims,
        num_classes=1,
        num_sums=args.K,
        num_input_distributions=args.K,
        exponential_family=exponential_family,
        exponential_family_args=exponential_family_args,
        use_em=False,
    )

    einet = EinsumNetwork.EinsumNetwork(graph, args=einsum_args)

    in_dims = int_shape
    first_layer_shape = einet.einet_layers[0].ef_array.params_shape[:-1]
    first_layer_shape = (*first_layer_shape, 1)
    # first_layer_shape twice because of mean and variance
    out_dims = (
        [first_layer_shape]
        + [first_layer_shape]
        + [
            einet.einet_layers[i].params_shape
            for i in range(1, len(einet.einet_layers))
        ]
    )

    print(f"Using NN={args.nn}")
    if args.nn == "cnn":
        pass  # TODO net = CNN(out_dims).to(device)
    elif args.nn == "mlp":
        hidden_dims = [int(hdim) for hdim in args.hidden_dims.split(",")]
        net = MLP(in_dims, out_dims, hidden_dims).to(device)
        for l in net.linear_layers:
            torch.nn.init.xavier_normal_(l.weight)
        for h in net.heads:
            torch.nn.init.xavier_normal_(h.weight)
    einet.param_nn = net
    einet.initialize()
    einet.to(device)
    return einet