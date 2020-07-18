import torch


def nx_to_tg(g, node_features=None, edge_features=None, convert_edges=True):
    node_features = node_features or []
    edge_features = edge_features or []

    n_nodes = g.number_of_nodes()
    nodes = torch.zeros(n_nodes, len(node_features), dtype=torch.float)
    for n, data in g.nodes(data=True):
        for j, feature in enumerate(node_features):
            nodes[i][j] = data[feature]
    n_edges = g.number_of_edges()
    edges = torch.zeros(n_edges, 2, dtype=torch.long)
    edge_attrs = torch.zeros(n_edges, len(edge_features), dtype=torch.long)
    if convert_edges:
        for i, edge in enumerate(g.edges):
            u, v = edge
            edges[i][0], edges[i][1] = u, v
        for j, feature in enumerate(edge_features):
            edge_attrs[i][j] = g.edges[edge][feature]
        if n_edges > 0:
            edges = edges.t()
            edges = to_undirected(edges)
    return Data(x=nodes, edge_attr=edge_attrs, edge_index=edges.contiguous())
