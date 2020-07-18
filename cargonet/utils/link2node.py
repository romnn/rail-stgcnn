import networkx as nx


def link2node(g, mapping=None):
    """
    Convert graph links to nodes
    """
    mapping = mapping or dict()
    n = len(mapping)
    result = nx.Graph()
    # Create a node for each link
    for u, v, data in g.edges(data=True):
        try:
            result.add_node(mapping[(u, v)], **data)
        except KeyError:
            result.add_node(n, **data)
            mapping[(u, v)] = n
            mapping[(v, u)] = n
            n += 1
    # Create edges for all nodes that shared a vertex in the source graph
    for orig, data in g.nodes(data=True):
        for nb in g.neighbors(orig):
            for inb in g.neighbors(orig):
                if nb != inb:
                    e1, e2 = (orig, nb), (orig, inb)
                    n1, n2 = mapping[e1], mapping[e2]
                    result.add_edge(n1, n2, **data)

    result = result.to_undirected()
    return result, mapping


def node2link(g):
    """
    Convert graph nodes to links
    """
    mapping = dict()
    result = nx.Graph()
    # Create a link for each node
    n = 0
    for u, v, data in g.edges(data=True):
        result.add_node(n, **data)
        mapping[(u, v)] = n
        mapping[(v, u)] = n
        n += 1
    # Create edges for all nodes that shared a vertex in the source graph
    for orig in g.nodes:
        for nb in g.neighbors(orig):
            for inb in g.neighbors(orig):
                if nb != inb:
                    e1, e2 = (orig, nb), (orig, inb)
                    n1, n2 = mapping[e1], mapping[e2]
                    result.add_edge(n1, n2)

    result = result.to_undirected()
    return result, mapping
