import numpy as np
import torch
from collections import defaultdict
from graphviz import Digraph
from torch.autograd import Function, Variable


def num_params(params):
    total_params, trainable = 0, 0
    for param in params:
        count = np.prod(param.data.shape)
        total_params += count
        if param.requires_grad:
            trainable += count
    return total_params, trainable


def rec_dd():
    return defaultdict(rec_dd)


def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)


def encode_datetime_cyclical(dt):
    hour_sin = np.sin(2 * np.pi * dt.hour / 23.0)
    hour_cos = np.cos(2 * np.pi * dt.hour / 23.0)
    month_sin = np.sin((dt.month - 1) * (2.0 * np.pi / 12))
    month_cos = np.cos((dt.month - 1) * (2.0 * np.pi / 12))
    return (month_sin, month_cos), (hour_sin, hour_cos)

def decode_datetime_cyclical(sin, cos, div):
    dec_sin = div * (np.arcsin(sin) / (2 * np.pi))
    dec_cos = div * (np.arccos(cos) / (2 * np.pi))
    return int(dec_cos) if dec_sin >= 0 else 23 - int(dec_cos)

def register_hooks(var):
    fn_dict = {}

    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input

        fn.register_hook(register_grad)

    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if grad_output is None:
            return False
        grad_output = grad_output.data
        return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()

    def make_dot():
        node_attr = dict(
            style="filled",
            shape="box",
            align="left",
            fontsize="12",
            ranksep="0.1",
            height="0.2",
        )
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return "(" + (", ").join(map(str, size)) + ")"

        def build_graph(fn):
            if hasattr(fn, "variable"):  # if GradAccumulator
                u = fn.variable
                node_name = "Variable\n " + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor="lightblue")
            else:
                assert fn in fn_dict, fn
                fillcolor = "white"
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = "red"
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, "variable", next_fn))
                    dot.edge(str(next_id), str(id(fn)))

        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot
