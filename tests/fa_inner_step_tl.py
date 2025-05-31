import step_tl
from sympy import Symbol, Min, Max, Basic
import torch
from functools import reduce
from collections import namedtuple
from dataclasses import dataclass
from typing import Union, Dict
import json

torch.manual_seed(42)
E = Symbol("E")
M = Symbol("M")
N = Symbol("N")
K = Symbol("K")
D = Symbol("D")
M_value = 5
N_value = 7
K_value = 9
D_value = 16
ctx = {
    M: M_value,
    N: N_value,
    K: K_value,
    D: D_value
}

input_dtype = {
    'E0': step_tl.Tile("float", [D, M]),
    'E1': step_tl.Tile("float", [D, N]),
    'E2': step_tl.Tile("float", [D, N])
}
input_data = {
    'E0': torch.randn(1, M_value, D_value),
    'E1': torch.randn(1, K_value, N_value, D_value),
    'E2': torch.randn(1, K_value, N_value, D_value),
}

class Matmul(step_tl.Fn):
    def __init__(self, input, output):
        super().__init__("Matmul", input, output)
    
    def apply(self, input):
        return [torch.einsum('ld,sd->ls', (input[0], input[1]))]

fn_matmul = Matmul(step_tl.STuple((step_tl.Tile("float", [D, M]), step_tl.Tile("float", [D, N]))), step_tl.Tile("float", [N, M]))

class ExpMaxDiff(step_tl.Fn):
    def __init__(self, input, output):
        super().__init__("ExpMaxDiff", input, output)

    def getInit(self):
        return [torch.full((M_value,), float('-inf')), torch.zeros(M_value, N_value), torch.zeros(M_value)]

    def apply(self, state, input):
        m_t = state[0] # [M], [N, M], [M]
        s_t = input[0] # [N, M]
        m_next = torch.maximum(torch.max(s_t, dim=1, keepdim=False).values, m_t)
        d_next = torch.exp(m_t - m_next)
        e_next = torch.exp(s_t - m_next.unsqueeze(-1))
        return [m_next, e_next, d_next]
    
fn_expmaxdiff = ExpMaxDiff(step_tl.Tile("float", [N, M]), step_tl.STuple((step_tl.Vector("float", [M]), step_tl.Tile("float", [N, M]), step_tl.Vector("float", [M]))))

class GetLastTwo(step_tl.Fn):
    def __init__(self, input, output):
        super().__init__("GetLastTwo", input, output)
    
    def apply(self, input):
        return [input[1], input[2]]
    
fn_getlasttwo = GetLastTwo(step_tl.STuple((step_tl.Vector("float", [M]), step_tl.Tile("float", [N, M]), step_tl.Vector("float", [M]))), step_tl.STuple((step_tl.Tile("float", [N, M]), step_tl.Vector("float", [M]))))

class WeightedSum(step_tl.Fn):
    def __init__(self, input, output):
        super().__init__("WeightedSum", input, output)

    def getInit(self):
        return [torch.zeros(M_value), torch.zeros(M_value, D_value)]
    
    def apply(self, state, input):
        v_t, e_t, d_t = input # [D, N], [N, M], [M]
        l_t, o_t = state # [M], [D, M]
        l_next = l_t * d_t + e_t.sum(dim=-1, keepdim=False)
        o_next = o_t * d_t.unsqueeze(-1) + e_t @ v_t
        return [l_next, o_next]
    
fn_weightedsum = WeightedSum(step_tl.STuple((step_tl.Tile("float", [D, N]), step_tl.Tile("float", [N, M]), step_tl.Vector("float", [M]))), step_tl.STuple((step_tl.Vector("float", [M]), step_tl.Tile("float", [D, M]))))

class Div(step_tl.Fn):
    def __init__(self, input, output):
        super().__init__("Div", input, output)
    
    def apply(self, input):
        l_t, o_t = input # [M], [D, M]
        return [o_t / l_t.unsqueeze(-1)]
    
fn_div = Div(step_tl.STuple((step_tl.Vector("float", [M]), step_tl.Tile("float", [D, M]))), step_tl.Tile("float", [D, M]))

def prepare():
    E0 = step_tl.Stream("E0", step_tl.Tile("float", [D, M]), 0, [1])
    E0.ctx = ctx
    E0.data = [input_data['E0']]

    E1 = step_tl.Stream("E1", step_tl.Tile("float", [D, N]), 1, [K, 1])
    E1.ctx = ctx
    E1.data = [input_data['E1']]

    E2 = step_tl.Stream("E2", step_tl.Tile("float", [D, N]), 1, [K, 1])
    E2.ctx = ctx
    E2.data = [input_data['E2']]
    return E0, E1, E2

def check_shape(S0):
    output_dtype_S0 = step_tl.Tile("float", [D, M])
    assert S0.dtype == step_tl.Tile("float", [D, M]), f"The output dtype should be {output_dtype_S0.dtype} but got {S0.dtype}"
    assert S0.shape == [1], f"The output shape should be [1] but got {S0.shape}"

def check_data(S0):
    q = input_data['E0'].squeeze(0) # [M, D]
    k = input_data['E1'].squeeze(0).view(-1, ctx[D]) # [K * N, D]
    v = input_data['E2'].squeeze(0).view(-1, ctx[D]) # [K * N, D]
    s = q @ k.transpose(0, 1) # [M, K * N]
    p = torch.softmax(s, dim=-1) # [M, K * N]
    o = p @ v # [M, D]
    S0_data_0 = o.unsqueeze(0)
    torch.testing.assert_close(S0.data[0], S0_data_0)

# def check_data(S0):
#     q = input_data['E0'].squeeze(0)
#     k = input_data['E1'].squeeze(0)
#     v = input_data['E2'].squeeze(0)
#     o = torch.zeros((ctx[M], ctx[D]), dtype=torch.float32)
#     l = torch.zeros((ctx[M]), dtype=torch.float32)
#     m = torch.full((ctx[M],), float('-inf'), dtype=torch.float32)
#     for idx in range(ctx[K]):
#       k_tile = k[idx, :]
#       v_tile = v[idx, :]
#       s = torch.einsum('ld,sd->ls', (q, k_tile))
#       m_next = torch.maximum(torch.max(s, dim=1, keepdim=False).values, m)
#       delta = torch.exp(m - m_next)
#       p = torch.exp(s - m_next.unsqueeze(-1))
#       l_next = l * delta + p.sum(dim=-1, keepdim=False)
#       o_next = o * delta.unsqueeze(-1) + p @ v_tile
#       o = o_next
#       l = l_next
#       m = m_next
#     S0_data_0 = o / l.unsqueeze(-1)
#     torch.testing.assert_close(S0.data[0], S0_data_0.unsqueeze(0))

def test():
    E0, E1, E2 = prepare()
    S0 = body(E0, E1, E2)
    check_shape(S0)
    check_data(S0)

def body(E0, E1, E2):
    E3 = step_tl.RepeatRef().apply((E0, E1))
    E4 = step_tl.Zip().apply((E3, E1))
    E5 = step_tl.Map(fn=fn_matmul).apply(E4)
    E6 = step_tl.Scan(fn=fn_expmaxdiff, b=1).apply(E5)
    E7 = step_tl.Map(fn=fn_getlasttwo).apply(E6)
    E8 = step_tl.Zip().apply((E2, E7))
    E9 = step_tl.Accum(fn=fn_weightedsum, b=1).apply(E8)
    E10 = step_tl.Map(fn=fn_div).apply(E9)
    return E10