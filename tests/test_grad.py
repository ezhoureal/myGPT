import pytest

from manual_grad.grad import Value

def test_value_addition():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c.data == 5.0, f"Expected 5.0, got {c.data}"

def test_value_multiplication():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c.data == 6.0, f"Expected 6.0, got {c.data}"

def test_value_power():
    a = Value(2.0)
    b = Value(3.0)
    c = a ** b
    assert c.data == 8.0, f"Expected 8.0, got {c.data}"

def test_value_relu():
    a = Value(-2.0)
    b = a.relu()
    assert b.data == 0.0, f"Expected 0.0, got {b.data}"

    c = Value(3.0)
    d = c.relu()
    assert d.data == 3.0, f"Expected 3.0, got {d.data}"

def test_value_backward_addition():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()
    assert a._grad == 1.0, f"Expected 1.0, got {a._grad}"
    assert b._grad == 1.0, f"Expected 1.0, got {b._grad}"

def test_value_backward_multiplication():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    assert a._grad == 3.0, f"Expected 3.0, got {a._grad}"
    assert b._grad == 2.0, f"Expected 2.0, got {b._grad}"

def test_value_backward_power():
    a = Value(2.0)
    b = Value(3.0)
    c = a ** b
    c.backward()
    assert pytest.approx(a._grad, rel=1e-5) == 12.0, f"Expected approx 12.0, got {a._grad}"

def test_value_zero_grad():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    a.zero_grad()
    b.zero_grad()
    assert a._grad == 0.0, f"Expected 0.0, got {a._grad}"
    assert b._grad == 0.0, f"Expected 0.0, got {b._grad}"

import torch
def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg._grad == xpt.grad.item()

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg._grad - apt.grad.item()) < tol
    assert abs(bmg._grad - bpt.grad.item()) < tol

def test_value_tanh():
    a = Value(0.0)
    b = a.tanh()
    assert b.data == 0.0, f"Expected 0.0, got {b.data}"

    a = Value(1.0)
    b = a.tanh()
    assert pytest.approx(b.data, rel=1e-5) == 0.76159, f"Expected approx 0.76159, got {b.data}"

    a = Value(-1.0)
    b = a.tanh()
    assert pytest.approx(b.data, rel=1e-5) == -0.76159, f"Expected approx -0.76159, got {b.data}"

def test_value_backward_tanh():
    a = Value(1.0)
    b = a.tanh()
    b.backward()
    expected_grad = 1 - b.data**2
    assert pytest.approx(a._grad, rel=1e-5) == expected_grad, f"Expected approx {expected_grad}, got {a._grad}"

    a = Value(-1.0)
    b = a.tanh()
    b.backward()
    expected_grad = 1 - b.data**2
    assert pytest.approx(a._grad, rel=1e-5) == expected_grad, f"Expected approx {expected_grad}, got {a._grad}"