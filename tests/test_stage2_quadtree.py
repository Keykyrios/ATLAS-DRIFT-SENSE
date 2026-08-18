import pytest
from src.geometry.quadtree import compute_morton_code, p_adic_valuation_4, p_adic_distance_4

def test_morton_prefix_ultrametric_ball_prop1():
    """
    Test Proposition 1: 4-adic valuation equals common Z-order prefix length,
    and d4 acts as an ultrametric bounding spatial distance.
    """
    # Two adjacent cells at same depth
    m1 = compute_morton_code(5, 5, 6)
    m2 = compute_morton_code(5, 4, 6)
    
    val = p_adic_valuation_4(m1, m2)
    # They should share a prefix up to the last few bits
    assert val > 0
    
    d4 = p_adic_distance_4(m1, m2)
    assert 0 < d4 < 1.0
    
    # Identical cells
    assert p_adic_valuation_4(m1, m1) == 16
    assert p_adic_distance_4(m1, m1) == 4.0 ** (-16)
    
    # Ultrametric inequality: d(x, z) <= max(d(x, y), d(y, z))
    m3 = compute_morton_code(1, 1, 6)
    d_xz = p_adic_distance_4(m1, m3)
    d_xy = p_adic_distance_4(m1, m2)
    d_yz = p_adic_distance_4(m2, m3)
    
    assert d_xz <= max(d_xy, d_yz)
