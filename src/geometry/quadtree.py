def _interleave_bits(x: int, y: int) -> int:
    """Interleaves bits of x and y to produce a Morton (Z-order) code."""
    res = 0
    for i in range(16):
        res |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
    return res

def compute_morton_code(x: int, y: int, depth: int) -> int:
    """
    Computes depth-d Morton code for cell coordinates (x, y).
    """
    return _interleave_bits(x, y)

def p_adic_valuation_4(morton1: int, morton2: int) -> int:
    """
    Returns the 4-adic valuation v4(morton1 - morton2), which equals 
    the length of the common Z-order prefix.
    Implemented as leading zero pairs of the XOR difference.
    """
    diff = morton1 ^ morton2
    if diff == 0:
        return 16 # Max depth for 32-bit interleaved
        
    # Find position of highest set bit
    # Every 2 bits represents one depth level in base 4
    depth = 0
    while diff > 0:
        diff >>= 2
        depth += 1
    return 16 - depth

def p_adic_distance_4(morton1: int, morton2: int) -> float:
    """
    d4(A, A') = 4^{-v4(A - A')}
    """
    val = p_adic_valuation_4(morton1, morton2)
    return 4.0 ** (-val)
