import numpy as np
import random
from sympy import Poly, invert, GF, symbols

def is_prime(n: int) -> bool:
    """
    Checks if a number is prime.
    """
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def is_power_of_two(n: int) -> bool:
    """
    Checks if a number is a power of two.
    """
    return (n > 0) and (n & (n - 1)) == 0

def poly_mul_ring(poly1: np.ndarray, poly2: np.ndarray, N: int, mod: int) -> np.ndarray:
    """
    Multiplies two polynomials in the ring R = Z_mod[x] / (x^N - 1).
    This is a corrected implementation that avoids the pitfalls of the previous version.
    
    Args:
        poly1: First polynomial as a numpy array of coefficients [c0, c1, ...].
        poly2: Second polynomial as a numpy array of coefficients [c0, c1, ...].
        N: The degree of the ring polynomial (x^N - 1).
        mod: The modulus for the coefficients.

    """
    len1 = len(poly1)
    len2 = len(poly2)
    
    full_mul_len = len1 + len2 - 1
    
    full_mul = np.zeros(full_mul_len, dtype=np.int64)

    # Standard polynomial multiplication.
    for i in range(len1):
        if poly1[i] == 0: continue  # Small optimization
        for j in range(len2):
            full_mul[i + j] += poly1[i] * poly2[j]

    # Reduce modulo x^N - 1 (i.e., x^k = x^(k mod N)). This is cyclic convolution.
    result = np.zeros(N, dtype=np.int64)
    for i in range(full_mul_len):
        result[i % N] += full_mul[i]

    # Reduce the coefficients modulo `mod`.
    # This produces results in the range [0, mod-1]. 
    return (result % mod).astype(int)


def random_poly(N: int) -> np.ndarray:
    """
    Generates a random ternary polynomial of length N, suitable for use as
    the private key polynomial 'f'.
    """
    d = N // 3 
    ones = d
    neg_ones = d - 1 # This ensures the sum of coeffs is 1.

    # Ensure we don't have more coefficients than the polynomial length.
    if ones + neg_ones > N:
        raise ValueError("Number of ones and negative ones exceeds polynomial degree N.")

    num_zeros = N - ones - neg_ones

    poly = np.concatenate((np.ones(ones), -np.ones(neg_ones), np.zeros(num_zeros)))
    np.random.shuffle(poly)  # Shuffle to randomize coefficient positions
    return poly.astype(int)

def poly_inverse_ring(poly: np.ndarray, N: int, mod: int) -> np.ndarray | None:
    """
    Finds the inverse of a polynomial in the ring R = Z_mod[x] / (x^N - 1).
    """
    x = symbols('x')
    # SymPy's Poly expects coefficients from highest power to lowest.
    # Our convention is lowest to highest, so we reverse the list.
    coeffs = poly[::-1].tolist()
    p_int = Poly(coeffs, x, domain='ZZ')
    R_int = Poly(x**N - 1, x, domain='ZZ')

    if is_prime(mod):
        try:
            # Invert in the finite field GF(mod)
            p = p_int.set_domain(GF(mod))
            R = R_int.set_domain(GF(mod))
            inv_poly = invert(p, R)
        except Exception:
            # No inverse exists
            return None
    elif is_power_of_two(mod):
        # Hensel's Lemma to "lift" the inverse.
        try:
            # 1. Find inverse modulo 2
            p_mod2 = p_int.set_domain(GF(2))
            R_mod2 = R_int.set_domain(GF(2))
            inv_poly_mod2 = p_mod2.invert(R_mod2)
        except Exception:
            # If it's not invertible mod 2, it's not invertible mod 2^k
            return None
        
        # 2. Lift the solution from mod 2 up to mod q
        inv_poly = Poly(inv_poly_mod2.all_coeffs(), x, domain='ZZ')
        
        e = int(np.log2(mod))
        current_mod = 2
        
        for _ in range(1, e):
            # At the start of the loop, inv_poly is the inverse mod current_mod
            current_mod *= 2
            
            # The Hensel's Lemma lifting step: v_new = v * (2 - f * v)
            term = (p_int * inv_poly).rem(R_int)
            inv_poly = (inv_poly * (2 - term)).rem(R_int)
            inv_poly = inv_poly.trunc(current_mod)
    else:
        raise ValueError("The modulus must be a prime or a power of two.")

    # Convert back to a numpy array of a fixed size N, reversing to our convention.
    inv_coeffs = inv_poly.all_coeffs()[::-1]
    inv_coeffs_padded = inv_coeffs + [0] * (N - len(inv_coeffs))
    return np.array(inv_coeffs_padded, dtype=int)


def key_gen(N: int, p: int, q: int, max_tries=100) -> tuple:
    """
    Generates NTRU public and private keys (h, f, f_p).
    """
    for i in range(max_tries):
        f = random_poly(N)
        # Try to find the inverse of f modulo p and q
        f_p = poly_inverse_ring(f, N, p)
        f_q = poly_inverse_ring(f, N, q)

        # We need both inverses to exist to proceed
        if f_q is not None and f_p is not None:
            # Generate another random polynomial g
            g = random_poly(N)
            # Calculate the public key h = p * f_q * g (mod q)
            h = poly_mul_ring(f_q, g, N, q)
            return h, f, f_p
        
    # If we fail after many tries, raise an error.
    raise RuntimeError(f"Maximum tries ({max_tries}) exceeded while generating private key. Could not find an invertible f.")

if __name__ == "__main__":
   
    N = 251 
    p = 3
    q = 128

    try:
        h, f, f_p = key_gen(N, p, q)
        print("Key generation successful.")
        print("Public Key (h):", h)
        print("Private Key (f):", f)
        print("Private Key Inverse (f_p):", f_p)
    except RuntimeError as e:
        print(e)
