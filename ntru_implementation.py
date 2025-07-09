# ntru_implementation.py

import numpy as np
import random

from keyCreator import poly_mul_ring

# --- Helper Functions ---

def poly_add(poly1, poly2, modulus):
    """
    Adds two polynomials coefficient-wise, taking results modulo `modulus`.
    Polynomials are represented as numpy arrays.
    """
    # Ensure polynomials have the same length for addition
    max_len = max(len(poly1), len(poly2))
    res = np.zeros(max_len, dtype=int)
    for i in range(max_len):
        coeff1 = poly1[i] if i < len(poly1) else 0
        coeff2 = poly2[i] if i < len(poly2) else 0
        res[i] = (coeff1 + coeff2) % modulus
    return res

def poly_mul(poly1, poly2, N, modulus):
    """
    Performs cyclic convolution (polynomial multiplication modulo X^N - 1).
    Coefficients are taken modulo `modulus`.
    """
    result_coeffs = np.zeros(N, dtype=int)
    for i in range(N):
        for j in range(N):
            # Calculate the index for the result polynomial (k = i + j mod N)
            k = (i + j) % N
            # Add the product of coefficients to the result at index k
            result_coeffs[k] = (result_coeffs[k] + poly1[i] * poly2[j]) % modulus
    return result_coeffs

def poly_mod_coeffs(poly, modulus):
    """
    Reduces all coefficients of a polynomial modulo `modulus` to be in [0, modulus-1).
    """
    return np.array([coeff % modulus for coeff in poly])

def poly_mod_centered(poly, modulus):
    """
    Reduces all coefficients of a polynomial modulo `modulus` to be in [-modulus/2, modulus/2).
    """
    centered_poly = np.zeros_like(poly, dtype=int)
    for i, coeff in enumerate(poly):
        reduced_coeff = coeff % modulus
        if reduced_coeff > modulus / 2:
            centered_poly[i] = reduced_coeff - modulus
        else:
            centered_poly[i] = reduced_coeff
    return centered_poly




def generate_random_poly(N, num_ones, num_minus_ones):
    """
    Generates a polynomial of degree < N with `num_ones` coefficients of 1,
    `num_minus_ones` coefficients of -1, and the rest 0.
    """
    poly = np.zeros(N, dtype=int)
    indices = list(range(N))
    random.shuffle(indices)

    # Place 1s
    for i in range(num_ones):
        poly[indices.pop()] = 1
    # Place -1s
    for i in range(num_minus_ones):
        poly[indices.pop()] = -1
    return poly


def ntru_encrypt(m, r, h, N, p, q):
    """
    Encrypts a message polynomial `m` using the public key `h`.
    `r` is the blinding polynomial.
    Returns the ciphertext polynomial `e`.
    e = p*r*h + m (mod q)
    """
    # Calculate p*r
    p_times_r = np.array([(coeff * p) for coeff in r])

    # Calculate (p*r)*h mod q
    pr_h = poly_mul_ring(p_times_r, h, N, q)

    # Calculate e = pr_h + m mod q
    e = poly_add(pr_h, m, q)
    return e

def ntru_decrypt(e, f, fp_inv, N, p, q):
    """
    Decrypts a ciphertext polynomial `e` using the private key `f` and `fp_inv`.
    Returns the original message polynomial `m`.
    """
    # Step 1: Calculate a = f * e (mod q)
    a = poly_mul_ring(f, e, N, q)

    # Step 2: Center coefficients of 'a' around 0 (lift from mod q to integers)
    a_prime = poly_mod_centered(a, q)

    # Step 3: Calculate b = a_prime * f_p_inv (mod p)
    b = poly_mul_ring(a_prime, fp_inv, N, p)

    # Step 4: Center coefficients of 'b' around 0 (lift from mod p to integers)
    # This step is implicitly done by the message structure (m has small coeffs)
    # and the properties of NTRU. The result 'b' should be 'm'.
    decrypted_message = poly_mod_centered(b, p) # To ensure coeffs are -1, 0, 1

    return decrypted_message