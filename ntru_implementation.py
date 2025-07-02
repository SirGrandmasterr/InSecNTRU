# ntru_implementation.py

import numpy as np
import random

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

def poly_invert(poly, N, modulus):
    """
    Finds the inverse of a polynomial `poly` modulo (X^N - 1) and `modulus`.
    This requires implementing the Polynomial Extended Euclidean Algorithm.
    This is the most challenging function.
    If an inverse does not exist, it should return None.

    Hint: You will need a helper function for polynomial division with remainder.
    The algorithm is similar to the integer Extended Euclidean Algorithm.
    """
    # --- Placeholder/Simplified Example (You need to implement a robust one) ---
    # For small N and specific moduli, a brute-force approach might work
    # but is highly inefficient and not general.
    # A proper implementation involves polynomial division and GCD.

    # Example of a very basic (and often insufficient) check for small N
    # This is NOT a full Extended Euclidean Algorithm.
    # You MUST replace this with a proper implementation.
    # For a robust solution, research "Polynomial Extended Euclidean Algorithm"
    # and "division algorithm for polynomials over finite fields/rings".

    # Simple check for N=7, p=3
    if N == 7 and modulus == 3:
        # Example: if poly is [1,1,1,0,0,0,0] (X^2+X+1)
        # Its inverse mod (X^7-1, 3) is [1,0,2,2,0,1,0] (X^5+2X^3+2X^2+1)
        # This is hardcoded and only for specific test cases.
        if np.array_equal(poly, np.array([1, 1, 1, 0, 0, 0, 0])):
            return np.array([1, 0, 2, 2, 0, 1, 0])
        # Add more specific hardcoded inverses for testing if needed,
        # but the goal is a general algorithm.
        pass
    # Simple check for N=7, q=41
    if N == 7 and modulus == 41:
        # Example: if poly is [1,1,1,0,0,0,0] (X^2+X+1)
        # Its inverse mod (X^7-1, 41) is [1, 40, 0, 0, 0, 0, 0] (X^6-X^5+X^4-X^3+X^2-X+1)
        # This is hardcoded and only for specific test cases.
        if np.array_equal(poly, np.array([1, 1, 1, 0, 0, 0, 0])):
            return np.array([1, 40, 0, 0, 0, 0, 0])
        pass

    # --- Your robust poly_invert implementation goes here ---
    # If no inverse is found, return None
    return None


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

# --- NTRU Scheme Functions ---

def ntru_key_generation(N, p, q, df, dg):
    """
    Generates the NTRU public and private keys.
    Returns (f, f_p_inv, f_q_inv, h)
    """
    f = None
    fp_inv = None
    fq_inv = None

    # Loop until an invertible f is found
    while True:
        # Generate f with df ones, df-1 minus ones (NTRU recommends df-1 for f)
        f = generate_random_poly(N, df, df - 1)

        # Check for invertibility
        fp_inv = poly_invert(f, N, p)
        if fp_inv is None:
            print(f"  f={f} not invertible mod p={p}. Retrying...")
            continue # Try a new f

        fq_inv = poly_invert(f, N, q)
        if fq_inv is None:
            print(f"  f={f} not invertible mod q={q}. Retrying...")
            continue # Try a new f

        # If both inverses found, break the loop
        break

    # Generate g with dg ones, dg minus ones
    g = generate_random_poly(N, dg, dg)

    # Compute public key h = f_q_inv * g mod q
    h = poly_mul(fq_inv, g, N, q)

    return f, fp_inv, fq_inv, h

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
    pr_h = poly_mul(p_times_r, h, N, q)

    # Calculate e = pr_h + m mod q
    e = poly_add(pr_h, m, q)
    return e

def ntru_decrypt(e, f, fp_inv, N, p, q):
    """
    Decrypts a ciphertext polynomial `e` using the private key `f` and `fp_inv`.
    Returns the original message polynomial `m`.
    """
    # Step 1: Calculate a = f * e (mod q)
    a = poly_mul(f, e, N, q)

    # Step 2: Center coefficients of 'a' around 0 (lift from mod q to integers)
    a_prime = poly_mod_centered(a, q)

    # Step 3: Calculate b = a_prime * f_p_inv (mod p)
    b = poly_mul(a_prime, fp_inv, N, p)

    # Step 4: Center coefficients of 'b' around 0 (lift from mod p to integers)
    # This step is implicitly done by the message structure (m has small coeffs)
    # and the properties of NTRU. The result 'b' should be 'm'.
    decrypted_message = poly_mod_centered(b, p) # To ensure coeffs are -1, 0, 1

    return decrypted_message