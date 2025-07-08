import numpy as np
from sympy import Poly, invert, GF, symbols

def is_prime(n: int) -> bool:
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def is_power_of_two(n: int) -> bool:
    return (n != 0) and (n & (n - 1)) == 0

def poly_mul_ring(poly1 : np.ndarray, poly2 : np.ndarray, N : int, mod : int) -> np.ndarray:
    # approach without using numpy's convolve

    # ensure both polynomials are of length N
    if len(poly1) != N or len(poly2) != N:
        raise ValueError("Both polynomials must be of length N.")

    result = np.zeros(N, dtype=int)
    # multiply polynomials with cyclic convolution
    for i in range(N):
        for j in range(N):
            k = (i + j) % N
            result[k] = (result[k] + poly1[i] * poly2[j]) % mod

    return result

def gen_rdm_ntru_poly(N : int) -> np.ndarray:
    # get set number of ones, negative ones and zeros (normal for NTRU)
    # TODO: make this a parameter to give the user security level control
    # TODO: ones and neg ones dont have to be equal. Include random offset.
    # TODO: maybe also make the exact amount random rather than N // 3. Read up if this makes sense.
    num_ones = N // 3
    num_neg_ones = num_ones
    num_zeros = N - num_ones - num_neg_ones

    poly = np.concatenate((np.ones(num_ones), -np.ones(num_neg_ones), np.zeros(num_zeros)))
    np.random.shuffle(poly)  # shuffle array to get randomized polynomial

    return poly.astype(int)

def poly_inverse_ring(poly : np.ndarray, N : int, mod : int) -> np.ndarray | None:
    # TODO: implement this without using sympy ? Will turn into a few lines tho\

    x = symbols('x')
    coeffs = poly[::-1].tolist() # converting to be compatible with sympy
    p = Poly(coeffs, x, domain=GF(mod))
    R = Poly(x**N - 1, x, domain=GF(mod)) # Ring
    if is_prime(mod):
        try:
            inv_poly = invert(p, R)
        except:
            # No inverse exists
            return None
    elif is_power_of_two(mod):
        try:
            inv_poly = invert(p, R, domain=GF(2))
        except:
            # No inverse exists
            return None
        e = int(np.log2(mod))
        for _ in range(1, e):
            inv_poly = ((2 * inv_poly - p * inv_poly ** 2) % R).trunc(mod)
    else:
        # TODO: This is because of NTRU specs i think?
        raise ValueError("The modulus must be a prime or a power of two.") 

    # convert back to numpy array
    inv_coeffs = inv_poly.all_coeffs()[::-1]
    inv_coeffs += [0] * (N - len(inv_coeffs)) # zero pad to length N
    return np.array(inv_coeffs, dtype=int)

def key_gen(N : int, p : int, q : int, max_tries=100) -> tuple:
    for _ in range(max_tries):
        f = gen_rdm_ntru_poly(N)
        f_p = poly_inverse_ring(f, N, p)
        f_q = poly_inverse_ring(f, N, q)

        if f_q is not None and f_p is not None:
            g = gen_rdm_ntru_poly(N)
            h = poly_mul_ring(f_q, g, N, q)

            return h, f, f_p
        
    raise ValueError("Maximum tries exceeded while generating private key polynomials.")

if __name__ == "__main__":
    # TODO: function to generate sane values
    
    N = 509
    p = 7
    q = 2048

    h, f, f_p = key_gen(N, p, q)

    print("Public Key (h):", h)
    print("Private Key (f):", f)
    print("Private Key Inverse (f_p):", f_p)