import numpy as np

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
    # generate rdm polynomial with gradient N-1 and coefficients in [-1, 0, 1]
    poly = np.zeros(N, dtype=int)
    for i in range(N):
        poly[i] = np.random.choice([-1, 0, 1])
    return poly

def poly_inverse_ring(poly : np.ndarray, N : int, mod : int) -> np.ndarray:
    # get the inverse of polynomial if it exists but stay in the ring
    return np.zeros(N, dtype=int)  # Placeholder for actual implementation

def key_gen(N : int, p : int, q : int, max_tries=100) -> tuple:
    for _ in range(max_tries):
        f = gen_rdm_ntru_poly(N)
        #f_p = poly_inverse_ring(f, N, p) # can be included for faster decryption
        f_q = poly_inverse_ring(f, N, q)
        if f_q is not None: #and f_p is not None:
            g = gen_rdm_ntru_poly(N)
            h = poly_mul_ring(f_q, g, N, q)

            return h, f # pubkey, privkey
        
    raise ValueError("Maximum tries exceeded while generating private key polynomials.")