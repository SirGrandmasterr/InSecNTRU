import numpy as np
import random
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

    # ensure both polynomials are of length N-1
    #if len(poly1) != N-1 or len(poly2) != N-1:
    #    raise ValueError("Both polynomials must be of length N.")
    full_mul = np.polymul(poly1, poly2)
    print("FULL MULL", len(full_mul))
    result = np.zeros(N, dtype=int)
    # multiply polynomials with cyclic convolution
    for i in range(len(full_mul)):
        new_power = i % N
        result[new_power] = (result[new_power] + full_mul[i])
        
    # Finally, reduce the coefficients modulo 'mod'
    result = result % mod
        
    return result

def random_poly(N : int) -> np.ndarray:
    # get set number of ones, negative ones and zeros (normal for NTRU)
    # TODO: make this a parameter to give the user security level control
    # TODO: ones and neg ones dont have to be equal. Include random offset.
    # TODO: maybe also make the exact amount random rather than N // 3. Read up if this makes sense.
    ones = random.randrange(0, N-1, 1)
    neg_ones = random.randrange(0, N-1-ones, 1)
    
    num_zeros = N-1 - ones - neg_ones

    poly = np.concatenate((np.ones(ones), -np.ones(neg_ones), np.zeros(num_zeros)))
    #print("Len of poly: ", len(poly))
    #print("Amount Ones", ones)
    #print("Amount neg_ones", neg_ones)
    #print("Amount null", num_zeros)
    np.random.shuffle(poly)  # shuffle array to get randomized polynomial
    #print("poly shuffled", poly)
    return poly.astype(int)

def poly_inverse_ring(poly : np.ndarray, N : int, mod : int) -> np.ndarray | None:
    

    x = symbols('x')
    coeffs = poly[::-1].tolist() 
    p_int = Poly(coeffs, x, domain='ZZ')
    R_int = Poly(x**N - 1, x, domain='ZZ') # Ring
    if is_prime(mod):
        try:
            p = p_int.set_domain(GF(mod))
            R = R_int.set_domain(GF(mod))
            inv_poly = invert(p, R)
        except :
            # No inverse exists
            return None
    elif is_power_of_two(mod):
        try:
            p_mod2 = p_int.set_domain(GF(2))
            R_mod2 = R_int.set_domain(GF(2))
            inv_poly_mod2 = p_mod2.invert(R_mod2)
        except:
            # No inverse exists
            return None
        inv_poly = Poly(inv_poly_mod2.all_coeffs(), x, domain='ZZ')
        e = int(np.log2(mod))
        for _ in range(1, e):
            # The Hensel's Lemma lifting step. All polynomials (p_int, inv_poly, R_int)
            # are now in the same integer domain 'ZZ', so operations are valid.
            inv_poly = (2 * inv_poly - p_int * inv_poly ** 2) % R_int

        # 4. Reduce the final integer coefficients by the target modulus.
        inv_poly = inv_poly.trunc(mod)
    else:
        raise ValueError("The modulus must be a prime or a power of two.")

    # Convert back to a numpy array of a fixed size N
    inv_coeffs = inv_poly.all_coeffs()[::-1]
    inv_coeffs_padded = inv_coeffs + [0] * (N - len(inv_coeffs))
    return np.array(inv_coeffs_padded, dtype=int)

def key_gen(N : int, p : int, q : int, max_tries=100) -> tuple:
    for _ in range(max_tries):
        f = random_poly(N)
        f_p = poly_inverse_ring(f, N, p)
        f_q = poly_inverse_ring(f, N, q)
        

        if f_q is not None and f_p is not None:
            g = random_poly(N)
            print(f_q)
            h = poly_mul_ring(f_q, g, N, q)

            return h, f, f_p
        
    raise ValueError("Maximum tries exceeded while generating private key polynomials.")

if __name__ == "__main__":
    # TODO: function to generate sane values
    #N=251 q=128 p=3
    N = 251 
    p = 3
    q = 128

    h, f, f_p = key_gen(N, p, q)

    print("Public Key (h):", h)
    print("Private Key (f):", f)
    print("Private Key Inverse (f_p):", f_p)