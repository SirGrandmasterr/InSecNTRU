
import numpy as np
import random
import math

from keyCreator import poly_mul_ring

def poly_add(poly1, poly2, modulus):
    """
    Adds two polynomials coefficient-wise, taking results modulo `modulus`.
    Polynomials are represented as numpy arrays.
    """
    max_len = max(len(poly1), len(poly2))
    p1_padded = np.pad(poly1, (0, max_len - len(poly1)), 'constant')
    p2_padded = np.pad(poly2, (0, max_len - len(poly2)), 'constant')
    return (p1_padded + p2_padded) % modulus

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
    half_mod = modulus / 2
    for i, coeff in enumerate(poly):
        reduced_coeff = coeff % modulus
        if reduced_coeff >= half_mod:
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

    for _ in range(num_ones):
        poly[indices.pop()] = 1
    for _ in range(num_minus_ones):
        poly[indices.pop()] = -1
    return poly

def ntru_encrypt(m, r, h, N, p, q):
    """
    Encrypts a message polynomial `m` using the public key `h`.
    e = p*r*h + m (mod q)
    """
    p_times_r = np.array([(coeff * p) for coeff in r])
    pr_h = poly_mul_ring(p_times_r, h, N, q)
    e = poly_add(pr_h, m, q)
    return e

def ntru_decrypt(e, f, fp_inv, N, p, q):
    """
    Decrypts a ciphertext polynomial `e` using the private key `f` and `fp_inv`.
    """
    a = poly_mul_ring(f, e, N, q)
    a_prime = poly_mod_centered(a, q)
    b = poly_mul_ring(a_prime, fp_inv, N, p)
    decrypted_message = poly_mod_centered(b, p)
    return decrypted_message

def _bytes_to_poly(chunk: bytes, N: int) -> np.ndarray:
    """
    Converts a chunk of bytes into a message polynomial with coefficients {-1, 0, 1}.
    We use a balanced ternary representation. Each byte (0-255) is converted to 6
    ternary digits, which become 6 coefficients in the polynomial.
    """
    if len(chunk) * 6 > N:
        raise ValueError("Chunk too fat for poly size N")
    
    poly = np.zeros(N, dtype=int)
    poly_idx = 0
    for byte in chunk:
        num = int(byte)
        for _ in range(6):
            rem = num % 3
            num //= 3
            if rem == 2:
                poly[poly_idx] = -1
                num += 1
            else:
                poly[poly_idx] = rem
            poly_idx += 1
    return poly

def _poly_to_bytes(poly: np.ndarray) -> bytes:
    """
    Converts a message polynomial back into a chunk of bytes.
    """
    byte_list = []
    num_bytes = len(poly) // 6
    for i in range(num_bytes):
        num = 0
        # Read the 6 coefficients for this byte, from most significant to least.
        # The coefficients are already in {-1, 0, 1}, so we can use them directly
        # in the standard base-conversion formula.
        for j in range(5, -1, -1):
            coeff = poly[i*6 + j]
            num = num * 3 + coeff
        byte_list.append(num)

    return bytes(byte_list)

def ntru_encrypt_string(message: str, h, N: int, p: int, q: int, dr: int) -> list:
    """
    Encrypts a string of arbitrary length.
    """
    message_bytes = message.encode('utf-8')
    block_size = N // 6
    
    # Pad the message to be a multiple of the block size using PKCS#7
    padding_len = block_size - (len(message_bytes) % block_size)
    padded_message = message_bytes + bytes([padding_len] * padding_len)
    
    ciphertexts = []
    for i in range(0, len(padded_message), block_size):
        chunk = padded_message[i:i+block_size]
        m_poly = _bytes_to_poly(chunk, N)
        r_poly = generate_random_poly(N, dr, dr)
        e_poly = ntru_encrypt(m_poly, r_poly, h, N, p, q)
        ciphertexts.append(e_poly)
        
    return ciphertexts

def ntru_decrypt_string(ciphertexts: list, f, fp_inv, N: int, p: int, q: int) -> str:
    """
    Decrypts a list of ciphertext polynomials back into a string.
    """
    decrypted_bytes = b''
    for e_poly in ciphertexts:
        m_poly = ntru_decrypt(e_poly, f, fp_inv, N, p, q)
        chunk = _poly_to_bytes(m_poly)
        decrypted_bytes += chunk
        
    # Remove PKCS#7 padding
    # This can fail if decrypted_bytes is empty or too short.
    if not decrypted_bytes:
        return "" # Return empty string if decryption produced no bytes.
        
    padding_len = decrypted_bytes[-1]
    
    # Basic sanity check on padding length
    if padding_len > len(decrypted_bytes):
        # This indicates a decryption failure.
        return "" # Or raise an error
        
    unpadded_bytes = decrypted_bytes[:-padding_len]
    
    return unpadded_bytes.decode('utf-8', errors='ignore')
