import numpy as np
import time
import ctypes
from keyCreator import key_gen

from ntru_implementation import ntru_encrypt, ntru_decrypt, generate_random_poly

rdtsc = ctypes.CDLL('./librdtsc.so')
rdtsc.read_cycles.restype = ctypes.c_uint64

# NTRU Parameters
N = 761
p = 3
q = 128
ITERATIONS = 10

def benchmark_keygen(N, p, q, iterations):
    """
    Benchmark the average number of CPU cycles taken for NTRU key generation.

    Args:
        N (int): Degree of the polynomial ring.
        p (int): Small modulus used for message space.
        q (int): Large modulus used for ciphertext space.
        iterations (int): Number of repetitions for averaging.

    Returns:
        int: Average number of CPU cycles for key generation.
    """
    total_cycles = 0
    for _ in range(iterations):
        start = rdtsc.read_cycles()
        h, f, f_p = key_gen(N, p, q)
        end = rdtsc.read_cycles()
        total_cycles += (end - start)
    return total_cycles // iterations

def benchmark_encryption(h, N, p, q, iterations):
    """
    Benchmark the average number of CPU cycles taken for NTRU encryption.

    Args:
        h (np.ndarray): Public key polynomial.
        N (int): Degree of the polynomial ring.
        p (int): Small modulus used for message space.
        q (int): Large modulus used for ciphertext space.
        iterations (int): Number of repetitions for averaging.

    Returns:
        int: Average number of CPU cycles for encryption.
    """
    total_cycles = 0
    for _ in range(iterations):
        m = generate_random_poly(N, num_ones=10, num_minus_ones=10)
        r = generate_random_poly(N, num_ones=10, num_minus_ones=10)
        start = rdtsc.read_cycles()
        _ = ntru_encrypt(m, r, h, N, p, q)
        end = rdtsc.read_cycles()
        total_cycles += (end - start)
    return total_cycles // iterations

def benchmark_decryption(f, f_p, h, N, p, q, iterations):
    """
    Benchmark the average number of CPU cycles taken for NTRU decryption.

    Args:
        f (np.ndarray): Private key polynomial.
        f_p (np.ndarray): Inverse of private key modulo p.
        h (np.ndarray): Public key polynomial.
        N (int): Degree of the polynomial ring.
        p (int): Small modulus used for message space.
        q (int): Large modulus used for ciphertext space.
        iterations (int): Number of repetitions for averaging.

    Returns:
        int: Average number of CPU cycles for decryption.
    """
    total_cycles = 0
    for _ in range(iterations):
        m = generate_random_poly(N, num_ones=10, num_minus_ones=10)
        r = generate_random_poly(N, num_ones=10, num_minus_ones=10)
        e = ntru_encrypt(m, r, h, N, p, q)
        start = rdtsc.read_cycles()
        decrypted_m = ntru_decrypt(e, f, f_p, N, p, q)
        end = rdtsc.read_cycles()
        total_cycles += (end - start)
        # sanity check
        if not np.array_equal(m, decrypted_m):
            print("[!] Decryption failed: message mismatch.")
    return total_cycles // iterations

# https://opensslntru.cr.yp.to/
def compare_with_opensslntru(build_keygen, build_enc, build_dec):
    """
    Compare benchmark results against official OpenSSLNTRU (libsntrup761) performance.

    Args:
        build_keygen (int): Your average keygen cycle count.
        build_enc (int): Your average encryption cycle count.
        build_dec (int): Your average decryption cycle count.

    Prints:
        Human-readable comparison against OpenSSLNTRU's performance reference values.
    """
    print("\n--- Comparison to OpenSSLNTRU libsntrup761 (Haswell cycles) ---")

    openssl_ref = {
        'Key Generation': 156_317,
        'Encryption':     46_914,
        'Decryption':     56_241
    }

    your_results = {
        'Key Generation': build_keygen,
        'Encryption':     build_enc,
        'Decryption':     build_dec
    }

    for op in ['Key Generation', 'Encryption', 'Decryption']:
        build_c = your_results[op]
        ref_c = openssl_ref[op]
        factor = build_c / ref_c
        print(f"{op:15}: {build_c:>8} cycles (ref: {ref_c})  => {factor:.1f}× slower")

# --- Run Benchmarks ---
print("Running NTRU benchmark with:")
print(f"  N = {N}, p = {p}, q = {q}, iterations = {ITERATIONS}\n")

# Generate keys once for use in encryption/decryption benchmarks
h, f, f_p = key_gen(N, p, q)

keygen_avg = benchmark_keygen(N, p, q, ITERATIONS)
encryption_avg = benchmark_encryption(h, N, p, q, ITERATIONS)
decryption_avg = benchmark_decryption(f, f_p, h, N, p, q, ITERATIONS)

# --- Results ---
compare_with_opensslntru(keygen_avg, encryption_avg, decryption_avg)
