import unittest
import numpy as np
import random
import sys
import os

# It's good practice to set up the path correctly to import local modules.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from keyCreator import key_gen, poly_mul_ring
    from ntru_implementation import (
        poly_mod_centered,
        generate_random_poly,
        ntru_encrypt,
        ntru_decrypt,
        ntru_encrypt_string,
        ntru_decrypt_string,
        _bytes_to_poly,
        _poly_to_bytes
    )
except ImportError as e:
    print(f"Error: Could not import from local files: {e}")
    print("Please ensure 'ntru_implementation.py' and 'keyCreator.py' are in the same directory.")
    sys.exit(1)

class TestNTRU(unittest.TestCase):

    def setUp(self):
        """Set up common parameters and keys for tests."""
        self.N = 509
        self.p = 7
        self.q = 2048
        self.dr = 15
        self.dm = 15
        
        
        # Generate keys once for all tests in this class to save time
        try:
            self.h, self.f, self.fp_inv = key_gen(self.N, self.p, self.q, max_tries=500)
        except RuntimeError as e:
            self.fail(f"Key generation failed during setup: {e}. This may be a random failure; try running tests again.")

    def assertPolyEqual(self, poly1, poly2, msg=None):
        """Custom assertion for comparing two polynomials."""
        max_len = max(len(poly1), len(poly2))
        p1_padded = np.pad(poly1, (0, max_len - len(poly1)), 'constant')
        p2_padded = np.pad(poly2, (0, max_len - len(poly2)), 'constant')
        self.assertTrue(np.array_equal(p1_padded, p2_padded), msg=f"Polynomials are not equal. {msg if msg else ''}")

    def test_poly_mul(self):
        """Test polynomial multiplication (cyclic convolution)."""
        print("\n--- Running polynomial multiplication test ---")
        N_test, mod_test = 7, 100
        poly1, poly2 = np.array([1, 1] + [0]*(N_test-2)), np.array([-1, 1] + [0]*(N_test-2))
        expected = np.array([-1, 0, 1] + [0]*(N_test-3))
        result = poly_mul_ring(poly1, poly2, N_test, mod_test)
        result_centered = poly_mod_centered(result, mod_test)
        self.assertPolyEqual(result_centered, expected, "poly_mul (X+1)(X-1) failed")
        print("\n---         PASSED         ---")

    def test_ntru_poly_cycle(self):
        """End-to-end test of NTRU for a single polynomial."""
        print("\n--- Running single polynomial NTRU cycle test ---")
        message_poly = generate_random_poly(self.N, self.dm, self.dm)
        blinding_poly = generate_random_poly(self.N, self.dr, self.dr)
        ciphertext = ntru_encrypt(message_poly, blinding_poly, self.h, self.N, self.p, self.q)
        decrypted_message = ntru_decrypt(ciphertext, self.f, self.fp_inv, self.N, self.p, self.q)
        self.assertPolyEqual(message_poly, decrypted_message, "Decryption failed for single polynomial cycle.")
        print("\n---         PASSED         ---")
        
    def test_byte_poly_conversion(self):
        """Test the conversion between byte chunks and polynomials."""
        print("\n--- Running byte-to-poly-to-byte conversion test ---")
        test_bytes = b"Hello, this is a test!"
        block_size = self.N // 6
        chunk = test_bytes[:block_size]
        
        poly = _bytes_to_poly(chunk, self.N)
        recovered_bytes = _poly_to_bytes(poly)
        
        # The recovered bytes should match the original chunk
        self.assertEqual(chunk, recovered_bytes[:len(chunk)])
        print("\n---         PASSED         ---")

    def test_string_encryption_decryption(self):
        """End-to-end test for encrypting and decrypting strings."""
        print("\n--- Running string encryption/decryption tests ---")
        
        # Test Case 1: Short message (fits in one block)
        short_message = "NTRU is a lattice-based cryptosystem."
        print(f"Testing short message: '{short_message}'")
        
        ciphertexts_short = ntru_encrypt_string(short_message, self.h, self.N, self.p, self.q, self.dr)
        decrypted_short = ntru_decrypt_string(ciphertexts_short, self.f, self.fp_inv, self.N, self.p, self.q)
        
        self.assertEqual(short_message, decrypted_short, "Decryption failed for short string.")
        print("\n---         PASSED         ---")
        
        # Test Case 2: Long message (requires multiple blocks)
        long_message = ("NTRU (Nth-degree Truncated polynomial Ring Units) is a lattice-based "
                        "public-key cryptosystem. It relies on the presumed difficulty of "
                        "factoring polynomials in a certain ring and finding short vectors "
                        "in a lattice. This implementation demonstrates the core concepts.")
        print(f"\nTesting long message (length {len(long_message)})...")
        
        ciphertexts_long = ntru_encrypt_string(long_message, self.h, self.N, self.p, self.q, self.dr)
        decrypted_long = ntru_decrypt_string(ciphertexts_long, self.f, self.fp_inv, self.N, self.p, self.q)
        
        self.assertEqual(long_message, decrypted_long, "Decryption failed for long string.")
        print("\n---         PASSED         ---")
        
        # Test Case 3: Edge case message (length is a multiple of block size)
        block_size = self.N // 6
        edge_message = "A" * block_size
        print(f"\nTesting edge case message (length {len(edge_message)})...")

        ciphertexts_edge = ntru_encrypt_string(edge_message, self.h, self.N, self.p, self.q, self.dr)
        decrypted_edge = ntru_decrypt_string(ciphertexts_edge, self.f, self.fp_inv, self.N, self.p, self.q)

        self.assertEqual(edge_message, decrypted_edge, "Decryption failed for edge case string.")
        print("\n---         PASSED         ---")


if __name__ == '__main__':
    print(f"\n--- Running tests with N={509}, p={7}, q={2048} ---")
    unittest.main(argv=['first-arg-is-ignored'],verbosity=2, exit=False)
