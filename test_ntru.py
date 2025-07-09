import unittest
import numpy as np
import random
import sys
import os

# It's good practice to set up the path correctly to import local modules.
# This ensures the script can be run from different directories.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from keyCreator import key_gen, poly_inverse_ring, poly_mul_ring
    from ntru_implementation import (
        poly_mod_centered,
        generate_random_poly,
        ntru_encrypt,
        ntru_decrypt
    )
except ImportError as e:
    print(f"Error: Could not import from local files: {e}")
    print("Please ensure 'ntru_implementation.py' and 'keyCreator.py' are in the same directory.")
    sys.exit(1)

class TestNTRU(unittest.TestCase):

    def setUp(self):
        """
        Set up common parameters for tests.
        These parameters are suitable for testing; not for production security.
        """
        self.N = 509  # A prime N
        self.p = 7    # Small prime modulus
        self.q = 2048 # Larger modulus (power of 2), coprime to p

        # Parameters for generating random polynomials (f, g, r, m)
        self.df = 60
        self.dg = 20
        self.dr = 15
        self.dm = 15
        
        print(f"\n--- Running tests with N={self.N}, p={self.p}, q={self.q} ---")

    def assertPolyEqual(self, poly1, poly2, msg=None):
        """Custom assertion for comparing two polynomials (numpy arrays)."""
        # Ensure polynomials are padded to the same length for comparison
        max_len = max(len(poly1), len(poly2))
        p1_padded = np.pad(poly1, (0, max_len - len(poly1)), 'constant')
        p2_padded = np.pad(poly2, (0, max_len - len(poly2)), 'constant')
        self.assertTrue(np.array_equal(p1_padded, p2_padded), msg=f"Polynomials are not equal: {p1_padded} vs {p2_padded}. {msg if msg else ''}")

    def test_poly_mul(self):
        """
        Test polynomial multiplication (cyclic convolution) modulo X^N-1 and a modulus.
        This test is now self-contained and uses appropriate parameters.
        """
        print("Testing poly_mul (cyclic convolution)...")
        N_test = 7
        mod_test = 100
        
        # Test case: (X+1) * (X-1) = X^2 - 1
        # In our coefficient convention [c0, c1, ...], X+1 is [1, 1, 0, ...]
        poly1 = np.array([1, 1, 0, 0, 0, 0, 0])
        # X-1 is [-1, 1, 0, ...]
        poly2 = np.array([-1, 1, 0, 0, 0, 0, 0])
        
        # The expected result is X^2 - 1, which is [-1, 0, 1, 0, ...]
        expected_centered = np.array([-1, 0, 1, 0, 0, 0, 0])
        
        # poly_mul_ring returns coefficients in [0, mod-1].
        result = poly_mul_ring(poly1, poly2, N_test, mod_test)
        
        # To compare with the expected centered result, we must center the output.
        # This also correctly tests the poly_mod_centered function's role.
        result_centered = poly_mod_centered(result, mod_test)
        
        self.assertPolyEqual(result_centered, expected_centered, "poly_mul simple case (X+1)(X-1) failed")
        print("poly_mul test passed.")

    def test_poly_invert(self):
        """
        Test polynomial inversion. This test is now self-contained.
        """
        print("Testing poly_invert...")
        N_test = 7
        p_test = 3
        
        # Test case: A simple invertible polynomial f = X+1
        f_test = np.array([1, 1, 0, 0, 0, 0, 0])
        
        inv_f = poly_inverse_ring(f_test, N_test, p_test)
        self.assertIsNotNone(inv_f, "poly_invert returned None for an invertible polynomial")
        
        if inv_f is not None:
            # Check if f * f_inv = 1
            product = poly_mul_ring(f_test, inv_f, N_test, p_test)
            expected_identity = np.zeros(N_test, dtype=int)
            expected_identity[0] = 1
            self.assertPolyEqual(product, expected_identity, "poly_invert failed: f * f_inv != 1")
        print("poly_invert test passed.")

    def test_ntru_full_cycle(self):
        """
        End-to-end test of NTRU key generation, encryption, and decryption.
        This is the most important test.
        """
        print("\n--- Running full NTRU cycle test ---")

        # Key Generation
        print("Generating keys...")
        try:
            public_key, private_key, invFmodP = key_gen(self.N, self.p, self.q, max_tries=200)
        except RuntimeError as e:
            self.fail(f"Key generation failed with an exception: {e}")

        # Message Generation
        message_poly = generate_random_poly(self.N, self.dm, self.dm)
        print(f"Original message m: (coefficients sum to {np.sum(message_poly)})")

        # Blinding Polynomial Generation
        blinding_poly = generate_random_poly(self.N, self.dr, self.dr)

        # Encryption
        print("Encrypting message...")
        ciphertext = ntru_encrypt(message_poly, blinding_poly, public_key, self.N, self.p, self.q)
        self.assertEqual(len(ciphertext), self.N, "Ciphertext has incorrect length")

        # Decryption
        print("Decrypting ciphertext...")
        decrypted_message = ntru_decrypt(ciphertext, private_key, invFmodP, self.N, self.p, self.q)
        self.assertEqual(len(decrypted_message), self.N, "Decrypted message has incorrect length")

        # Verify Decryption
        self.assertPolyEqual(message_poly, decrypted_message, "Decryption failed: original message does not match decrypted message!")
        print("Full NTRU cycle test PASSED: Original message matches decrypted message.")

    def test_ntru_multiple_cycles(self):
        """Run multiple full NTRU cycles with different random inputs."""
        print("\n--- Running multiple full NTRU cycles ---")
        num_tests = 5

        for i in range(num_tests):
            print(f"\n--- Running cycle {i+1}/{num_tests} ---")
            try:
                # Key Generation
                public_key, private_key, invFmodP = key_gen(self.N, self.p, self.q, max_tries=200)
                
                # Message and Blinding Polynomial Generation
                message_poly = generate_random_poly(self.N, self.dm, self.dm)
                blinding_poly = generate_random_poly(self.N, self.dr, self.dr)

                # Encryption
                ciphertext = ntru_encrypt(message_poly, blinding_poly, public_key, self.N, self.p, self.q)

                # Decryption
                decrypted_message = ntru_decrypt(ciphertext, private_key, invFmodP, self.N, self.p, self.q)

                # Verify Decryption
                self.assertPolyEqual(message_poly, decrypted_message, f"Decryption failed in cycle {i+1}")
                print(f"Cycle {i+1} PASSED.")
            except (RuntimeError, AssertionError) as e:
                self.fail(f"Test cycle {i+1} failed with an error: {e}")

        print(f"\nAll {num_tests} NTRU cycles passed successfully!")

if __name__ == '__main__':
    # This setup allows running the tests from the command line.
    unittest.main(argv=['first-arg-is-ignored'],verbosity=2, exit=False)
