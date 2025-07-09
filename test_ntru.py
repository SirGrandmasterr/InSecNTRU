# test_ntru.py

import unittest
import numpy as np
import random
import sys
import os
from keyCreator import key_gen, poly_inverse_ring, poly_mul_ring

# Ensure the ntru_implementation.py file is in the same directory or on the Python path
try:
    from ntru_implementation import (
        poly_add,
        poly_mod_coeffs,
        poly_mod_centered,
        generate_random_poly,
        ntru_encrypt,
        ntru_decrypt
    )
except ImportError:
    print("Error: Could not import functions from 'ntru_implementation.py'.")
    print("Please ensure 'ntru_implementation.py' is in the same directory.")
    sys.exit(1)

    

class TestNTRU(unittest.TestCase):

    def setUp(self):
        """
        Set up common parameters for tests.
        These are small parameters suitable for testing, not for production security.
        """
        self.N = 509  # A small prime N
        self.p = 7  # Small modulus
        self.q = 2048 # Larger modulus, coprime to p

        # Number of 1s and -1s for polynomials f, g, r
        # For N=7, these values are small.
        # Ensure df, dg, dr are chosen such that df + (df-1) <= N-1 (for f's non-zero coeffs)
        # and dg + dg <= N-1 (for g's non-zero coeffs)
        self.df = 2
        self.dg = 2
        self.dr = 2

        # Example message parameters (usually small coefficients)
        self.dm = 1 # Number of 1s and -1s in the message polynomial
        
        print(f"\n--- Running tests with N={self.N}, p={self.p}, q={self.q} ---")

    def assertPolyEqual(self, poly1, poly2, msg=None):
        """Custom assertion for comparing two polynomials (numpy arrays)."""
        self.assertTrue(np.array_equal(poly1, poly2), msg=f"Polynomials are not equal: {poly1} vs {poly2}. {msg if msg else ''}")

    def test_poly_add(self):
        """Test polynomial addition modulo a given integer."""
        print("Testing poly_add...")
        poly1 = np.array([1, 2, 3])
        poly2 = np.array([4, 5, 6])
        mod = 7
        expected = np.array([5, 0, 2]) # (1+4)%7, (2+5)%7, (3+6)%7

        result = poly_add(poly1, poly2, mod)
        self.assertPolyEqual(result, expected, "poly_add simple case failed")

        poly3 = np.array([1, -1, 0])
        poly4 = np.array([0, 1, -1])
        mod_neg = 3
        expected_neg = np.array([1, 0, 2]) # (1+0)%3, (-1+1)%3, (0-1)%3 = (-1)%3 = 2

        result_neg = poly_add(poly3, poly4, mod_neg)
        self.assertPolyEqual(result_neg, expected_neg, "poly_add with negative coefficients failed")
        print("poly_add test passed.")

    def test_poly_mul(self):
        """Test polynomial multiplication (cyclic convolution) modulo X^N-1 and a modulus."""
        print("Testing poly_mul (cyclic convolution)...")
        # N=7, (X+1)(X-1) = X^2-1
        poly1 = np.array([1, 1, 0, 0, 0, 0, 0]) # X+1
        poly2 = np.array([-1, 1, 0, 0, 0, 0, 0]) # X-1
        N_test = 7
        mod_test = 100 # Large enough not to affect coefficients for now
        expected = np.array([-1, 0, 1, 0, 0, 0, 0]) # X^2-1
        result = poly_mul_ring(poly1, poly2, N_test, mod_test)
        self.assertPolyEqual(result, expected, "poly_mul simple case (X+1)(X-1) failed")

        # Test cyclic property: X^(N-1) * X = X^N = 1 mod (X^N-1)
        poly_x_n_minus_1 = np.zeros(N_test)
        poly_x_n_minus_1[N_test-1] = 1 # X^(N-1)
        poly_x = np.zeros(N_test)
        poly_x[1] = 1 # X
        expected_cyclic = np.zeros(N_test)
        expected_cyclic[0] = 1 # 1
        result_cyclic = poly_mul_ring(poly_x_n_minus_1, poly_x, N_test, mod_test)
        self.assertPolyEqual(result_cyclic, expected_cyclic, "poly_mul cyclic property failed (X^(N-1) * X)")

        # Test with coefficients wrapping around modulus
        poly_a = np.array([1, 2])
        poly_b = np.array([3, 4])
        N_small = 2 # (a0 + a1X)(b0 + b1X) = a0b0 + (a0b1+a1b0)X + a1b1X^2
                    # mod X^2-1: a0b0 + a1b1 + (a0b1+a1b0)X
        mod_small = 5
        # a0b0 + a1b1 = 1*3 + 2*4 = 3 + 8 = 11 => 1 mod 5
        # a0b1 + a1b0 = 1*4 + 2*3 = 4 + 6 = 10 => 0 mod 5
        expected_wrap = np.array([1, 0])
        result_wrap = poly_mul_ring(poly_a, poly_b, N_small, mod_small)
        self.assertPolyEqual(result_wrap, expected_wrap, "poly_mul with modulus wrap failed")
        print("poly_mul test passed.")

    def test_poly_mod_coeffs(self):
        """Test coefficient reduction to [0, modulus-1)."""
        print("Testing poly_mod_coeffs...")
        poly = np.array([-5, 0, 12, 7])
        mod = 7
        expected = np.array([2, 0, 5, 0]) # (-5)%7=2, 0%7=0, 12%7=5, 7%7=0
        result = poly_mod_coeffs(poly, mod)
        self.assertPolyEqual(result, expected, "poly_mod_coeffs failed")
        print("poly_mod_coeffs test passed.")

    def test_poly_mod_centered(self):
        """Test coefficient reduction to [-modulus/2, modulus/2)."""
        print("Testing poly_mod_centered...")
        poly = np.array([-5, 0, 12, 7, 20])
        mod = 7
        # -5 mod 7 = 2. Centered: 2 (since 2 is in [-3.5, 3.5))
        # 0 mod 7 = 0. Centered: 0
        # 12 mod 7 = 5. Centered: 5-7 = -2 (since 5 is not in [-3.5, 3.5))
        # 7 mod 7 = 0. Centered: 0
        # 20 mod 7 = 6. Centered: 6-7 = -1
        expected = np.array([2, 0, -2, 0, -1])
        result = poly_mod_centered(poly, mod)
        self.assertPolyEqual(result, expected, "poly_mod_centered failed")

        mod_even = 10
        poly_even = np.array([13, -7, 5, 10])
        # 13 mod 10 = 3. Centered: 3 (in [-5, 5))
        # -7 mod 10 = 3. Centered: 3
        # 5 mod 10 = 5. Centered: 5 (or -5 depending on convention, but usually [-mod/2, mod/2))
        # 10 mod 10 = 0. Centered: 0
        expected_even = np.array([3, 3, 5, 0]) # Assuming 5 is included in the positive half
        result_even = poly_mod_centered(poly_even, mod_even)
        self.assertPolyEqual(result_even, expected_even, "poly_mod_centered with even modulus failed")
        print("poly_mod_centered test passed.")

    def test_poly_invert(self):
        """
        Test polynomial inversion using the Extended Euclidean Algorithm.
        This test relies heavily on a correct `poly_invert` implementation.
        """
        print("Testing poly_invert...")
        # Test case 1: A simple invertible polynomial
        # For N=7, p=3, f = X^2 + X + 1
        f_test_p = np.array([1, 1, 1, 0, 0, 0, 0])
        inv_f_p = poly_inverse_ring(f_test_p, self.N, self.p)
        self.assertIsNotNone(inv_f_p, "poly_invert returned None for a potentially invertible poly (mod p)")
        if inv_f_p is not None:
            product_p = poly_mul_ring(f_test_p, inv_f_p, self.N, self.p)
            expected_p = np.zeros(self.N)
            expected_p[0] = 1 # Should be 1 mod (X^N-1)
            self.assertPolyEqual(product_p, expected_p, "poly_invert (mod p) failed: f * f_inv != 1")

        # Test case 2: For N=7, q=41, f = X^2 + X + 1
        f_test_q = np.array([1, 1, 1, 0, 0, 0, 0])
        inv_f_q = poly_inverse_ring(f_test_q, self.N, self.q)
        self.assertIsNotNone(inv_f_q, "poly_invert returned None for a potentially invertible poly (mod q)")
        if inv_f_q is not None:
            product_q = poly_mul_ring(f_test_q, inv_f_q, self.N, self.q)
            expected_q = np.zeros(self.N)
            expected_q[0] = 1 # Should be 1 mod (X^N-1)
            self.assertPolyEqual(product_q, expected_q, "poly_invert (mod q) failed: f * f_inv != 1")

        # Test case 3: A polynomial that should NOT be invertible (e.g., has a factor in common with X^N-1)
        # For N=7, (X-1) is a factor of X^7-1. So, if f=(X-1), it should not be invertible.
        # This test might require specific handling in your poly_invert to return None or raise an error.
        f_non_invertible = np.array([-1, 1, 0, 0, 0, 0, 0]) # X-1
        inv_non_inv = poly_inverse_ring(f_non_invertible, self.N, self.p) # Or self.q
        self.assertIsNone(inv_non_inv, "poly_invert returned an inverse for a non-invertible polynomial")
        print("poly_invert test passed (check for None for non-invertible cases).")


    def test_generate_random_poly(self):
        """Test generation of random polynomials with specific coefficient counts."""
        print("Testing generate_random_poly...")
        rand_poly = generate_random_poly(self.N, self.df, self.df - 1) # f-like poly
        self.assertEqual(len(rand_poly), self.N, "Generated poly has incorrect length")
        ones = np.sum(rand_poly == 1)
        minus_ones = np.sum(rand_poly == -1)
        zeros = np.sum(rand_poly == 0)
        self.assertEqual(ones, self.df, f"Incorrect number of 1s: {ones} vs {self.df}")
        self.assertEqual(minus_ones, self.df - 1, f"Incorrect number of -1s: {minus_ones} vs {self.df - 1}")
        self.assertEqual(ones + minus_ones + zeros, self.N, "Total coefficients do not sum to N")
        print("generate_random_poly test passed.")

    def test_ntru_full_cycle(self):
        """
        End-to-end test of NTRU key generation, encryption, and decryption.
        This is the most important test.
        """
        print("\n--- Running full NTRU cycle test ---")

        # Key Generation
        print("Generating keys...")
        public_key, private_key, invFmodP = key_gen(self.N, self.p, self.q, 100)

        self.assertIsNotNone(public_key, "Key generation failed: f is None")
        self.assertIsNotNone(private_key, "Key generation failed: fp_inv is None")
        self.assertIsNotNone(invFmodP, "Key generation failed: fq_inv is None")

        print(f"Generated publicKey: {public_key}")
        print(f"Generated privateKey: {private_key}")
        print(f"Generated invFmodP: {invFmodP}")

        # Message Generation
        # A simple message polynomial (e.g., 1, 0, -1, 0, ...)
        message_poly = generate_random_poly(self.N, self.dm, self.dm)
        print(f"Original message m: {message_poly}")

        # Blinding Polynomial Generation
        blinding_poly = generate_random_poly(self.N, self.dr, self.dr)
        print(f"Blinding polynomial r: {blinding_poly}")

        # Encryption
        print("Encrypting message...")
        ciphertext = ntru_encrypt(message_poly, blinding_poly, public_key, self.N, self.p, self.q)
        self.assertIsNotNone(ciphertext, "Encryption failed: ciphertext is None")
        self.assertEqual(len(ciphertext), self.N, "Ciphertext has incorrect length")
        print(f"Ciphertext e: {ciphertext}")

        # Decryption
        print("Decrypting ciphertext...")
        decrypted_message = ntru_decrypt(ciphertext, private_key, invFmodP, self.N, self.p, self.q)
        self.assertIsNotNone(decrypted_message, "Decryption failed: decrypted_message is None")
        self.assertEqual(len(decrypted_message), self.N, "Decrypted message has incorrect length")
        print(f"Decrypted message m': {decrypted_message}")

        # Verify Decryption
        self.assertPolyEqual(message_poly, decrypted_message, "Decryption failed: original message does not match decrypted message!")
        print("Full NTRU cycle test PASSED: Original message matches decrypted message.")

    def test_ntru_multiple_cycles(self):
        """Run multiple full NTRU cycles with different random inputs."""
        print("\n--- Running multiple full NTRU cycles ---")
        num_tests = 5 # Increase for more rigorous testing

        for i in range(num_tests):
            print(f"\n--- Running cycle {i+1}/{num_tests} ---")
            # Key Generation
            public_key, private_key, invFmodP = key_gen(self.N, self.p, self.q, 100)
            self.assertIsNotNone(public_key, "Key generation failed in multiple cycles")
            self.assertIsNotNone(private_key, "Key generation failed in multiple cycles")
            self.assertIsNotNone(invFmodP, "Key generation failed in multiple cycles")
            

            # Message and Blinding Polynomial Generation
            message_poly = generate_random_poly(self.N, self.dm, self.dm)
            blinding_poly = generate_random_poly(self.N, self.dr, self.dr)

            # Encryption
            ciphertext = ntru_encrypt(message_poly, blinding_poly, public_key, self.N, self.p, self.q)
            self.assertIsNotNone(ciphertext, "Encryption failed in multiple cycles")

            # Decryption
            decrypted_message = ntru_decrypt(ciphertext, private_key, invFmodP, self.N, self.p, self.q)
            self.assertIsNotNone(decrypted_message, "Decryption failed in multiple cycles")

            # Verify Decryption
            self.assertPolyEqual(message_poly, decrypted_message, f"Decryption failed in cycle {i+1}: original message does not match decrypted message!")
            print(f"Cycle {i+1} PASSED.")

        print(f"\nAll {num_tests} NTRU cycles passed successfully!")

if __name__ == '__main__':

    unittest.main(argv=['first-arg-is-ignored'], exit=False) # exit=False to allow running in environments like Jupyter