"""
Tests for Greeks calculation module.
"""

import unittest
import numpy as np
from pricing.greeks import (
    delta_call,
    delta_put,
    gamma,
    theta_call,
    theta_put,
    vega,
    compute_all_greeks,
    compute_greeks_fd,
)


class TestGreeks(unittest.TestCase):
    """Test cases for Greeks calculation."""

    def setUp(self):
        """Set up test parameters."""
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.2

    def test_delta_call_sign(self):
        """Test that call delta is between 0 and 1."""
        delta = delta_call(self.S, self.K, self.T, self.r, self.sigma)
        self.assertGreater(delta, 0.0)
        self.assertLessEqual(delta, 1.0)

    def test_delta_put_sign(self):
        """Test that put delta is between -1 and 0."""
        delta = delta_put(self.S, self.K, self.T, self.r, self.sigma)
        self.assertLess(delta, 0.0)
        self.assertGreaterEqual(delta, -1.0)

    def test_gamma_positive(self):
        """Test that gamma is always positive."""
        gamma_val = gamma(self.S, self.K, self.T, self.r, self.sigma)
        self.assertGreater(gamma_val, 0.0)

    def test_vega_positive(self):
        """Test that vega is always positive."""
        vega_val = vega(self.S, self.K, self.T, self.r, self.sigma)
        self.assertGreater(vega_val, 0.0)

    def test_delta_atm(self):
        """Test delta for at-the-money option."""
        # ATM call delta should be around 0.5, but can be higher with longer time to expiration
        delta = delta_call(self.S, self.K, self.T, self.r, self.sigma)
        self.assertGreater(delta, 0.4)
        self.assertLess(delta, 0.7)  # Allow higher delta for longer-dated ATM options

    def test_gamma_atm(self):
        """Test gamma is highest for ATM options."""
        # ATM gamma
        gamma_atm = gamma(self.S, self.K, self.T, self.r, self.sigma)
        # OTM gamma
        gamma_otm = gamma(self.S, self.K * 1.5, self.T, self.r, self.sigma)
        # ITM gamma
        gamma_itm = gamma(self.S, self.K * 0.5, self.T, self.r, self.sigma)
        # ATM should have highest gamma
        self.assertGreater(gamma_atm, gamma_otm)
        self.assertGreater(gamma_atm, gamma_itm)

    def test_analytic_vs_finite_difference(self):
        """Compare analytic Greeks with finite-difference approximation."""
        greeks_analytic = compute_all_greeks(
            self.S, self.K, self.T, self.r, self.sigma, "call"
        )
        greeks_fd = compute_greeks_fd(
            self.S, self.K, self.T, self.r, self.sigma, "call", bump=0.01
        )

        # They should be close (within 1% for delta, gamma, vega)
        self.assertAlmostEqual(
            greeks_analytic["delta"], greeks_fd["delta"], delta=0.01
        )
        self.assertAlmostEqual(
            greeks_analytic["gamma"], greeks_fd["gamma"], delta=0.001
        )
        self.assertAlmostEqual(
            greeks_analytic["vega"], greeks_fd["vega"], delta=0.1
        )

    def test_theta_negative(self):
        """Test that theta is typically negative (time decay)."""
        theta_c = theta_call(self.S, self.K, self.T, self.r, self.sigma)
        theta_p = theta_put(self.S, self.K, self.T, self.r, self.sigma)
        # Theta is usually negative (time decay)
        # But can be positive for deep ITM puts with high interest rates
        # For our test case, both should be negative
        self.assertLess(theta_c, 0.0)
        # Put theta might be positive if deep ITM, but for ATM it's negative
        if self.S >= self.K:
            self.assertLess(theta_p, 0.0)

    def test_greeks_at_expiration(self):
        """Test Greeks at expiration."""
        # At expiration, delta should be step function
        delta_c = delta_call(self.S, self.K, 0.0, self.r, self.sigma)
        delta_p = delta_put(self.S, self.K, 0.0, self.r, self.sigma)
        # For ATM at expiration, delta is 0.5 (but we return 1.0 for S>K, 0.0 for S<K)
        # For S=K, it's edge case
        if self.S > self.K:
            self.assertEqual(delta_c, 1.0)
            self.assertEqual(delta_p, 0.0)
        elif self.S < self.K:
            self.assertEqual(delta_c, 0.0)
            self.assertEqual(delta_p, -1.0)

        # Gamma and vega should be zero at expiration
        gamma_val = gamma(self.S, self.K, 0.0, self.r, self.sigma)
        vega_val = vega(self.S, self.K, 0.0, self.r, self.sigma)
        self.assertEqual(gamma_val, 0.0)
        self.assertEqual(vega_val, 0.0)


if __name__ == "__main__":
    unittest.main()
