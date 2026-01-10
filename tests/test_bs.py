"""
Tests for Black-Scholes pricing module.
"""

import unittest
import numpy as np
from pricing.black_scholes import black_scholes_call, black_scholes_put


class TestBlackScholes(unittest.TestCase):
    """Test cases for Black-Scholes pricing."""

    def setUp(self):
        """Set up test parameters."""
        self.S = 100.0  # Spot price
        self.K = 100.0  # Strike price
        self.T = 1.0  # 1 year to maturity
        self.r = 0.05  # 5% risk-free rate
        self.sigma = 0.2  # 20% volatility

    def test_call_price_known_value(self):
        """Test call price against known value."""
        # Known value for S=100, K=100, T=1, r=0.05, sigma=0.2
        price = black_scholes_call(self.S, self.K, self.T, self.r, self.sigma)
        expected = 10.450583572185565
        self.assertAlmostEqual(price, expected, places=6)

    def test_put_price_known_value(self):
        """Test put price against known value."""
        price = black_scholes_put(self.S, self.K, self.T, self.r, self.sigma)
        expected = 5.573526022256971
        self.assertAlmostEqual(price, expected, places=6)

    def test_put_call_parity(self):
        """Test put-call parity: C - P = S - K*e^(-rT)."""
        call_price = black_scholes_call(self.S, self.K, self.T, self.r, self.sigma)
        put_price = black_scholes_put(self.S, self.K, self.T, self.r, self.sigma)
        lhs = call_price - put_price
        rhs = self.S - self.K * np.exp(-self.r * self.T)
        self.assertAlmostEqual(lhs, rhs, places=6)

    def test_call_at_expiration(self):
        """Test call price at expiration (T=0) equals intrinsic value."""
        price = black_scholes_call(self.S, self.K, 0.0, self.r, self.sigma)
        intrinsic = max(self.S - self.K, 0.0)
        self.assertAlmostEqual(price, intrinsic, places=6)

    def test_put_at_expiration(self):
        """Test put price at expiration (T=0) equals intrinsic value."""
        price = black_scholes_put(self.S, self.K, 0.0, self.r, self.sigma)
        intrinsic = max(self.K - self.S, 0.0)
        self.assertAlmostEqual(price, intrinsic, places=6)

    def test_call_deep_itm(self):
        """Test call price for deep ITM option."""
        # Deep ITM: S >> K
        S_itm = 200.0
        price = black_scholes_call(S_itm, self.K, self.T, self.r, self.sigma)
        # Should be close to intrinsic value (discounted)
        intrinsic = S_itm - self.K
        discounted_intrinsic = intrinsic * np.exp(-self.r * self.T)
        self.assertGreater(price, discounted_intrinsic * 0.95)  # At least 95% of intrinsic

    def test_call_deep_otm(self):
        """Test call price for deep OTM option."""
        # Deep OTM: S << K
        S_otm = 50.0
        price = black_scholes_call(S_otm, self.K, self.T, self.r, self.sigma)
        # Should be small but positive
        self.assertGreater(price, 0.0)
        self.assertLess(price, 1.0)  # Should be very small

    def test_zero_volatility(self):
        """Test pricing with zero volatility."""
        call_price = black_scholes_call(self.S, self.K, self.T, self.r, 0.0)
        put_price = black_scholes_put(self.S, self.K, self.T, self.r, 0.0)
        # Should equal discounted intrinsic value
        call_intrinsic = max(self.S - self.K, 0.0) * np.exp(-self.r * self.T)
        put_intrinsic = max(self.K - self.S, 0.0) * np.exp(-self.r * self.T)
        self.assertAlmostEqual(call_price, call_intrinsic, places=6)
        self.assertAlmostEqual(put_price, put_intrinsic, places=6)

    def test_negative_inputs(self):
        """Test that negative inputs raise errors."""
        with self.assertRaises(ValueError):
            black_scholes_call(-100, self.K, self.T, self.r, self.sigma)
        with self.assertRaises(ValueError):
            black_scholes_call(self.S, -100, self.T, self.r, self.sigma)


if __name__ == "__main__":
    unittest.main()
