"""
Tests for implied volatility computation module.
"""

import unittest
import numpy as np
from pricing.implied_vol import compute_implied_vol
from pricing.black_scholes import black_scholes_call, black_scholes_put


class TestImpliedVolatility(unittest.TestCase):
    """Test cases for implied volatility computation."""

    def setUp(self):
        """Set up test parameters."""
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma_true = 0.2  # True volatility

    def test_iv_inversion_call(self):
        """Test IV inversion: price option → compute IV → re-price → should match."""
        # Price option with known volatility
        market_price = black_scholes_call(
            self.S, self.K, self.T, self.r, self.sigma_true
        )

        # Compute implied volatility
        iv = compute_implied_vol(
            market_price, self.S, self.K, self.T, self.r, "call"
        )

        # Re-price with computed IV
        recomputed_price = black_scholes_call(self.S, self.K, self.T, self.r, iv)

        # Should match original price
        self.assertAlmostEqual(market_price, recomputed_price, places=6)
        # IV should be close to true volatility
        self.assertAlmostEqual(iv, self.sigma_true, places=4)

    def test_iv_inversion_put(self):
        """Test IV inversion for put option."""
        market_price = black_scholes_put(
            self.S, self.K, self.T, self.r, self.sigma_true
        )

        iv = compute_implied_vol(
            market_price, self.S, self.K, self.T, self.r, "put"
        )

        recomputed_price = black_scholes_put(self.S, self.K, self.T, self.r, iv)

        self.assertAlmostEqual(market_price, recomputed_price, places=6)
        self.assertAlmostEqual(iv, self.sigma_true, places=4)

    def test_iv_different_strikes(self):
        """Test IV computation for different strikes."""
        strikes = [80, 90, 100, 110, 120]

        for K in strikes:
            market_price = black_scholes_call(
                self.S, K, self.T, self.r, self.sigma_true
            )
            iv = compute_implied_vol(
                market_price, self.S, K, self.T, self.r, "call"
            )
            self.assertAlmostEqual(iv, self.sigma_true, places=3)

    def test_iv_different_maturities(self):
        """Test IV computation for different maturities."""
        maturities = [0.25, 0.5, 1.0, 2.0]

        for T in maturities:
            market_price = black_scholes_call(
                self.S, self.K, T, self.r, self.sigma_true
            )
            iv = compute_implied_vol(
                market_price, self.S, self.K, T, self.r, "call"
            )
            self.assertAlmostEqual(iv, self.sigma_true, places=3)

    def test_iv_invalid_price(self):
        """Test that invalid prices raise errors."""
        # Price below intrinsic value
        with self.assertRaises(ValueError):
            compute_implied_vol(0.01, self.S, self.K, self.T, self.r, "call")

        # Negative price
        with self.assertRaises(ValueError):
            compute_implied_vol(-1.0, self.S, self.K, self.T, self.r, "call")

    def test_iv_expired_option(self):
        """Test that expired options raise errors."""
        with self.assertRaises(ValueError):
            compute_implied_vol(5.0, self.S, self.K, 0.0, self.r, "call")

    def test_iv_high_volatility(self):
        """Test IV computation for high volatility."""
        sigma_high = 1.0  # 100% volatility
        market_price = black_scholes_call(
            self.S, self.K, self.T, self.r, sigma_high
        )
        iv = compute_implied_vol(
            market_price, self.S, self.K, self.T, self.r, "call"
        )
        self.assertAlmostEqual(iv, sigma_high, places=2)

    def test_iv_low_volatility(self):
        """Test IV computation for low volatility."""
        sigma_low = 0.05  # 5% volatility
        market_price = black_scholes_call(
            self.S, self.K, self.T, self.r, sigma_low
        )
        iv = compute_implied_vol(
            market_price, self.S, self.K, self.T, self.r, "call"
        )
        self.assertAlmostEqual(iv, sigma_low, places=3)


if __name__ == "__main__":
    unittest.main()
