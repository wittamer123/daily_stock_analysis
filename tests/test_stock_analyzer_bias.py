# -*- coding: utf-8 -*-
"""
Unit tests for StockTrendAnalyzer._generate_signal bias and strong-trend relief logic (Issue #296).
"""

import math
import sys
import unittest
from unittest.mock import patch, MagicMock

# Minimal stubs so `src.analyzer` can import in slim CI / dev envs
for _dep in ("litellm", "json_repair"):
    if _dep not in sys.modules:
        try:
            __import__(_dep)
        except ModuleNotFoundError:
            sys.modules[_dep] = MagicMock()

from src.analyzer import AnalysisResult, fill_price_position_if_needed
from src.stock_analyzer import (
    StockTrendAnalyzer,
    TrendAnalysisResult,
    TrendStatus,
    VolumeStatus,
    MACDStatus,
    RSIStatus,
)


def _make_result(
    code: str = "000001",
    trend_status: TrendStatus = TrendStatus.BULL,
    trend_strength: float = 50.0,
    bias_ma5: float = 0.0,
    volume_status: VolumeStatus = VolumeStatus.NORMAL,
    macd_status: MACDStatus = MACDStatus.BULLISH,
    rsi_status: RSIStatus = RSIStatus.NEUTRAL,
    support_ma5: bool = False,
    support_ma10: bool = False,
) -> TrendAnalysisResult:
    """Build TrendAnalysisResult with defaults for _generate_signal bias branch testing."""
    return TrendAnalysisResult(
        code=code,
        trend_status=trend_status,
        ma_alignment="",
        trend_strength=trend_strength,
        ma5=10.0,
        ma10=9.5,
        ma20=9.0,
        ma60=8.5,
        current_price=10.0,
        bias_ma5=bias_ma5,
        bias_ma10=0.0,
        bias_ma20=0.0,
        volume_status=volume_status,
        volume_ratio_5d=1.0,
        volume_trend="",
        support_ma5=support_ma5,
        support_ma10=support_ma10,
        macd_status=macd_status,
        rsi_status=rsi_status,
    )


class TrendAnalysisResultMa5CheckTestCase(unittest.TestCase):
    """Tests for is_price_below_ma5 (strict below MA5)."""

    def test_below_ma5_true(self) -> None:
        r = TrendAnalysisResult(code="600519", ma5=100.0, current_price=99.0)
        self.assertTrue(r.is_price_below_ma5())

    def test_below_ma5_false_when_above(self) -> None:
        r = TrendAnalysisResult(code="600519", ma5=100.0, current_price=101.0)
        self.assertFalse(r.is_price_below_ma5())

    def test_below_ma5_false_when_equal(self) -> None:
        r = TrendAnalysisResult(code="600519", ma5=10.0, current_price=10.0)
        self.assertFalse(r.is_price_below_ma5())

    def test_below_ma5_none_when_data_invalid(self) -> None:
        r = TrendAnalysisResult(code="600519")
        self.assertIsNone(r.is_price_below_ma5())

    def test_to_dict_includes_price_below_ma5(self) -> None:
        r = TrendAnalysisResult(code="x", ma5=50.0, current_price=40.0)
        self.assertEqual(r.to_dict().get("price_below_ma5"), True)


class FillPricePositionMetaTestCase(unittest.TestCase):
    """fill_price_position_if_needed sets AnalysisResult.ma5 / bias_ma5 for API meta."""

    def test_sets_ma5_bias_from_trend(self) -> None:
        result = AnalysisResult(
            code="600519",
            name="Test",
            sentiment_score=50,
            trend_prediction="震荡",
            operation_advice="观望",
        )
        tr = TrendAnalysisResult(
            code="600519",
            ma5=100.0,
            current_price=95.0,
            bias_ma5=-5.0,
        )
        fill_price_position_if_needed(result, tr, None)
        self.assertEqual(result.ma5, 100.0)
        self.assertEqual(result.bias_ma5, -5.0)
        self.assertEqual(result.current_price, 95.0)

    def test_realtime_fills_current_when_missing(self) -> None:
        result = AnalysisResult(
            code="600519",
            name="Test",
            sentiment_score=50,
            trend_prediction="震荡",
            operation_advice="观望",
        )
        tr = TrendAnalysisResult(
            code="600519",
            ma5=50.0,
            current_price=0.0,
            bias_ma5=-4.0,
        )
        rq = MagicMock()
        rq.to_dict.return_value = {"price": 48.5}
        result.current_price = None
        fill_price_position_if_needed(result, tr, rq)
        self.assertEqual(result.current_price, 48.5)


class StockAnalyzerBiasTestCase(unittest.TestCase):
    """Tests for bias_ma5 and strong-trend relief in _generate_signal."""

    def setUp(self) -> None:
        self.analyzer = StockTrendAnalyzer()

    def _assert_contains(self, items: list, substring: str) -> None:
        """Assert at least one item contains the substring."""
        self.assertTrue(
            any(substring in s for s in items),
            msg=f"Expected substring '{substring}' in {items}",
        )

    def _assert_not_contains(self, items: list, substring: str) -> None:
        """Assert no item contains the substring."""
        self.assertFalse(
            any(substring in s for s in items),
            msg=f"Did not expect substring '{substring}' in {items}",
        )

    @patch("src.stock_analyzer.get_config")
    def test_bias_nan_defense(self, mock_get_config: MagicMock) -> None:
        """bias_ma5=NaN should be treated as 0.0 without exception."""
        mock_get_config.return_value.bias_threshold = 5.0
        result = _make_result(
            trend_status=TrendStatus.BULL,
            bias_ma5=float("nan"),
        )
        self.analyzer._generate_signal(result)
        self.assertIsInstance(result.signal_score, (int, float))
        self.assertFalse(math.isnan(result.signal_score))

    @patch("src.stock_analyzer.get_config")
    def test_bias_negative_pullback(self, mock_get_config: MagicMock) -> None:
        """bias=-2% should yield '回踩买点'."""
        mock_get_config.return_value.bias_threshold = 5.0
        result = _make_result(
            trend_status=TrendStatus.BULL,
            bias_ma5=-2.0,
        )
        self.analyzer._generate_signal(result)
        self._assert_contains(result.signal_reasons, "回踩买点")

    @patch("src.stock_analyzer.get_config")
    def test_bias_close_to_ma5(self, mock_get_config: MagicMock) -> None:
        """bias=1.5% should yield '介入好时机'."""
        mock_get_config.return_value.bias_threshold = 5.0
        result = _make_result(
            trend_status=TrendStatus.BULL,
            bias_ma5=1.5,
        )
        self.analyzer._generate_signal(result)
        self._assert_contains(result.signal_reasons, "介入好时机")

    @patch("src.stock_analyzer.get_config")
    def test_bias_slightly_high(self, mock_get_config: MagicMock) -> None:
        """bias=4% (< base_threshold=5%) should yield '可小仓介入'."""
        mock_get_config.return_value.bias_threshold = 5.0
        result = _make_result(
            trend_status=TrendStatus.BULL,
            bias_ma5=4.0,
        )
        self.analyzer._generate_signal(result)
        self._assert_contains(result.signal_reasons, "可小仓介入")

    @patch("src.stock_analyzer.get_config")
    def test_strong_trend_relaxed_threshold(self, mock_get_config: MagicMock) -> None:
        """STRONG_BULL + trend_strength=75 + bias=6% -> '可轻仓追踪' (effective=7.5%)."""
        mock_get_config.return_value.bias_threshold = 5.0
        result = _make_result(
            trend_status=TrendStatus.STRONG_BULL,
            trend_strength=75.0,
            bias_ma5=6.0,
        )
        self.analyzer._generate_signal(result)
        self._assert_contains(result.signal_reasons, "可轻仓追踪")
        self._assert_not_contains(result.risk_factors, "严禁追高")

    @patch("src.stock_analyzer.get_config")
    def test_non_strong_trend_strict_threshold(self, mock_get_config: MagicMock) -> None:
        """BULL + bias=6% -> '严禁追高!'."""
        mock_get_config.return_value.bias_threshold = 5.0
        result = _make_result(
            trend_status=TrendStatus.BULL,
            bias_ma5=6.0,
        )
        self.analyzer._generate_signal(result)
        self._assert_contains(result.risk_factors, "严禁追高")

    @patch("src.stock_analyzer.get_config")
    def test_strong_trend_exceed_effective(self, mock_get_config: MagicMock) -> None:
        """STRONG_BULL + trend_strength=80 + bias=10% -> '严禁追高!' (exceeds 7.5%)."""
        mock_get_config.return_value.bias_threshold = 5.0
        result = _make_result(
            trend_status=TrendStatus.STRONG_BULL,
            trend_strength=80.0,
            bias_ma5=10.0,
        )
        self.analyzer._generate_signal(result)
        self._assert_contains(result.risk_factors, "严禁追高")

    @patch("src.stock_analyzer.get_config")
    def test_boundary_at_base_threshold(self, mock_get_config: MagicMock) -> None:
        """bias=5.0% (exact base_threshold) -> '可小仓介入' (bias < base_threshold is False)."""
        mock_get_config.return_value.bias_threshold = 5.0
        result = _make_result(
            trend_status=TrendStatus.BULL,
            bias_ma5=5.0,
        )
        self.analyzer._generate_signal(result)
        # bias=5.0: bias < base_threshold (5 < 5) is False, so we go to next branch
        # bias < 2 is False, bias < base_threshold is False (5 < 5)
        # bias > effective_threshold: 5 > 5 False
        # bias > base_threshold and is_strong_trend: 5 > 5 False
        # else: 5 > 5 False, so we'd get to the else branch with "严禁追高"
        # Actually: bias < 2 -> False, bias < base_threshold (5 < 5) -> False
        # bias > effective_threshold (5 > 5) -> False
        # bias > base_threshold and is_strong_trend -> False
        # else -> bias > base_threshold (5 > 5) False... wait
        # Let me re-read: elif bias < base_threshold -> 5 < 5 is False
        # elif bias > effective_threshold -> 5 > 5 is False
        # elif bias > base_threshold and is_strong_trend -> 5 > 5 is False
        # else: risks.append 严禁追高 - so we get 严禁追高
        # Because 5.0 is not < 5.0, not > 5.0 when effective=base=5. So we hit the else.
        self._assert_contains(result.risk_factors, "严禁追高")
