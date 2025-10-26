"""Tests for financial mathematics tools."""

import json
import pytest


@pytest.mark.asyncio
async def test_financial_fv(mcp_client):
    """Test future value calculation."""
    result = await mcp_client.call_tool(
        "financial_calcs", {"calculation": "fv", "rate": 0.05, "periods": 10, "payment": -100}
    )
    data = json.loads(result.content[0].text)
    # FV of annuity: PMT * ((1+r)^n - 1) / r
    expected = 100 * (((1.05) ** 10 - 1) / 0.05)
    assert abs(data["result"] - expected) < 1


@pytest.mark.asyncio
async def test_financial_pv(mcp_client):
    """Test present value calculation (annuity)."""
    result = await mcp_client.call_tool(
        "financial_calcs", {"calculation": "pv", "rate": 0.05, "periods": 10, "payment": -100}
    )
    data = json.loads(result.content[0].text)
    # PV calculation returns negative value representing outflow
    assert "result" in data


@pytest.mark.asyncio
async def test_financial_pv_lump_sum(mcp_client):
    """Test present value of a lump sum."""
    # What is the present value of £10,000 received in 10 years at 5% interest?
    # PV = FV / (1 + r)^n = 10000 / (1.05^10) = 6139.13
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.05, "periods": 10, "future_value": 10000}
    )
    data = json.loads(result.content[0].text)
    expected = 10000 / (1.05 ** 10)  # 6139.13
    assert abs(data["result"] - (-expected)) < 0.01  # Negative because it's cash outflow


@pytest.mark.asyncio
async def test_financial_pv_lump_sum_zero_rate(mcp_client):
    """Test present value of lump sum with zero interest rate."""
    # With 0% rate, PV = FV (money doesn't change value)
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.0, "periods": 10, "future_value": 10000}
    )
    data = json.loads(result.content[0].text)
    assert abs(data["result"] - (-10000.0)) < 0.01


@pytest.mark.asyncio
async def test_financial_pv_missing_both_params(mcp_client):
    """Test error when PV is calculated without future_value or payment."""
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "financial_calcs",
            {"calculation": "pv", "rate": 0.05, "periods": 10}
        )
    assert ("future_value" in str(exc_info.value).lower() and "payment" in str(exc_info.value).lower())


@pytest.mark.asyncio
async def test_financial_pv_coupon_bond(mcp_client):
    """Test present value of a coupon bond (combined payment + future_value)."""
    # £1,000 bond paying £30 annual coupons for 10 years at 5% yield
    # PV = PV(face value) + PV(coupons)
    # PV(face value) = 1000 / (1.05^10) = £613.91
    # PV(coupons) = 30 × [(1 - 1.05^-10) / 0.05] = £231.65
    # Total = £845.56
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.05, "periods": 10, "payment": 30, "future_value": 1000}
    )
    data = json.loads(result.content[0].text)
    # PV of lump sum component
    pv_face_value = 1000 / (1.05 ** 10)  # 613.91
    # PV of annuity component
    pv_coupons = 30 * ((1 - (1.05 ** -10)) / 0.05)  # 231.65
    expected = -(pv_face_value + pv_coupons)  # -845.56 (negative = outflow)
    assert abs(data["result"] - expected) < 0.01


@pytest.mark.asyncio
async def test_financial_npv(mcp_client):
    """Test net present value calculation."""
    cash_flows = [-1000, 300, 400, 500, 200]
    result = await mcp_client.call_tool(
        "financial_calcs", {"calculation": "npv", "rate": 0.1, "cash_flows": cash_flows}
    )
    data = json.loads(result.content[0].text)
    assert "result" in data


@pytest.mark.asyncio
async def test_compound_interest_annual(mcp_client):
    """Test compound interest with annual compounding."""
    result = await mcp_client.call_tool(
        "compound_interest",
        {"principal": 1000, "rate": 0.05, "time": 10, "frequency": "annual"},
    )
    data = json.loads(result.content[0].text)
    expected = 1000 * (1.05**10)
    assert abs(data["result"] - expected) < 0.01


@pytest.mark.asyncio
async def test_compound_interest_continuous(mcp_client):
    """Test compound interest with continuous compounding."""
    result = await mcp_client.call_tool(
        "compound_interest",
        {"principal": 1000, "rate": 0.05, "time": 10, "frequency": "continuous"},
    )
    data = json.loads(result.content[0].text)
    # Continuous: A = Pe^(rt)
    import math

    expected = 1000 * math.exp(0.05 * 10)
    assert abs(data["result"] - expected) < 0.01


@pytest.mark.asyncio
async def test_financial_pmt_basic(mcp_client):
    """Test payment (PMT) calculation for a loan."""
    # Loan: $10,000 at 5% for 12 periods
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pmt", "rate": 0.05, "periods": 12, "present_value": -10000},
    )
    data = json.loads(result.content[0].text)
    # PMT should be positive (payment outflow)
    assert data["result"] > 0
    # Expected PMT ≈ 1128.25
    assert 1100 < data["result"] < 1150


@pytest.mark.asyncio
async def test_financial_pmt_zero_rate(mcp_client):
    """Test PMT calculation with zero interest rate."""
    # With 0% rate, PMT = PV / periods
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pmt", "rate": 0.0, "periods": 10, "present_value": -1000},
    )
    data = json.loads(result.content[0].text)
    # PMT = 1000 / 10 = 100
    assert abs(data["result"] - 100.0) < 0.01


@pytest.mark.asyncio
async def test_financial_pmt_missing_params(mcp_client):
    """Test error when required parameters are missing for PMT."""
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool("financial_calcs", {"calculation": "pmt", "rate": 0.05})
    assert "requires" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_financial_irr_simple_investment(mcp_client):
    """Test IRR for a simple investment."""
    # Initial investment of -$1000, returns of $500, $400, $300, $200
    cash_flows = [-1000, 500, 400, 300, 200]
    result = await mcp_client.call_tool(
        "financial_calcs", {"calculation": "irr", "rate": 0.1, "cash_flows": cash_flows}
    )
    data = json.loads(result.content[0].text)
    # IRR should be positive for profitable investment
    assert data["result"] > 0
    # Expected IRR approximately 0.149 (14.9%)
    assert 0.12 < data["result"] < 0.18


@pytest.mark.asyncio
async def test_financial_irr_complex_cash_flows(mcp_client):
    """Test IRR with more complex cash flow pattern."""
    # Project with initial investment and varying returns
    cash_flows = [-5000, 1000, 1500, 2000, 2500, 1000]
    result = await mcp_client.call_tool(
        "financial_calcs", {"calculation": "irr", "rate": 0.1, "cash_flows": cash_flows}
    )
    data = json.loads(result.content[0].text)
    # IRR should be positive
    assert data["result"] > 0
    assert data["result"] < 1.0  # Should be reasonable (< 100%)


@pytest.mark.asyncio
async def test_financial_irr_negative_return(mcp_client):
    """Test IRR for a losing investment."""
    # Investment that loses money
    cash_flows = [-1000, 100, 150, 200, 100]
    result = await mcp_client.call_tool(
        "financial_calcs", {"calculation": "irr", "rate": 0.1, "cash_flows": cash_flows}
    )
    data = json.loads(result.content[0].text)
    # IRR should be negative for unprofitable investment
    assert data["result"] < 0


@pytest.mark.asyncio
async def test_financial_irr_insufficient_cash_flows(mcp_client):
    """Test error when IRR has insufficient cash flows."""
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "financial_calcs", {"calculation": "irr", "rate": 0.1, "cash_flows": [-1000]}
        )
    assert "at least 2" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_financial_irr_convergence(mcp_client):
    """Test IRR convergence with standard cash flows."""
    # Well-behaved cash flows that should converge quickly
    # Initial investment of $10,000, returns of $3,000 for 5 years
    cash_flows = [-10000, 3000, 3000, 3000, 3000, 3000]
    result = await mcp_client.call_tool(
        "financial_calcs", {"calculation": "irr", "rate": 0.1, "cash_flows": cash_flows}
    )
    data = json.loads(result.content[0].text)
    # Expected IRR: 15.24% (from audit and verified calculation)
    assert abs(data["result"] - 0.1524) < 0.0001


@pytest.mark.asyncio
async def test_financial_fv_zero_rate(mcp_client):
    """Test future value calculation with zero interest rate."""
    # With 0% rate, FV = PV + (PMT × periods)
    result = await mcp_client.call_tool(
        "financial_calcs", {"calculation": "fv", "rate": 0.0, "periods": 10, "payment": -100}
    )
    data = json.loads(result.content[0].text)
    # FV = 100 × 10 = 1000
    assert abs(data["result"] - 1000.0) < 0.01


@pytest.mark.asyncio
async def test_financial_pv_zero_rate(mcp_client):
    """Test present value calculation with zero interest rate."""
    # With 0% rate, PV = PMT × periods
    result = await mcp_client.call_tool(
        "financial_calcs", {"calculation": "pv", "rate": 0.0, "periods": 10, "payment": -100}
    )
    data = json.loads(result.content[0].text)
    # PV = 100 × 10 = 1000
    assert abs(data["result"] - 1000.0) < 0.01


@pytest.mark.asyncio
async def test_financial_npv_missing_cash_flows(mcp_client):
    """Test error when NPV is calculated without cash flows."""
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool("financial_calcs", {"calculation": "npv", "rate": 0.1})
    assert "requires" in str(exc_info.value).lower() or "cash_flows" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_compound_interest_semi_annual(mcp_client):
    """Test compound interest with semi-annual compounding."""
    result = await mcp_client.call_tool(
        "compound_interest",
        {"principal": 1000, "rate": 0.06, "time": 5, "frequency": "semi-annual"},
    )
    data = json.loads(result.content[0].text)
    # Semi-annual: n=2, A = 1000 * (1 + 0.06/2)^(2*5)
    expected = 1000 * (1 + 0.06 / 2) ** (2 * 5)
    assert abs(data["result"] - expected) < 0.01


@pytest.mark.asyncio
async def test_compound_interest_quarterly(mcp_client):
    """Test compound interest with quarterly compounding."""
    result = await mcp_client.call_tool(
        "compound_interest",
        {"principal": 1000, "rate": 0.08, "time": 3, "frequency": "quarterly"},
    )
    data = json.loads(result.content[0].text)
    # Quarterly: n=4, A = 1000 * (1 + 0.08/4)^(4*3)
    expected = 1000 * (1 + 0.08 / 4) ** (4 * 3)
    assert abs(data["result"] - expected) < 0.01


@pytest.mark.asyncio
async def test_compound_interest_monthly(mcp_client):
    """Test compound interest with monthly compounding."""
    result = await mcp_client.call_tool(
        "compound_interest",
        {"principal": 5000, "rate": 0.05, "time": 2, "frequency": "monthly"},
    )
    data = json.loads(result.content[0].text)
    # Monthly: n=12, A = 5000 * (1 + 0.05/12)^(12*2)
    expected = 5000 * (1 + 0.05 / 12) ** (12 * 2)
    assert abs(data["result"] - expected) < 0.01


@pytest.mark.asyncio
async def test_compound_interest_daily(mcp_client):
    """Test compound interest with daily compounding."""
    result = await mcp_client.call_tool(
        "compound_interest",
        {"principal": 2000, "rate": 0.04, "time": 1, "frequency": "daily"},
    )
    data = json.loads(result.content[0].text)
    # Daily: n=365, A = 2000 * (1 + 0.04/365)^(365*1)
    expected = 2000 * (1 + 0.04 / 365) ** (365 * 1)
    assert abs(data["result"] - expected) < 0.01


# ============================================================================
# Comprehensive Real-World TVM Test Scenarios
# ============================================================================


# Section 1: Basic Lump Sum Problems


@pytest.mark.asyncio
async def test_financial_fv_sales_growth(mcp_client):
    """Test FV: $100M at 8% for 10 years (Problem 1.1)."""
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "fv", "rate": 0.08, "periods": 10, "present_value": -100, "payment": 0}
    )
    data = json.loads(result.content[0].text)
    # Expected: $215.89 million
    assert abs(data["result"] - 215.89) < 0.01


@pytest.mark.asyncio
async def test_financial_pv_government_bond(mcp_client):
    """Test PV: $1000 in 3 years at 4% (Problem 1.2)."""
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.04, "periods": 3, "future_value": 1000, "payment": 0}
    )
    data = json.loads(result.content[0].text)
    # Expected: -$889.00 (negative = cash outflow to purchase)
    assert abs(data["result"] - (-889.00)) < 0.01


@pytest.mark.asyncio
async def test_financial_rate_treasury_bond(mcp_client):
    """Test rate solving: $613.81 → $1000 in 10 years (Problem 1.3)."""
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "rate", "periods": 10, "present_value": -613.81, "future_value": 1000}
    )
    data = json.loads(result.content[0].text)
    # Expected: 5.00%
    assert abs(data["result"] - 0.05) < 0.0001


# Section 2: Annuity Problems


@pytest.mark.asyncio
async def test_financial_pv_annuity_payment(mcp_client):
    """Test PV of annuity: $1000/year for 5 years at 6% (Problem 2.1)."""
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.06, "periods": 5, "payment": -1000, "future_value": 0}
    )
    data = json.loads(result.content[0].text)
    # Expected: 4212.36 (amount you'd pay to receive the annuity)
    assert abs(data["result"] - 4212.36) < 0.01


@pytest.mark.asyncio
async def test_financial_pmt_withdrawal(mcp_client):
    """Test PMT: Withdraw from $200k at 6% over 15 years (Problem 2.2)."""
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pmt", "rate": 0.06, "periods": 15, "present_value": -200000, "future_value": 0}
    )
    data = json.loads(result.content[0].text)
    # Expected: $20,592.55
    assert abs(data["result"] - 20592.55) < 0.01


@pytest.mark.asyncio
async def test_financial_pmt_mortgage(mcp_client):
    """Test PMT: $190k mortgage at 7%/12 for 360 months (Problem 2.3)."""
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pmt", "rate": 0.07/12, "periods": 360, "present_value": -190000, "future_value": 0}
    )
    data = json.loads(result.content[0].text)
    # Expected: $1,264 (approximately)
    assert abs(data["result"] - 1264) < 1


# Section 3: Bond Pricing Problems


@pytest.mark.asyncio
async def test_financial_pv_zero_coupon_bond_100k(mcp_client):
    """Test PV: Zero-coupon bond $100k at 10% for 4 years (Problem 3.1)."""
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.10, "periods": 4, "future_value": 100000, "payment": 0}
    )
    data = json.loads(result.content[0].text)
    # Expected: -$68,301
    assert abs(data["result"] - (-68301)) < 1


@pytest.mark.asyncio
async def test_financial_pv_coupon_bond_annual(mcp_client):
    """Test PV: $100k bond, $7k coupons, 9% yield, 15 years (Problem 3.2)."""
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.09, "periods": 15, "payment": 7000, "future_value": 100000}
    )
    data = json.loads(result.content[0].text)
    # Expected: -$83,879 (bond trades at discount)
    assert abs(data["result"] - (-83879)) < 1


@pytest.mark.asyncio
async def test_financial_pv_coupon_bond_semiannual(mcp_client):
    """Test PV: $100k bond, $4k semi-annual coupons, 3.5% rate, 10 periods (Problem 3.3)."""
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.07/2, "periods": 10, "payment": 4000, "future_value": 100000}
    )
    data = json.loads(result.content[0].text)
    # Expected: -$104,158.30 (bond trades at premium)
    # Corrected from audit: Previous comment had wrong expected value of -$104,376
    assert abs(data["result"] - (-104158.30)) < 1


@pytest.mark.asyncio
async def test_financial_pv_bond_discount(mcp_client):
    """Test PV: $1000 bond, 5% coupon, 6% yield, semi-annual (Problem 3.4)."""
    # Semi-annual: $25 coupon, 20 periods, 3% per period
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.03, "periods": 20, "payment": 25, "future_value": 1000}
    )
    data = json.loads(result.content[0].text)
    # Expected: -$925.61 (trades at discount)
    assert abs(data["result"] - (-925.61)) < 0.01


@pytest.mark.asyncio
async def test_financial_pv_openstax_bond(mcp_client):
    """Test PV: OpenStax 4% coupon bond, 5% YTM, semi-annual (Problem 3.5)."""
    # Semi-annual: $20 coupon, 30 periods, 2.5% per period
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.025, "periods": 30, "payment": 20, "future_value": 1000}
    )
    data = json.loads(result.content[0].text)
    # Expected: -$895.35
    assert abs(data["result"] - (-895.35)) < 0.01


# Section 4: Uneven Cash Flow Problems


@pytest.mark.asyncio
async def test_financial_npv_uneven_cashflows(mcp_client):
    """Test NPV: 10-year uneven cash flow stream at 8% (Problem 4.1)."""
    cash_flows = [0, 10000, 10000, 10000, 12000, 12000, 12000, 12000, 15000, 15000, 15000]
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "npv", "rate": 0.08, "cash_flows": cash_flows}
    )
    data = json.loads(result.content[0].text)
    # Expected: $79,877
    assert abs(data["result"] - 79877) < 1


@pytest.mark.asyncio
async def test_financial_rate_savings_from_zero(mcp_client):
    """Test rate: Savings from zero with quarterly contributions (Problem: PV=0 regression)."""
    # Start with £0, save £30k quarterly, reach £550k in 4 years (16 quarters)
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "rate", "periods": 16, "payment": -30000, "present_value": 0, "future_value": 550000}
    )
    data = json.loads(result.content[0].text)
    # Expected: ~1.79% per quarter
    assert abs(data["result"] - 0.0179) < 0.001
    # Verify PV was included in metadata
    assert data["present_value"] == 0


@pytest.mark.asyncio
async def test_financial_rate_annuity_due(mcp_client):
    """Test rate: Annuity due with payments at beginning (when='begin')."""
    # SmartChoice laptop: $399 cash or $59.88/month in advance for 12 months
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "rate", "periods": 12, "payment": 59.88, "present_value": -399, "future_value": 0, "when": "begin"}
    )
    data = json.loads(result.content[0].text)
    # Expected: ~13.1% monthly (~157% APR) for annuity due
    assert 0.12 < data["result"] < 0.14  # Between 12% and 14% monthly
    # Verify when parameter in metadata
    assert data["when"] == "begin"


@pytest.mark.asyncio
async def test_financial_pmt_annuity_due(mcp_client):
    """Test PMT: Payment calculation with annuity due (when='begin')."""
    # Lease with payments at start of month
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pmt", "rate": 0.005, "periods": 36, "present_value": -20000, "future_value": 0, "when": "begin"}
    )
    data = json.loads(result.content[0].text)
    # Payment should be slightly less than ordinary annuity due to earlier compounding
    assert data["result"] > 0
    # Annuity due payment should be less than ordinary annuity
    assert 600 < data["result"] < 610  # Ordinary: ~$608.44, Annuity due: ~$605.41


# ============================================================================
# CFA and Textbook Verified Test Scenarios (From Independent Audit)
# ============================================================================


@pytest.mark.asyncio
async def test_financial_deferred_annuity_cfa(mcp_client):
    """Test deferred annuity from CFA Level 1 problem set.

    Source: CFA Level 1 - SlideShare Problem A
    Problem: Calculate PV of instrument that pays $20,000 at end of each year
    for 4 years starting 3 years from now. Discount rate: 8%

    Solution:
    Step 1: Find PV at year 3 (when annuity starts)
    PV₃ = 20,000 × [(1 - 1.08⁻⁴) / 0.08] = $66,242.54

    Step 2: Discount back to today
    PV₀ = 66,242.54 / 1.08³ = $52,585.46
    """
    # Step 1: PV at year 3 (when annuity starts)
    result_at_start = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.08, "periods": 4, "payment": -20000, "future_value": 0}
    )
    data_at_start = json.loads(result_at_start.content[0].text)
    assert abs(data_at_start["result"] - 66242.54) < 0.01

    # Step 2: Discount back to today
    result_today = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.08, "periods": 3, "future_value": -data_at_start["result"], "payment": 0}
    )
    data_today = json.loads(result_today.content[0].text)
    assert abs(data_today["result"] - 52585.46) < 0.01


@pytest.mark.asyncio
async def test_financial_negative_interest_rate_swiss_bond(mcp_client):
    """Test negative interest rate scenario from CFA curriculum.

    Source: AnalystPrep CFA Level 1 - Time Value of Money
    Problem: Swiss government 15-year bond at -0.08% yield.
    PV of CHF 100 face value at issuance?

    Solution:
    N = 15 years, I/Y = -0.08%, FV = 100, PMT = 0 (zero-coupon)
    PV = 100 / (1 - 0.0008)^15 = CHF 101.20

    Note: PV > FV when rate is negative
    """
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": -0.0008, "periods": 15, "future_value": 100, "payment": 0}
    )
    data = json.loads(result.content[0].text)
    # PV should be greater than FV (negative result represents cost)
    assert data["result"] < -101.00  # Negative because it's cash outflow
    assert data["result"] > -102.00
    # More precise check
    assert abs(data["result"] - (-101.20)) < 0.01


@pytest.mark.asyncio
async def test_financial_annuity_due_cfa_problem(mcp_client):
    """Test annuity due from CFA Level 1 end-of-chapter question.

    Source: CFA Level 1 PDF - SlideShare
    Problem: Investment pays 300 Euros annually for 5 years, first payment today.
    PV at 4% discount rate is closest to:
    A) 1336  B) 1389  C) 1625

    Calculator Solution:
    N = 5, I/YR = 4%, PMT = 300, FV = 0, BGN mode
    CPT PV = -1,388.968567

    Answer: B) 1389
    """
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.04, "periods": 5, "payment": -300, "future_value": 0, "when": "begin"}
    )
    data = json.loads(result.content[0].text)
    assert abs(data["result"] - 1388.97) < 0.01
    # Verify when parameter in metadata
    assert data["when"] == "begin"


@pytest.mark.asyncio
async def test_financial_balloon_payment_mortgage(mcp_client):
    """Test balloon mortgage from Omnicalculator validation.

    Source: Omnicalculator - Balloon Payment Calculator
    Problem: Jack's $100,000 balloon mortgage:
    - Term: 5 years
    - Amortisation: 30 years
    - Interest rate: 7%
    - Monthly payment: $665.30
    - Balloon payment (after 5 years): $94,131.59

    Solution:
    Step 1: Calculate monthly payment based on 30-year amortisation
    N = 360, I/Y = 7/12, PV = -100,000, FV = 0
    CPT PMT = $665.30

    Step 2: Calculate remaining balance after 60 payments
    N = 60, I/Y = 7/12, PMT = $665.30, PV = -100,000
    CPT FV = $94,131.59
    """
    # Step 1: Get monthly payment
    result_pmt = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pmt", "rate": 0.07/12, "periods": 360, "present_value": -100000, "future_value": 0}
    )
    data_pmt = json.loads(result_pmt.content[0].text)
    assert abs(data_pmt["result"] - 665.30) < 0.01

    # Step 2: Calculate balloon (remaining balance after 60 payments)
    result_balloon = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "fv", "rate": 0.07/12, "periods": 60, "payment": 665.30, "present_value": -100000}
    )
    data_balloon = json.loads(result_balloon.content[0].text)
    assert abs(data_balloon["result"] - 94131.59) < 1.00


@pytest.mark.asyncio
async def test_financial_deferred_annuity_changing_rates(mcp_client):
    """Test deferred annuity with changing interest rates.

    Source: Vaia Corporate Finance Textbook
    Problem: 15-year annuity paying $750/year. First payment in year 6.
    Interest rate: 12% for years 1-5, then 15% thereafter.

    Solution:
    Step 1: PV at year 5 (start of payments - 1) using 15% rate
    PV₅ = 750 × [(1 - 1.15⁻¹⁵) / 0.15]

    Step 2: Discount back to today using 12% rate
    PV₀ = PV₅ / 1.12⁵
    """
    # Step 1: PV at year 5 (one year before first payment) using 15% rate
    result_at_yr5 = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.15, "periods": 15, "payment": -750, "future_value": 0}
    )
    data_at_yr5 = json.loads(result_at_yr5.content[0].text)
    # PV at year 5 should be approximately $4,372.56
    assert 4300 < data_at_yr5["result"] < 4400

    # Step 2: Discount back 5 years at 12%
    result_today = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.12, "periods": 5, "future_value": -data_at_yr5["result"], "payment": 0}
    )
    data_today = json.loads(result_today.content[0].text)
    # Final PV should be approximately $2,481
    assert 2400 < data_today["result"] < 2550


@pytest.mark.asyncio
async def test_financial_mixed_cashflows_cfa(mcp_client):
    """Test mixed cash flows from CFA problem set.

    Source: CFA Level 1 PDF - Problem B
    Problem: Calculate PV of instrument paying $20,000 at end of years 1, 2, 3,
    and $30,000 at end of year 4. Discount rate: 8%

    Solution Method 2 (Annuity + lump sum):
    PV = 20,000 × [(1 - 1.08⁻³) / 0.08] + 30,000 / 1.08⁴
    PV = 51,542.13 + 22,050.70 = $73,592.84
    """
    # Annuity component (3 payments of 20,000 received)
    # Use positive payment since we RECEIVE these cash flows
    result_annuity = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.08, "periods": 3, "payment": 20000, "future_value": 0}
    )
    data_annuity = json.loads(result_annuity.content[0].text)

    # Lump sum component (30,000 at year 4 received)
    result_lumpsum = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.08, "periods": 4, "future_value": 30000, "payment": 0}
    )
    data_lumpsum = json.loads(result_lumpsum.content[0].text)

    # Total PV (take absolute values since we want the instrument's value)
    total_pv = abs(data_annuity["result"]) + abs(data_lumpsum["result"])
    assert abs(total_pv - 73592.83) < 0.01


@pytest.mark.asyncio
async def test_financial_bond_semiannual_cfa(mcp_client):
    """Test semi-annual coupon bond from CFA curriculum.

    Source: AnalystPrep CFA Level 1 - Example 2
    Problem: 2-year bond, face value $1000, 6% annual coupon (paid semi-annually),
    market discount rate 5%. Price closest to?

    Calculator Solution:
    N = 4 (2 years × 2 periods)
    I/Y = 2.5 (5% / 2)
    PMT = 30 (6% × 1000 / 2)
    FV = 1000
    CPT PV = -1,018.81
    """
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.025, "periods": 4, "payment": 30, "future_value": 1000}
    )
    data = json.loads(result.content[0].text)
    assert abs(data["result"] - (-1018.81)) < 0.01


@pytest.mark.asyncio
async def test_financial_bond_at_discount_cfa(mcp_client):
    """Test bond trading at discount from CFA curriculum.

    Source: AnalystPrep CFA Level 1
    Problem: Calculate bond price:
    - Face value: 100
    - Coupon rate: 7.8% paid semi-annually (3.9 per period)
    - Periods: 20 (10 years)
    - Yield: 8.5% annually (4.25% per period)

    Calculator Solution:
    N = 20, I/Y = 4.25, PMT = 3.9, FV = 100
    CPT PV = -95.35
    """
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.0425, "periods": 20, "payment": 3.9, "future_value": 100}
    )
    data = json.loads(result.content[0].text)
    assert abs(data["result"] - (-95.35)) < 0.01


@pytest.mark.asyncio
async def test_financial_zero_coupon_fv_cfa(mcp_client):
    """Test zero-coupon FV calculation from CFA curriculum.

    Source: AnalystPrep CFA Level 1 - Example 1
    Problem: What is the future value of $8,000 invested for 4 years at 8% interest?

    Calculator Solution:
    N = 4, I/Y = 8, PV = -8,000, PMT = 0
    CPT FV = 10,883.91
    """
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "fv", "rate": 0.08, "periods": 4, "present_value": -8000, "payment": 0}
    )
    data = json.loads(result.content[0].text)
    assert abs(data["result"] - 10883.91) < 0.01


@pytest.mark.asyncio
async def test_financial_ordinary_annuity_cfa_validation(mcp_client):
    """Test validation: Given PV from CFA curriculum.

    Source: CFA Level 1 PDF - Given Fact
    Statement: "At 5% interest rate per year compounded annually, the Present Value
    of a 10-year ordinary annuity with annual payments of $2000 is $15443.47."

    This is a validation test to verify our calculations match CFA standards.
    """
    result = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.05, "periods": 10, "payment": -2000, "future_value": 0, "when": "end"}
    )
    data = json.loads(result.content[0].text)
    assert abs(data["result"] - 15443.47) < 0.01


# ============================================================================
# Perpetuity and Growing Annuity Tests (New Tools/Features)
# ============================================================================


@pytest.mark.asyncio
async def test_financial_perpetuity_monthly_cfa(mcp_client):
    """Test monthly perpetuity from CFA curriculum.

    Source: Soleadea CFA Level 1 - TVM Practical Problems
    Problem: How much is a promise of receiving perpetually a sum of USD 1,000
    each month worth now if the stated annual interest rate is 6%?

    Solution:
    Monthly rate: 6% / 12 = 0.5% = 0.005
    PV = Payment / rate = 1,000 / 0.005 = $200,000
    """
    result = await mcp_client.call_tool(
        "perpetuity",
        {"payment": 1000, "rate": 0.005}
    )
    data = json.loads(result.content[0].text)
    assert abs(data["result"] - 200000.00) < 0.01
    # Verify metadata
    assert data["type"] == "level_ordinary"
    assert data["payment"] == 1000
    assert data["rate"] == 0.005


@pytest.mark.asyncio
async def test_financial_perpetuity_quarterly(mcp_client):
    """Test quarterly perpetuity from corporate finance textbook.

    Source: Vaia Corporate Finance Textbook
    Problem: A prestigious investment bank designed a new security that pays a
    quarterly dividend of $5 in perpetuity. The first dividend occurs one quarter
    from today. What is the price of the security if the stated annual interest
    rate is 7 percent, compounded quarterly?

    Solution:
    Quarterly rate: 7% / 4 = 1.75% = 0.0175
    PV = 5 / 0.0175 = $285.71
    """
    result = await mcp_client.call_tool(
        "perpetuity",
        {"payment": 5, "rate": 0.0175}
    )
    data = json.loads(result.content[0].text)
    assert abs(data["result"] - 285.71) < 0.01
    # Verify metadata
    assert data["type"] == "level_ordinary"


@pytest.mark.asyncio
async def test_financial_growing_annuity_salary(mcp_client):
    """Test growing annuity using extended math_financial_calcs tool.

    Source: Vaia Corporate Finance - Problem 53 (Tom Adams)
    Problem: Tom Adams has received a job offer:
    - Base salary: $45,000
    - Initial bonus: $10,000 (paid immediately)
    - Salary growth rate: 3.5% per year
    - Annual bonus: 10% of salary
    - Working years: 25
    - Discount rate: 12%

    What is the present value of Tom's job offer?

    Solution:
    1. PV of salary: $45,000 growing at 3.5% for 25 years at 12%
    2. PV of bonus: 10% of growing salary stream
    3. Initial bonus: $10,000 (no discounting)
    """
    # Salary component
    result_salary = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.12, "periods": 25, "payment": -45000, "growth_rate": 0.035}
    )
    data_salary = json.loads(result_salary.content[0].text)

    # Bonus component (10% of salary)
    result_bonus = await mcp_client.call_tool(
        "financial_calcs",
        {"calculation": "pv", "rate": 0.12, "periods": 25, "payment": -4500, "growth_rate": 0.035}
    )
    data_bonus = json.loads(result_bonus.content[0].text)

    # Total PV
    total_pv = data_salary["result"] + data_bonus["result"] + 10000

    # Expected: Salary component ≈ $455,816 (verified calculation)
    # Formula: PV = C/(r-g) × [1 - (1+g)^n/(1+r)^n]
    # PV = 45000/(0.12-0.035) × [1 - 1.035^25/1.12^25] = $455,816
    assert 454000 < data_salary["result"] < 457000
    # Bonus component (10% of salary)
    assert 45400 < data_bonus["result"] < 45700
    # Total: salary + bonus + $10k initial = ~$511,000
    assert 509000 < total_pv < 513000

    # Verify growth_rate in metadata
    assert data_salary["growth_rate"] == 0.035
    assert data_bonus["growth_rate"] == 0.035
