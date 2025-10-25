"""Tests for financial mathematics tools."""

import json
import pytest


@pytest.mark.asyncio
async def test_financial_fv(mcp_client):
    """Test future value calculation."""
    result = await mcp_client.call_tool(
        "math_financial_calcs", {"calculation": "fv", "rate": 0.05, "periods": 10, "payment": -100}
    )
    data = json.loads(result.content[0].text)
    # FV of annuity: PMT * ((1+r)^n - 1) / r
    expected = 100 * (((1.05) ** 10 - 1) / 0.05)
    assert abs(data["result"] - expected) < 1


@pytest.mark.asyncio
async def test_financial_pv(mcp_client):
    """Test present value calculation."""
    result = await mcp_client.call_tool(
        "math_financial_calcs", {"calculation": "pv", "rate": 0.05, "periods": 10, "payment": -100}
    )
    data = json.loads(result.content[0].text)
    # PV calculation returns negative value representing outflow
    assert "result" in data


@pytest.mark.asyncio
async def test_financial_npv(mcp_client):
    """Test net present value calculation."""
    cash_flows = [-1000, 300, 400, 500, 200]
    result = await mcp_client.call_tool(
        "math_financial_calcs", {"calculation": "npv", "rate": 0.1, "cash_flows": cash_flows}
    )
    data = json.loads(result.content[0].text)
    assert "result" in data


@pytest.mark.asyncio
async def test_compound_interest_annual(mcp_client):
    """Test compound interest with annual compounding."""
    result = await mcp_client.call_tool(
        "math_compound_interest",
        {"principal": 1000, "rate": 0.05, "time": 10, "frequency": "annual"},
    )
    data = json.loads(result.content[0].text)
    expected = 1000 * (1.05**10)
    assert abs(data["result"] - expected) < 0.01


@pytest.mark.asyncio
async def test_compound_interest_continuous(mcp_client):
    """Test compound interest with continuous compounding."""
    result = await mcp_client.call_tool(
        "math_compound_interest",
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
        "math_financial_calcs",
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
        "math_financial_calcs",
        {"calculation": "pmt", "rate": 0.0, "periods": 10, "present_value": -1000},
    )
    data = json.loads(result.content[0].text)
    # PMT = 1000 / 10 = 100
    assert abs(data["result"] - 100.0) < 0.01


@pytest.mark.asyncio
async def test_financial_pmt_missing_params(mcp_client):
    """Test error when required parameters are missing for PMT."""
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool("math_financial_calcs", {"calculation": "pmt", "rate": 0.05})
    assert "requires" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_financial_irr_simple_investment(mcp_client):
    """Test IRR for a simple investment."""
    # Initial investment of -$1000, returns of $500, $400, $300, $200
    cash_flows = [-1000, 500, 400, 300, 200]
    result = await mcp_client.call_tool(
        "math_financial_calcs", {"calculation": "irr", "rate": 0.1, "cash_flows": cash_flows}
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
        "math_financial_calcs", {"calculation": "irr", "rate": 0.1, "cash_flows": cash_flows}
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
        "math_financial_calcs", {"calculation": "irr", "rate": 0.1, "cash_flows": cash_flows}
    )
    data = json.loads(result.content[0].text)
    # IRR should be negative for unprofitable investment
    assert data["result"] < 0


@pytest.mark.asyncio
async def test_financial_irr_insufficient_cash_flows(mcp_client):
    """Test error when IRR has insufficient cash flows."""
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool(
            "math_financial_calcs", {"calculation": "irr", "rate": 0.1, "cash_flows": [-1000]}
        )
    assert "at least 2" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_financial_irr_convergence(mcp_client):
    """Test IRR convergence with standard cash flows."""
    # Well-behaved cash flows that should converge quickly
    cash_flows = [-10000, 3000, 3000, 3000, 3000, 3000]
    result = await mcp_client.call_tool(
        "math_financial_calcs", {"calculation": "irr", "rate": 0.1, "cash_flows": cash_flows}
    )
    data = json.loads(result.content[0].text)
    # Should converge to a reasonable rate
    assert -0.5 < data["result"] < 0.5


@pytest.mark.asyncio
async def test_financial_fv_zero_rate(mcp_client):
    """Test future value calculation with zero interest rate."""
    # With 0% rate, FV = PV + (PMT × periods)
    result = await mcp_client.call_tool(
        "math_financial_calcs", {"calculation": "fv", "rate": 0.0, "periods": 10, "payment": -100}
    )
    data = json.loads(result.content[0].text)
    # FV = 100 × 10 = 1000
    assert abs(data["result"] - 1000.0) < 0.01


@pytest.mark.asyncio
async def test_financial_pv_zero_rate(mcp_client):
    """Test present value calculation with zero interest rate."""
    # With 0% rate, PV = PMT × periods
    result = await mcp_client.call_tool(
        "math_financial_calcs", {"calculation": "pv", "rate": 0.0, "periods": 10, "payment": -100}
    )
    data = json.loads(result.content[0].text)
    # PV = 100 × 10 = 1000
    assert abs(data["result"] - 1000.0) < 0.01


@pytest.mark.asyncio
async def test_financial_npv_missing_cash_flows(mcp_client):
    """Test error when NPV is calculated without cash flows."""
    with pytest.raises(Exception) as exc_info:
        await mcp_client.call_tool("math_financial_calcs", {"calculation": "npv", "rate": 0.1})
    assert "requires" in str(exc_info.value).lower() or "cash_flows" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_compound_interest_semi_annual(mcp_client):
    """Test compound interest with semi-annual compounding."""
    result = await mcp_client.call_tool(
        "math_compound_interest",
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
        "math_compound_interest",
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
        "math_compound_interest",
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
        "math_compound_interest",
        {"principal": 2000, "rate": 0.04, "time": 1, "frequency": "daily"},
    )
    data = json.loads(result.content[0].text)
    # Daily: n=365, A = 2000 * (1 + 0.04/365)^(365*1)
    expected = 2000 * (1 + 0.04 / 365) ** (365 * 1)
    assert abs(data["result"] - expected) < 0.01
