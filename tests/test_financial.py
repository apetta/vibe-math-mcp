"""Tests for financial mathematics tools."""

import json
import pytest


@pytest.mark.asyncio
async def test_financial_fv(mcp_client):
    """Test future value calculation."""
    result = await mcp_client.call_tool(
        "math_financial_calcs",
        {"calculation": "fv", "rate": 0.05, "periods": 10, "payment": -100}
    )
    data = json.loads(result.content[0].text)
    # FV of annuity: PMT * ((1+r)^n - 1) / r
    expected = 100 * (((1.05)**10 - 1) / 0.05)
    assert abs(data["result"] - expected) < 1


@pytest.mark.asyncio
async def test_financial_pv(mcp_client):
    """Test present value calculation."""
    result = await mcp_client.call_tool(
        "math_financial_calcs",
        {"calculation": "pv", "rate": 0.05, "periods": 10, "payment": -100}
    )
    data = json.loads(result.content[0].text)
    # PV calculation returns negative value representing outflow
    assert "result" in data


@pytest.mark.asyncio
async def test_financial_npv(mcp_client):
    """Test net present value calculation."""
    cash_flows = [-1000, 300, 400, 500, 200]
    result = await mcp_client.call_tool(
        "math_financial_calcs",
        {"calculation": "npv", "rate": 0.1, "cash_flows": cash_flows}
    )
    data = json.loads(result.content[0].text)
    assert "result" in data


@pytest.mark.asyncio
async def test_compound_interest_annual(mcp_client):
    """Test compound interest with annual compounding."""
    result = await mcp_client.call_tool(
        "math_compound_interest",
        {"principal": 1000, "rate": 0.05, "time": 10, "frequency": "annual"}
    )
    data = json.loads(result.content[0].text)
    expected = 1000 * (1.05 ** 10)
    assert abs(data["result"] - expected) < 0.01


@pytest.mark.asyncio
async def test_compound_interest_continuous(mcp_client):
    """Test compound interest with continuous compounding."""
    result = await mcp_client.call_tool(
        "math_compound_interest",
        {"principal": 1000, "rate": 0.05, "time": 10, "frequency": "continuous"}
    )
    data = json.loads(result.content[0].text)
    # Continuous: A = Pe^(rt)
    import math
    expected = 1000 * math.exp(0.05 * 10)
    assert abs(data["result"] - expected) < 0.01
