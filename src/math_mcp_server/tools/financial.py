"""Financial mathematics tools."""

import math
from typing import List, Literal, Optional
from mcp.types import ToolAnnotations
import numpy as np

from ..server import mcp
from ..core import format_result


@mcp.tool(
    name="math_financial_calcs",
    description="Financial calculations: present value (PV), future value (FV), payment (PMT), IRR, NPV.",
    annotations=ToolAnnotations(
        title="Financial Calculations",
        readOnlyHint=True,
        idempotentHint=True,
    )
)
async def financial_calcs(
    calculation: Literal["pv", "fv", "pmt", "irr", "npv"],
    rate: float,
    periods: Optional[int] = None,
    payment: Optional[float] = None,
    present_value: Optional[float] = None,
    future_value: Optional[float] = None,
    cash_flows: Optional[List[float]] = None
) -> str:
    """
    Perform time value of money calculations.

    Examples:
        - calculation="fv", rate=0.05, periods=10, payment=-100 → Future value of annuity
        - calculation="pv", rate=0.05, periods=10, payment=-100 → Present value of annuity
        - calculation="irr", cash_flows=[-1000, 300, 400, 500] → Internal rate of return
        - calculation="npv", rate=0.1, cash_flows=[-1000, 300, 400, 500] → Net present value

    Args:
        calculation: Type of financial calculation
        rate: Interest rate per period (e.g., 0.05 for 5%)
        periods: Number of periods
        payment: Payment amount per period (negative for outflows)
        present_value: Present value (negative for outflows)
        future_value: Future value
        cash_flows: Series of cash flows for IRR/NPV calculations

    Returns:
        JSON with calculated financial result
    """
    try:
        if calculation == "pv":
            # Present Value
            if payment is None or periods is None:
                raise ValueError("PV calculation requires rate, periods, and payment")
            if rate == 0:
                result = -payment * periods
            else:
                result = payment * ((1 - (1 + rate) ** -periods) / rate)

        elif calculation == "fv":
            # Future Value
            if payment is None or periods is None:
                raise ValueError("FV calculation requires rate, periods, and payment")
            if present_value is None:
                present_value = 0
            if rate == 0:
                result = -present_value - payment * periods
            else:
                result = -present_value * (1 + rate) ** periods - payment * (((1 + rate) ** periods - 1) / rate)

        elif calculation == "pmt":
            # Payment
            if present_value is None or periods is None:
                raise ValueError("PMT calculation requires rate, periods, and present_value")
            if rate == 0:
                result = -present_value / periods
            else:
                result = -present_value * (rate * (1 + rate) ** periods) / ((1 + rate) ** periods - 1)

        elif calculation == "irr":
            # Internal Rate of Return
            if cash_flows is None or len(cash_flows) < 2:
                raise ValueError("IRR calculation requires cash_flows with at least 2 values")

            # Newton-Raphson method to find IRR
            guess = 0.1
            max_iter = 100
            tolerance = 1e-6

            for _ in range(max_iter):
                npv_val = sum(cf / (1 + guess) ** i for i, cf in enumerate(cash_flows))
                npv_derivative = sum(-i * cf / (1 + guess) ** (i + 1) for i, cf in enumerate(cash_flows))

                if abs(npv_val) < tolerance:
                    break

                if npv_derivative == 0:
                    raise ValueError("IRR calculation failed: derivative is zero")

                guess = guess - npv_val / npv_derivative
            else:
                raise ValueError("IRR calculation did not converge")

            result = guess

        elif calculation == "npv":
            # Net Present Value
            if cash_flows is None:
                raise ValueError("NPV calculation requires cash_flows and rate")
            result = sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))

        else:
            raise ValueError(f"Unknown calculation type: {calculation}")

        return format_result(
            float(result),
            {"calculation": calculation, "rate": rate}
        )
    except Exception as e:
        raise ValueError(f"Financial calculation failed: {str(e)}")


@mcp.tool(
    name="math_compound_interest",
    description="Calculate compound interest with various compounding frequencies.",
    annotations=ToolAnnotations(
        title="Compound Interest",
        readOnlyHint=True,
        idempotentHint=True,
    )
)
async def compound_interest(
    principal: float,
    rate: float,
    time: float,
    frequency: Literal["annual", "semi-annual", "quarterly", "monthly", "daily", "continuous"] = "annual"
) -> str:
    """
    Calculate compound interest with different compounding frequencies.

    Formula: A = P(1 + r/n)^(nt) for discrete compounding
             A = Pe^(rt) for continuous compounding

    Examples:
        - principal=1000, rate=0.05, time=10, frequency="annual" → 1628.89
        - principal=1000, rate=0.05, time=10, frequency="monthly" → 1647.01
        - principal=1000, rate=0.05, time=10, frequency="continuous" → 1648.72

    Args:
        principal: Initial amount
        rate: Annual interest rate (e.g., 0.05 for 5%)
        time: Time period in years
        frequency: Compounding frequency

    Returns:
        JSON with final amount and interest earned
    """
    try:
        freq_map = {
            "annual": 1,
            "semi-annual": 2,
            "quarterly": 4,
            "monthly": 12,
            "daily": 365,
        }

        if frequency == "continuous":
            # Continuous compounding: A = Pe^(rt)
            final_amount = principal * math.exp(rate * time)
        else:
            # Discrete compounding: A = P(1 + r/n)^(nt)
            n = freq_map[frequency]
            final_amount = principal * (1 + rate / n) ** (n * time)

        interest_earned = final_amount - principal

        return format_result(
            float(final_amount),
            {
                "principal": principal,
                "rate": rate,
                "time": time,
                "frequency": frequency,
                "interest_earned": float(interest_earned),
            }
        )
    except Exception as e:
        raise ValueError(f"Compound interest calculation failed: {str(e)}")
