"""Financial mathematics tools."""

import json
import math
from typing import Any, Dict, List, Literal, Optional, Union, cast
from mcp.types import ToolAnnotations
import numpy_financial as npf

from ..server import mcp
from ..core import format_result


@mcp.tool(
    name="math_financial_calcs",
    description="Financial calculations: present value (PV), future value (FV), payment (PMT), IRR, NPV.",
    annotations=ToolAnnotations(
        title="Financial Calculations",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def financial_calcs(
    calculation: Literal["pv", "fv", "pmt", "rate", "irr", "npv"],
    rate: Optional[float] = None,
    periods: Optional[int] = None,
    payment: Optional[float] = None,
    present_value: Optional[float] = None,
    future_value: Optional[float] = None,
    cash_flows: Union[str, List[float], None] = None,
    when: Literal["end", "begin"] = "end",
) -> str:
    """
    Time Value of Money (TVM) calculations.

    The TVM equation has 5 variables - know 4, solve for the 5th:
        PV  = Present Value (lump sum now)
        FV  = Future Value (lump sum at maturity)
        PMT = Payment (regular periodic cash flow)
        N   = Number of periods
        I/Y = Interest rate per period

    Parameters:
        calculation: What to solve for ("pv", "fv", "pmt", "irr", "npv")
        rate: Interest/discount rate per period (e.g., 0.05 = 5% annually)
        periods: Number of compounding periods (n)
        payment: Regular periodic cash flow - happens EVERY period
                 Examples: £30 bond coupon, £500 monthly contribution
        present_value: Single lump sum at time 0
                       Examples: Initial investment, loan principal
        future_value: Single lump sum at maturity
                      Examples: £1000 bond face value, savings goal
        cash_flows: Series of cash flows for IRR/NPV calculations
        when: Payment timing - 'end' (default) for ordinary annuity (payments at period end),
              'begin' for annuity due (payments at period start)
              Examples: Mortgages use 'end', leases/rent typically use 'begin'

    Sign convention:
        Negative = cash out (you pay)
        Positive = cash in (you receive)

    Common patterns:

        ZERO-COUPON BOND: PV of £1000 in 10 years at 5%
        ├─ Solving for: PV
        ├─ Given: FV (£1000), rate (5%), periods (10)
        └─ Call: calculation="pv", rate=0.05, periods=10, future_value=1000
           Result: £613.91

        COUPON BOND: PV of £30 annual coupons + £1000 face value at 5% yield
        ├─ Solving for: PV
        ├─ Given: PMT (£30), FV (£1000), rate (5%), periods (10)
        └─ Call: calculation="pv", rate=0.05, periods=10,
                 payment=30, future_value=1000
           Result: £845.57

        RETIREMENT SAVINGS: How much will I have with £500/month for 30 years at 7%?
        ├─ Solving for: FV
        ├─ Given: PMT (-£500), PV (0), rate (7%/12), periods (360)
        └─ Call: calculation="fv", rate=0.07/12, periods=360,
                 payment=-500, present_value=0
           Result: £566,764

        MORTGAGE PAYMENT: Monthly payment on £200k loan, 30 years, 4% APR
        ├─ Solving for: PMT
        ├─ Given: PV (-£200k), FV (0), rate (4%/12), periods (360)
        └─ Call: calculation="pmt", rate=0.04/12, periods=360,
                 present_value=-200000, future_value=0
           Result: £954.83

        INTEREST RATE: What rate on £613.81 grows to £1000 in 10 years?
        ├─ Solving for: I/Y (rate)
        ├─ Given: PV (-£613.81), FV (£1000), periods (10)
        └─ Call: calculation="rate", periods=10,
                 present_value=-613.81, future_value=1000
           Result: 0.05 (5%)

    Returns:
        JSON with result and calculation metadata
    """
    try:
        # Parse stringified JSON from XML serialization
        if isinstance(cash_flows, str):
            cash_flows = cast(List[float], json.loads(cash_flows))

        if calculation == "pv":
            # Present Value: solve for PV given FV and/or PMT
            if rate is None:
                raise ValueError("PV calculation requires rate")
            if periods is None:
                raise ValueError("PV calculation requires periods")

            # Ensure at least one component was provided
            if future_value is None and payment is None:
                raise ValueError(
                    "PV: provide rate + periods + (future_value AND/OR payment)"
                )

            # Use numpy-financial for battle-tested calculation
            result = npf.pv(
                rate, periods,
                float(payment) if payment is not None else 0.0,
                float(future_value) if future_value is not None else 0.0,  # type: ignore[arg-type]
                when=when
            )

        elif calculation == "rate":
            # Solve for interest rate
            if periods is None or present_value is None:
                raise ValueError("Rate calculation requires periods and present_value")

            # Ensure we have either future_value or payment
            if (future_value is None or future_value == 0) and (payment is None or payment == 0):
                raise ValueError("Rate calculation requires either future_value or payment")

            # Use numpy-financial for battle-tested calculation
            # Note: numpy-financial handles PV=0 correctly for annuity scenarios
            result = npf.rate(
                periods,
                float(payment) if payment is not None else 0.0,
                present_value,
                float(future_value) if future_value is not None else 0.0,
                when=when
            )

        elif calculation == "fv":
            # Future Value
            if rate is None:
                raise ValueError("FV calculation requires rate")
            if payment is None or periods is None:
                raise ValueError("FV calculation requires rate, periods, and payment")

            # Use numpy-financial for battle-tested calculation
            result = npf.fv(
                rate, periods, payment,
                float(present_value) if present_value is not None else 0.0,
                when=when
            )

        elif calculation == "pmt":
            # Payment
            if rate is None:
                raise ValueError("PMT calculation requires rate")
            if present_value is None or periods is None:
                raise ValueError("PMT calculation requires rate, periods, and present_value")

            # Use numpy-financial for battle-tested calculation
            result = npf.pmt(
                rate, periods, present_value,
                float(future_value) if future_value is not None else 0.0,  # type: ignore[arg-type]
                when=when
            )

        elif calculation == "irr":
            # Internal Rate of Return
            if cash_flows is None or len(cash_flows) < 2:
                raise ValueError("IRR calculation requires cash_flows with at least 2 values")

            # Use numpy-financial for battle-tested calculation
            result = npf.irr(cash_flows)

        elif calculation == "npv":
            # Net Present Value
            if rate is None:
                raise ValueError("NPV calculation requires rate")
            if cash_flows is None:
                raise ValueError("NPV calculation requires cash_flows and rate")

            # Use numpy-financial for battle-tested calculation
            # Note: npf.npv treats first value as t=0 (present), matching our convention
            result = npf.npv(rate, cash_flows)

        else:
            raise ValueError(f"Unknown calculation type: {calculation}")

        # Build metadata with all provided parameters for better context management
        metadata: Dict[str, Any] = {"calculation": calculation}
        if rate is not None:
            metadata["rate"] = rate
        if periods is not None:
            metadata["periods"] = periods
        if payment is not None:
            metadata["payment"] = payment
        if present_value is not None:
            metadata["present_value"] = present_value
        if future_value is not None:
            metadata["future_value"] = future_value
        if cash_flows is not None:
            metadata["cash_flows"] = cash_flows
        if when != "end":
            metadata["when"] = when

        return format_result(float(result), metadata)
    except Exception as e:
        raise ValueError(f"Financial calculation failed: {str(e)}")


@mcp.tool(
    name="math_compound_interest",
    description="Calculate compound interest with various compounding frequencies.",
    annotations=ToolAnnotations(
        title="Compound Interest",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def compound_interest(
    principal: float,
    rate: float,
    time: float,
    frequency: Literal[
        "annual", "semi-annual", "quarterly", "monthly", "daily", "continuous"
    ] = "annual",
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
            },
        )
    except Exception as e:
        raise ValueError(f"Compound interest calculation failed: {str(e)}")
