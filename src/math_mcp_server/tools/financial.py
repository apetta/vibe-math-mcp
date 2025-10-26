"""Financial mathematics tools."""

import json
import math
from typing import List, Literal, Optional, Union, cast
from mcp.types import ToolAnnotations

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
            result = 0.0  # Initialize to accumulate components

            # Calculate PV of lump sum if provided
            if future_value is not None and periods is not None:
                # Lump sum: PV = FV / (1 + r)^n
                if rate == 0:
                    result += -future_value
                else:
                    result += -future_value / ((1 + rate) ** periods)

            # Calculate PV of annuity if provided
            if payment is not None and periods is not None:
                # Annuity: PV = -PMT × [(1 - (1+r)^-n) / r]
                if rate == 0:
                    result += -payment * periods
                else:
                    result += -payment * ((1 - (1 + rate) ** -periods) / rate)

            # Ensure at least one component was provided
            if future_value is None and payment is None:
                raise ValueError(
                    "PV: provide rate + periods + (future_value AND/OR payment)"
                )

        elif calculation == "rate":
            # Solve for interest rate
            if periods is None or present_value is None:
                raise ValueError("Rate calculation requires periods and present_value")

            # Scenario 1: Lump sum only (no payment or payment=0)
            if payment is None or payment == 0:
                if future_value is None or future_value == 0:
                    raise ValueError("Rate calculation requires future_value for lump sum")
                if present_value == 0:
                    raise ValueError("Cannot solve for rate with PV=0")
                # Analytical solution: r = (FV/-PV)^(1/n) - 1
                # Signs must be opposite for positive rate
                result = (future_value / -present_value) ** (1 / periods) - 1

            # Scenario 2 & 3: Annuity or combined (use Newton-Raphson)
            else:
                # Newton-Raphson method to find rate
                guess = 0.1  # Initial guess: 10%
                max_iter = 100
                tolerance = 1e-8

                for iteration in range(max_iter):
                    # Calculate PV at current rate guess
                    pv_calc = 0.0

                    # Add FV component if provided
                    if future_value is not None and future_value != 0:
                        pv_calc += -future_value / ((1 + guess) ** periods)

                    # Add PMT component
                    if guess == 0:
                        pv_calc += -payment * periods
                    else:
                        pv_calc += -payment * ((1 - (1 + guess) ** -periods) / guess)

                    # Error = calculated PV - actual PV
                    error = pv_calc - present_value

                    if abs(error) < tolerance:
                        break

                    # Calculate derivative for Newton-Raphson
                    # d(PV)/d(r) for FV component: n*FV*(1+r)^(-n-1)
                    # d(PV)/d(r) for PMT: complex derivative of annuity formula
                    derivative = 0.0

                    if future_value is not None and future_value != 0:
                        derivative += periods * future_value * ((1 + guess) ** (-periods - 1))

                    if guess == 0:
                        # At r=0, derivative is approximately -PMT*periods^2/2
                        derivative += -payment * periods * periods / 2
                    else:
                        # Derivative of annuity PV formula
                        term1 = ((1 + guess) ** -periods) / guess
                        term2 = ((1 - (1 + guess) ** -periods) / (guess ** 2))
                        term3 = periods * ((1 + guess) ** (-periods - 1)) / guess
                        derivative += -payment * (term1 - term2 - term3)

                    if abs(derivative) < 1e-10:
                        raise ValueError("Rate calculation failed: derivative too small")

                    guess = guess - error / derivative

                    # Prevent negative rates below -99%
                    if guess < -0.99:
                        guess = -0.99

                else:
                    raise ValueError("Rate calculation did not converge after 100 iterations")

                result = guess

        elif calculation == "fv":
            # Future Value
            if rate is None:
                raise ValueError("FV calculation requires rate")
            if payment is None or periods is None:
                raise ValueError("FV calculation requires rate, periods, and payment")
            if present_value is None:
                present_value = 0
            if rate == 0:
                result = -present_value - payment * periods
            else:
                result = -present_value * (1 + rate) ** periods - payment * (
                    ((1 + rate) ** periods - 1) / rate
                )

        elif calculation == "pmt":
            # Payment
            if rate is None:
                raise ValueError("PMT calculation requires rate")
            if present_value is None or periods is None:
                raise ValueError("PMT calculation requires rate, periods, and present_value")
            if rate == 0:
                result = -present_value / periods
            else:
                result = (
                    -present_value * (rate * (1 + rate) ** periods) / ((1 + rate) ** periods - 1)
                )

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
                npv_derivative = sum(
                    -i * cf / (1 + guess) ** (i + 1) for i, cf in enumerate(cash_flows)
                )

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
            if rate is None:
                raise ValueError("NPV calculation requires rate")
            if cash_flows is None:
                raise ValueError("NPV calculation requires cash_flows and rate")
            result = sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))

        else:
            raise ValueError(f"Unknown calculation type: {calculation}")

        return format_result(float(result), {"calculation": calculation, "rate": rate})
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
