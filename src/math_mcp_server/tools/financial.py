"""Financial mathematics tools."""

import json
import math
from typing import Annotated, Any, Dict, List, Literal, Union, cast
from pydantic import Field
from mcp.types import ToolAnnotations
import numpy_financial as npf

from ..server import mcp
from ..core import format_result


@mcp.tool(
    name="financial_calcs",
    description="""Time Value of Money (TVM) calculations: solve for PV, FV, PMT, rate, IRR, or NPV.

The TVM equation has 5 variables - know 4, solve for the 5th:
    PV  = Present Value (lump sum now)
    FV  = Future Value (lump sum at maturity)
    PMT = Payment (regular periodic cash flow)
    N   = Number of periods
    I/Y = Interest rate per period

Sign convention: negative = cash out (you pay), positive = cash in (you receive)

Examples:

ZERO-COUPON BOND: PV of £1000 in 10 years at 5%
    calculation="pv", rate=0.05, periods=10, future_value=1000
    Result: £613.91

COUPON BOND: PV of £30 annual coupons + £1000 face value at 5% yield
    calculation="pv", rate=0.05, periods=10, payment=30, future_value=1000
    Result: £845.57

RETIREMENT SAVINGS: FV with £500/month for 30 years at 7%
    calculation="fv", rate=0.07/12, periods=360, payment=-500, present_value=0
    Result: £566,764

MORTGAGE PAYMENT: Monthly payment on £200k loan, 30 years, 4% APR
    calculation="pmt", rate=0.04/12, periods=360, present_value=-200000, future_value=0
    Result: £954.83

INTEREST RATE: What rate grows £613.81 to £1000 in 10 years?
    calculation="rate", periods=10, present_value=-613.81, future_value=1000
    Result: 0.05 (5%)

GROWING ANNUITY: Salary stream with 3.5% raises, discounted at 12%
    calculation="pv", rate=0.12, periods=25, payment=-45000, growth_rate=0.035
    Result: £402,586""",
    annotations=ToolAnnotations(
        title="Financial Calculations",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def financial_calcs(
    calculation: Annotated[
        Literal["pv", "fv", "pmt", "rate", "irr", "npv"],
        Field(description="What to solve for: pv, fv, pmt, rate, irr, or npv"),
    ],
    rate: Annotated[
        float | None,
        Field(description="Interest/discount rate per period (e.g., 0.05 for 5% annual)"),
    ] = None,
    periods: Annotated[int | None, Field(description="Number of compounding periods", ge=1)] = None,
    payment: Annotated[
        float | None,
        Field(description="Regular periodic payment (negative=pay out, positive=receive)"),
    ] = None,
    present_value: Annotated[
        float | None,
        Field(description="Single lump sum at time 0 (negative=pay, positive=receive)"),
    ] = None,
    future_value: Annotated[
        float | None,
        Field(description="Single lump sum at maturity (negative=owe, positive=receive)"),
    ] = None,
    cash_flows: Annotated[
        Union[str, List[float], None],
        Field(description="Series of cash flows for IRR/NPV (e.g., [-100, 30, 30, 130])"),
    ] = None,
    when: Annotated[
        Literal["end", "begin"],
        Field(description="Payment timing: 'end' (ordinary) or 'begin' (annuity due)"),
    ] = "end",
    growth_rate: Annotated[
        float, Field(description="Payment growth rate per period (0.0 for level annuity)", ge=0)
    ] = 0.0,
) -> str:
    """Time Value of Money calculations."""
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
                raise ValueError("PV: provide rate + periods + (future_value AND/OR payment)")

            # Handle growing annuity case
            if growth_rate != 0.0 and payment is not None and payment != 0:
                # Validate growth rate
                if growth_rate < 0:
                    raise ValueError("Growth rate cannot be negative")

                # Calculate PV of growing annuity (formula works with positive values)
                payment_abs = abs(payment)
                if abs(rate - growth_rate) < 1e-10:
                    # Special case: rate == growth_rate
                    pv_annuity = payment_abs * periods / (1 + rate)
                else:
                    # Standard growing annuity formula
                    growth_factor = (1 + growth_rate) / (1 + rate)
                    pv_annuity = payment_abs * (1 - growth_factor**periods) / (rate - growth_rate)

                # Adjust for annuity due
                if when == "begin":
                    pv_annuity *= 1 + rate

                # Apply sign: payment < 0 (pay out) → PV > 0 (value received)
                pv_annuity = pv_annuity if payment < 0 else -pv_annuity

                # Add PV of lump sum if present
                if future_value is not None and future_value != 0:
                    pv_lumpsum = abs(future_value) / ((1 + rate) ** periods)
                    # future_value > 0 (receive) → PV < 0 (cost)
                    pv_lumpsum = -pv_lumpsum if future_value > 0 else pv_lumpsum
                    pv_annuity += pv_lumpsum

                result = pv_annuity
            else:
                # Use numpy-financial for standard (non-growing) calculation
                result = npf.pv(
                    rate,
                    periods,
                    float(payment) if payment is not None else 0.0,
                    float(future_value) if future_value is not None else 0.0,  # type: ignore[arg-type]
                    when=when,
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
                when=when,
            )

        elif calculation == "fv":
            # Future Value
            if rate is None:
                raise ValueError("FV calculation requires rate")
            if payment is None or periods is None:
                raise ValueError("FV calculation requires rate, periods, and payment")

            # Handle growing annuity case
            if growth_rate != 0.0 and payment != 0:
                # Validate growth rate
                if growth_rate < 0:
                    raise ValueError("Growth rate cannot be negative")

                # Calculate FV of growing annuity (formula works with positive values)
                payment_abs = abs(payment)
                if abs(rate - growth_rate) < 1e-10:
                    # Special case: rate == growth_rate
                    fv_annuity = payment_abs * periods * ((1 + rate) ** (periods - 1))
                else:
                    # Standard growing annuity FV formula
                    fv_annuity = payment_abs * (
                        ((1 + rate) ** periods - (1 + growth_rate) ** periods)
                        / (rate - growth_rate)
                    )

                # Adjust for annuity due
                if when == "begin":
                    fv_annuity *= 1 + rate

                # Apply sign: payment < 0 (pay) → FV > 0 (accumulate)
                fv_annuity = fv_annuity if payment < 0 else -fv_annuity

                # Add FV of present value if present
                if present_value is not None and present_value != 0:
                    fv_pv = present_value * ((1 + rate) ** periods)
                    fv_annuity += fv_pv

                result = fv_annuity
            else:
                # Use numpy-financial for standard (non-growing) calculation
                result = npf.fv(
                    rate,
                    periods,
                    payment,
                    float(present_value) if present_value is not None else 0.0,
                    when=when,
                )

        elif calculation == "pmt":
            # Payment
            if rate is None:
                raise ValueError("PMT calculation requires rate")
            if present_value is None or periods is None:
                raise ValueError("PMT calculation requires rate, periods, and present_value")

            # Use numpy-financial for battle-tested calculation
            result = npf.pmt(
                rate,
                periods,
                present_value,
                float(future_value) if future_value is not None else 0.0,  # type: ignore[arg-type]
                when=when,
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
        if growth_rate != 0.0:
            metadata["growth_rate"] = growth_rate

        return format_result(float(result), metadata)
    except Exception as e:
        raise ValueError(f"Financial calculation failed: {str(e)}")


@mcp.tool(
    name="compound_interest",
    description="""Calculate compound interest with various compounding frequencies.

Formulas:
    Discrete: A = P(1 + r/n)^(nt)
    Continuous: A = Pe^(rt)

Examples:

ANNUAL COMPOUNDING: £1000 at 5% for 10 years
    principal=1000, rate=0.05, time=10, frequency="annual"
    Result: £1628.89

MONTHLY COMPOUNDING: £1000 at 5% for 10 years
    principal=1000, rate=0.05, time=10, frequency="monthly"
    Result: £1647.01

CONTINUOUS COMPOUNDING: £1000 at 5% for 10 years
    principal=1000, rate=0.05, time=10, frequency="continuous"
    Result: £1648.72""",
    annotations=ToolAnnotations(
        title="Compound Interest",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def compound_interest(
    principal: Annotated[float, Field(description="Initial principal amount (e.g., 1000)")],
    rate: Annotated[float, Field(description="Annual interest rate (e.g., 0.05 for 5%)")],
    time: Annotated[float, Field(description="Time period in years (e.g., 10)", ge=0)],
    frequency: Annotated[
        Literal["annual", "semi-annual", "quarterly", "monthly", "daily", "continuous"],
        Field(description="Compounding frequency"),
    ] = "annual",
) -> str:
    """Calculate compound interest."""
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


@mcp.tool(
    name="perpetuity",
    description="""Calculate present value of a perpetuity (infinite series of payments).

A perpetuity is an annuity that continues forever. Common in:
    - Preferred stock dividends
    - Endowment funds
    - Real estate with infinite rental income
    - UK Consol bonds (historically)

Formulas:
    Level Ordinary: PV = C / r
    Level Due: PV = C / r × (1 + r)
    Growing: PV = C / (r - g), where r > g

Examples:

LEVEL PERPETUITY: £1000 annual payment at 5%
    payment=1000, rate=0.05
    Result: PV = £20,000

GROWING PERPETUITY: £1000 payment growing 3% annually at 8% discount
    payment=1000, rate=0.08, growth_rate=0.03
    Result: PV = £20,000

PERPETUITY DUE: £1000 at period start at 5%
    payment=1000, rate=0.05, when='begin'
    Result: PV = £21,000""",
    annotations=ToolAnnotations(
        title="Perpetuity Calculations",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def perpetuity(
    payment: Annotated[float, Field(description="Periodic payment amount (e.g., 1000)")],
    rate: Annotated[float, Field(description="Discount rate per period (e.g., 0.05)", gt=0)],
    growth_rate: Annotated[
        float | None,
        Field(
            description="Payment growth rate (None or 0 for level, e.g., 0.03 for growing)", ge=0
        ),
    ] = None,
    when: Annotated[
        Literal["end", "begin"], Field(description="Payment timing: 'end' or 'begin'")
    ] = "end",
) -> str:
    """Calculate present value of perpetuity."""
    try:
        # Validate inputs
        if rate <= 0:
            raise ValueError("Discount rate must be positive")

        if growth_rate is not None:
            if growth_rate < 0:
                raise ValueError("Growth rate cannot be negative")
            if growth_rate >= rate:
                raise ValueError(
                    f"Growth rate ({growth_rate}) must be less than discount rate ({rate}) "
                    "for perpetuity to have finite value"
                )

        # Calculate present value based on type
        if growth_rate is not None and growth_rate > 0:
            # Growing perpetuity: PV = C / (r - g)
            pv = payment / (rate - growth_rate)
            perpetuity_type = "growing"
        elif when == "begin":
            # Perpetuity due (payments at beginning): PV = C/r × (1+r)
            pv = (payment / rate) * (1 + rate)
            perpetuity_type = "level_due"
        else:
            # Ordinary perpetuity (payments at end): PV = C / r
            pv = payment / rate
            perpetuity_type = "level_ordinary"

        # Build metadata
        metadata: Dict[str, Any] = {
            "type": perpetuity_type,
            "payment": payment,
            "rate": rate,
        }
        if growth_rate is not None:
            metadata["growth_rate"] = growth_rate
        if when != "end":
            metadata["when"] = when

        return format_result(float(pv), metadata)

    except Exception as e:
        raise ValueError(f"Perpetuity calculation failed: {str(e)}")
