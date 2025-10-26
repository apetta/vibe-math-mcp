"""Statistical analysis tools using Polars for performance."""

from typing import Annotated, Dict, List, Literal, Union
from pydantic import Field
from mcp.types import ToolAnnotations
import polars as pl

from ..server import mcp
from ..core import format_result, format_json


@mcp.tool(
    name="statistics",
    description="""Comprehensive statistical analysis using Polars.

Analysis types:
    - describe: Count, mean, std, min, max, median
    - quartiles: Q1, Q2, Q3, IQR
    - outliers: IQR-based detection (values beyond Q1-1.5×IQR or Q3+1.5×IQR)

Examples:

DESCRIPTIVE STATISTICS:
    data=[1,2,3,4,5,100], analyses=["describe"]
    Result: {count:6, mean:19.17, std:39.25, min:1, max:100, median:3.5}

QUARTILES:
    data=[1,2,3,4,5], analyses=["quartiles"]
    Result: {Q1:2, Q2:3, Q3:4, IQR:2}

OUTLIER DETECTION:
    data=[1,2,3,4,5,100], analyses=["outliers"]
    Result: {outlier_values:[100], outlier_count:1, lower_bound:-1, upper_bound:8.5}

FULL ANALYSIS:
    data=[1,2,3,4,5,100], analyses=["describe","quartiles","outliers"]
    Result: All three analyses combined""",
    annotations=ToolAnnotations(
        title="Statistical Analysis",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def statistics(
    data: Annotated[List[float], Field(description="List of numerical values (e.g., [1,2,3,4,5,100])")],
    analyses: Annotated[List[Literal["describe", "quartiles", "outliers"]], Field(description="Types of analysis to perform")],
) -> str:
    """Comprehensive statistical analysis."""
    try:
        df = pl.DataFrame({"values": data})

        results = {}

        if "describe" in analyses:
            # Comprehensive descriptive statistics
            results["describe"] = {
                "count": len(data),
                "mean": float(df.select(pl.col("values").mean()).item()),
                "std": float(df.select(pl.col("values").std()).item()),
                "min": float(df.select(pl.col("values").min()).item()),
                "max": float(df.select(pl.col("values").max()).item()),
                "median": float(df.select(pl.col("values").median()).item()),
            }

        if "quartiles" in analyses:
            # Quartile analysis
            results["quartiles"] = {
                "Q1": float(df.select(pl.col("values").quantile(0.25)).item()),
                "Q2": float(df.select(pl.col("values").quantile(0.50)).item()),
                "Q3": float(df.select(pl.col("values").quantile(0.75)).item()),
                "IQR": float(
                    df.select(
                        pl.col("values").quantile(0.75) - pl.col("values").quantile(0.25)
                    ).item()
                ),
            }

        if "outliers" in analyses:
            # IQR-based outlier detection
            q1 = df.select(pl.col("values").quantile(0.25)).item()
            q3 = df.select(pl.col("values").quantile(0.75)).item()
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers_df = df.filter(
                (pl.col("values") < lower_bound) | (pl.col("values") > upper_bound)
            )

            results["outliers"] = {
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outlier_values": outliers_df.select("values").to_series().to_list(),
                "outlier_count": len(outliers_df),
            }

        return format_json(results)
    except Exception as e:
        raise ValueError(f"Statistical analysis failed: {str(e)}")


@mcp.tool(
    name="pivot_table",
    description="""Create pivot tables from tabular data using Polars.

Like Excel pivot tables: reshape data with row/column dimensions and aggregated values.

Example:

SALES BY REGION AND PRODUCT:
    data=[
        {"region":"North","product":"A","sales":100},
        {"region":"North","product":"B","sales":150},
        {"region":"South","product":"A","sales":80},
        {"region":"South","product":"B","sales":120}
    ],
    index="region", columns="product", values="sales", aggfunc="sum"
    Result:
        product |  A   |  B
        --------|------|------
        North   | 100  | 150
        South   |  80  | 120

COUNT AGGREGATION:
    Same data with aggfunc="count"
    Result: Count of entries per region-product combination

AVERAGE SCORES:
    data=[{"dept":"Sales","role":"Manager","score":85}, ...]
    index="dept", columns="role", values="score", aggfunc="mean"
    Result: Average scores by department and role""",
    annotations=ToolAnnotations(
        title="Pivot Table",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def pivot_table(
    data: Annotated[List[Dict[str, Union[str, float]]], Field(description="List of row dictionaries")],
    index: Annotated[str, Field(description="Column name for row index")],
    columns: Annotated[str, Field(description="Column name for pivot columns")],
    values: Annotated[str, Field(description="Column name to aggregate")],
    aggfunc: Annotated[Literal["sum", "mean", "count", "min", "max"], Field(description="Aggregation function")] = "sum",
) -> str:
    """Create pivot tables."""
    try:
        df = pl.DataFrame(data)

        # Map aggfunc to Polars-compatible values
        agg_map = {
            "sum": "sum",
            "mean": "mean",
            "count": "len",  # Polars uses "len" for count
            "min": "min",
            "max": "max",
        }

        if aggfunc not in agg_map:
            raise ValueError(f"Unknown aggregation function: {aggfunc}")

        # Polars pivot requires eager mode
        pivot_df = df.pivot(
            on=columns,
            index=index,
            values=values,
            aggregate_function=agg_map[aggfunc],  # type: ignore[arg-type]
        )

        # Convert to dict for JSON response
        result = pivot_df.to_dicts()

        return format_result(
            result, {"index": index, "columns": columns, "values": values, "aggfunc": aggfunc}
        )
    except Exception as e:
        raise ValueError(
            f"Pivot table creation failed: {str(e)}. "
            f"Ensure data contains columns: {index}, {columns}, {values}"
        )


@mcp.tool(
    name="correlation",
    description="""Calculate correlation matrices between multiple variables using Polars.

Methods:
    - pearson: Linear correlation (-1 to +1, 0 = no linear relationship)
    - spearman: Rank-based correlation (monotonic, robust to outliers)

Examples:

PEARSON CORRELATION:
    data={"x":[1,2,3], "y":[2,4,6], "z":[1,1,1]},
    method="pearson", output_format="matrix"
    Result: {
        "x": {"x":1.0, "y":1.0, "z":NaN},
        "y": {"x":1.0, "y":1.0, "z":NaN},
        "z": {"x":NaN, "y":NaN, "z":NaN}
    }

PAIRWISE FORMAT:
    data={"height":[170,175,168], "weight":[65,78,62]},
    method="pearson", output_format="pairs"
    Result: [{"var1":"height", "var2":"weight", "correlation":0.89}]

SPEARMAN (RANK):
    data={"x":[1,2,100], "y":[2,4,200]},
    method="spearman"
    Result: Perfect correlation (1.0) despite non-linear relationship""",
    annotations=ToolAnnotations(
        title="Correlation Analysis",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def correlation(
    data: Annotated[Dict[str, List[float]], Field(description="Dict of variable names to values (e.g., {'x':[1,2,3],'y':[2,4,6]})")],
    method: Annotated[Literal["pearson", "spearman"], Field(description="Correlation method")] = "pearson",
    output_format: Annotated[Literal["matrix", "pairs"], Field(description="Output format: 'matrix' or 'pairs'")] = "matrix",
) -> str:
    """Calculate correlation matrices."""
    try:
        df = pl.DataFrame(data)

        # Verify all columns have same length
        lengths = [len(v) for v in data.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All variables must have the same number of observations")

        if method == "spearman":
            # Rank transformation for Spearman
            rank_cols = [pl.col(c).rank().alias(c) for c in df.columns]
            df = df.select(rank_cols)

        # Compute correlation matrix using NumPy (Polars corr requires NumPy)
        corr_matrix = df.to_pandas().corr().to_dict()

        if output_format == "pairs":
            # Convert to pairwise format
            pairs = []
            columns = list(data.keys())
            for i, col1 in enumerate(columns):
                for col2 in columns[i + 1 :]:
                    pairs.append(
                        {"var1": col1, "var2": col2, "correlation": corr_matrix[col1][col2]}
                    )
            result = pairs
        else:
            result = corr_matrix

        return format_result(
            result, {"method": method, "variables": list(data.keys()), "n_observations": lengths[0]}
        )
    except Exception as e:
        raise ValueError(f"Correlation analysis failed: {str(e)}")
