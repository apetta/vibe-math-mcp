"""Statistical analysis tools using Polars for performance."""

from typing import Dict, List, Literal, Union
from mcp.types import ToolAnnotations
import polars as pl

from ..server import mcp
from ..core import format_result, format_json


@mcp.tool(
    name="statistics",
    description="Comprehensive statistical analysis: descriptive statistics, quartiles, outlier detection.",
    annotations=ToolAnnotations(
        title="Statistical Analysis",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def statistics(
    data: List[float], analyses: List[Literal["describe", "quartiles", "outliers"]]
) -> str:
    """
    Perform comprehensive statistical analysis using Polars.

    Examples:
        - data=[1,2,3,4,5,100], analyses=["describe","outliers"] → full stats + outliers
        - data=[1,2,3,4,5], analyses=["quartiles"] → Q1, Q2, Q3

    Args:
        data: List of numerical values
        analyses: Types of analysis to perform

    Returns:
        JSON with statistical results
    """
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
    description="Create pivot tables from tabular data with aggregation functions.",
    annotations=ToolAnnotations(
        title="Pivot Table",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def pivot_table(
    data: List[Dict[str, Union[str, float]]],
    index: str,
    columns: str,
    values: str,
    aggfunc: Literal["sum", "mean", "count", "min", "max"] = "sum",
) -> str:
    """
    Create pivot tables using Polars for speed.

    Examples:
        - Sales data pivoted by region and product with sum aggregation
        - User data pivoted by department and role with count aggregation

    Args:
        data: List of dictionaries representing rows
        index: Column name for row index
        columns: Column name for pivot columns
        values: Column name to aggregate
        aggfunc: Aggregation function

    Returns:
        JSON with pivot table result
    """
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
    description="Calculate correlation matrices between multiple variables (Pearson or Spearman).",
    annotations=ToolAnnotations(
        title="Correlation Analysis",
        readOnlyHint=True,
        idempotentHint=True,
    ),
)
async def correlation(
    data: Dict[str, List[float]],
    method: Literal["pearson", "spearman"] = "pearson",
    output_format: Literal["matrix", "pairs"] = "matrix",
) -> str:
    """
    Compute correlation matrix using Polars.

    Examples:
        - data={"x": [1,2,3], "y": [2,4,6], "z": [1,1,1]}, method="pearson" → correlation matrix
        - data={"x": [1,2,3], "y": [2,4,6]}, output_format="pairs" → pairwise correlations

    Args:
        data: Dictionary of variable names to value lists
        method: Correlation method (pearson or spearman)
        output_format: Output as matrix or pairwise correlations

    Returns:
        JSON with correlation results
    """
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
