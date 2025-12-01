from typing import Any, Callable, Tuple

import narwhals as nw
from narwhals import Implementation
import pandas as pd


def dataframe_converter(df: Any) -> Tuple[pd.DataFrame, Callable[[pd.DataFrame], Any]]:
    """Convert a dataframe-like object to pandas and return a converter back to native."""

    narwhals_df = nw.from_native(df, eager_only=True)
    implementation = narwhals_df.implementation
    namespace = nw.get_native_namespace(narwhals_df)
    pandas_df = narwhals_df.to_pandas()

    def to_native(result_df: pd.DataFrame):
        if implementation == Implementation.PANDAS:
            return result_df

        # Prefer explicit helpers when available
        from_pandas = getattr(namespace, "from_pandas", None)
        if callable(from_pandas):
            try:
                return from_pandas(result_df)
            except Exception:
                pass

        constructor = getattr(namespace, "DataFrame", None)
        if callable(constructor):
            try:
                return constructor(result_df)
            except Exception:
                pass

        if implementation == Implementation.PYARROW:
            try:
                import pyarrow as pa

                return pa.Table.from_pandas(result_df, preserve_index=False)
            except Exception:
                return result_df

        return result_df

    return pandas_df, to_native
