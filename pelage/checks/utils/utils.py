"""Utility functions for pelage."""


def compare_schema(data_schema: dict, expected_schema: dict) -> str:
    """Returns a list of mismatched dtypes: (column, expected_type, actual_type)"""
    unmatched_colum_dtypes = [
        (key, value, data_schema[key])
        for key, value in expected_schema.items()
        if value != data_schema[key]
    ]
    messages = [
        f"{column=}, {expected_type=}, {real_dtype=}"
        for (column, expected_type, real_dtype) in unmatched_colum_dtypes
    ]

    return "\n".join(messages)
