# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from functools import reduce
from operator import iconcat

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("to_dataframe",)

numpy = Numpy.instance()
np = NumpyMetadata.instance()


def _default_levelname(index: int) -> str:
    return "sub" * index + "entry"


@high_level_function()
def to_dataframe(
    array, *, how="inner", levelname=_default_levelname, anonymous="values"
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        how (None or str): Passed to
            [pd.merge](https://pandas.pydata.org/pandas-docs/version/1.0.3/reference/api/pandas.merge.html)
            to combine DataFrames for each multiplicity into one DataFrame. If
            None, a list of Pandas DataFrames is returned.
        levelname (int -> str): Computes a name for each level of the row index
            from the number of levels deep.
        anonymous (str): Column name to use if the `array` does not contain
            records; otherwise, column names are derived from record fields.

    Converts Awkward data structures into Pandas
    [MultiIndex](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html)
    rows and columns. The resulting DataFrame(s) contains no Awkward structures.

    #ak.Array structures can't be losslessly converted into a single DataFrame;
    different fields in a record structure might have different nested list
    lengths, but a DataFrame can have only one index.

    If `how` is None, this function always returns a list of DataFrames (even
    if it contains only one DataFrame); otherwise `how` is passed to
    [pd.merge](https://pandas.pydata.org/pandas-docs/version/1.0.3/reference/api/pandas.merge.html)
    to merge them into a single DataFrame with the associated loss of data.

    In the following example, nested lists are converted into MultiIndex rows.
    The index level names `"entry"`, `"subentry"` and `"subsubentry"` can be
    controlled with the `levelname` parameter. The column name `"values"` is
    assigned because this array has no fields; it can be controlled with the
    `anonymous` parameter.

        >>> ak.to_dataframe(ak.Array([[[1.1, 2.2], [], [3.3]],
        ...                           [],
        ...                           [[4.4], [5.5, 6.6]],
        ...                           [[7.7]],
        ...                           [[8.8]]]))
                                    values
        entry subentry subsubentry
        0     0        0               1.1
                       1               2.2
              2        0               3.3
        2     0        0               4.4
              1        0               5.5
                       1               6.6
        3     0        0               7.7
        4     0        0               8.8

    In this example, nested records are converted into MultiIndex columns.
    (MultiIndex rows and columns can be mixed; these examples are deliberately
    simple.)

        >>> ak.to_dataframe(ak.Array([
        ...     {"I": {"a": _, "b": {"i": _}}, "II": {"x": {"y": {"z": _}}}}
        ...     for _ in range(0, 50, 10)]))
                I      II
                a   b   x
                    i   y
                        z
        entry
        0       0   0   0
        1      10  10  10
        2      20  20  20
        3      30  30  30
        4      40  40  40

    The following two examples show how fields of different length lists are
    merged. With `how="inner"` (default), only subentries that exist for all
    fields are preserved; with `how="outer"`, all subentries are preserved at
    the expense of requiring missing values.

        >>> ak.to_dataframe(ak.Array([{"x": [], "y": [4.4, 3.3, 2.2, 1.1]},
        ...                           {"x": [1], "y": [3.3, 2.2, 1.1]},
        ...                           {"x": [1, 2], "y": [2.2, 1.1]},
        ...                           {"x": [1, 2, 3], "y": [1.1]},
        ...                           {"x": [1, 2, 3, 4], "y": []}]),
        ...                          how="inner")
                        x    y
        entry subentry
        1     0         1  3.3
        2     0         1  2.2
              1         2  1.1
        3     0         1  1.1

    The same with `how="outer"`:

        >>> ak.to_dataframe(ak.Array([{"x": [], "y": [4.4, 3.3, 2.2, 1.1]},
        ...                           {"x": [1], "y": [3.3, 2.2, 1.1]},
        ...                           {"x": [1, 2], "y": [2.2, 1.1]},
        ...                           {"x": [1, 2, 3], "y": [1.1]},
        ...                           {"x": [1, 2, 3, 4], "y": []}]),
        ...                          how="outer")
                          x    y
        entry subentry
        0     0         NaN  4.4
              1         NaN  3.3
              2         NaN  2.2
              3         NaN  1.1
        1     0         1.0  3.3
              1         NaN  2.2
              2         NaN  1.1
        2     0         1.0  2.2
              1         2.0  1.1
        3     0         1.0  1.1
              1         2.0  NaN
              2         3.0  NaN
        4     0         1.0  NaN
              1         2.0  NaN
              2         3.0  NaN
              3         4.0  NaN
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, how, levelname, anonymous)


def _impl(array, how, levelname, anonymous):
    try:
        import pandas
    except ImportError as err:
        raise ImportError(
            """install the 'pandas' package with:

    pip install pandas --upgrade

or

    conda install pandas"""
        ) from err

    if how is not None:
        out = None
        for df in to_dataframe(
            array, how=None, levelname=levelname, anonymous=anonymous
        ):
            if out is None:
                out = df
            else:
                out = pandas.merge(out, df, how=how, left_index=True, right_index=True)
        return out

    def recurse(layout, row_arrays, col_names):
        if layout.is_indexed and not layout.is_option:
            return recurse(layout.project(), row_arrays, col_names)

        elif layout.parameter("__array__") in ("string", "bytestring"):
            return [(ak.operations.to_numpy(layout), row_arrays, col_names)]

        elif layout.is_option and layout.purelist_depth == 2 and len(row_arrays) == 0:
            # Special handling for TOP-LEVEL option types containing single-level nested lists (option[var * T]).
            # When None or empty lists appear, we need to ensure at least one row
            # is created per entry for proper merging with non-nested fields.
            # Only apply at top level (len(row_arrays) == 0) to avoid interfering with deeper nesting.
            # Note: purelist_depth==2 means option[var * T] where T has purelist_depth 1.

            # Get the mask indicating which entries are None
            mask_index = ak.operations.is_none(layout, highlevel=False)

            # Project out the option to get the nested list content
            projected = layout.project()

            # Process the projected content through normal nested list logic
            offsets, flattened = projected._offsets_and_flattened(axis=1, depth=1)
            starts, stops = offsets.data[:-1], offsets.data[1:]
            counts = stops - starts
            if ak._util.win or ak._util.bits32:
                counts = layout.backend.nplike.astype(counts, np.int32)

            # For entries with None (not in projected), set count to 1 to create a row
            # We need to adjust counts to include the None entries
            full_counts = numpy.zeros(len(layout), dtype=counts.dtype)

            # Convert mask to numpy boolean array
            mask_array = numpy.asarray(ak.operations.to_numpy(mask_index), dtype=bool)

            # Map the projected counts back to the full array
            valid_indices = ~mask_array
            full_counts[valid_indices] = counts

            # For entries with None, set count to 1 so they get a row
            full_counts[mask_array] = 1
            none_indices = mask_array

            # Create row arrays based on full counts
            if len(row_arrays) == 0:
                newrows = [
                    numpy.repeat(numpy.arange(len(full_counts), dtype=full_counts.dtype), full_counts)
                ]
            else:
                newrows = [numpy.repeat(x, full_counts) for x in row_arrays]

            # For subentry indices, we need to handle None entries specially
            subentry_indices = []
            for i, count in enumerate(full_counts):
                if none_indices[i]:
                    # None entry gets subentry index 0
                    subentry_indices.append(0)
                else:
                    # Normal entries get their range of indices
                    subentry_indices.extend(range(count))

            newrows.append(numpy.asarray(subentry_indices, dtype=full_counts.dtype))

            # Create the data column with NaN for None entries
            # First, convert flattened to numpy and create an array of the right length
            flat_numpy = ak.operations.to_numpy(flattened)
            total_rows = numpy.sum(full_counts)

            # Map flattened data to the correct positions
            if np.issubdtype(flat_numpy.dtype, np.floating):
                column_data = numpy.full(total_rows, np.nan, dtype=flat_numpy.dtype)
            elif np.issubdtype(flat_numpy.dtype, np.str_):
                # Ensure dtype is wide enough for "nan" (3 characters)
                char_width = flat_numpy.dtype.itemsize // 4
                if char_width < 3:
                    target_dtype = np.dtype(("U", 3))
                else:
                    target_dtype = flat_numpy.dtype
                column_data = numpy.full(total_rows, "nan", dtype=target_dtype)
            elif np.issubdtype(flat_numpy.dtype, np.bytes_):
                # Ensure dtype is wide enough for b"nan" (3 bytes)
                byte_width = flat_numpy.dtype.itemsize
                if byte_width < 3:
                    target_dtype = np.dtype(("S", 3))
                else:
                    target_dtype = flat_numpy.dtype
                column_data = numpy.full(total_rows, b"nan", dtype=target_dtype)
            else:
                # For other types, create a masked array
                column_data = numpy.ma.masked_all(total_rows, dtype=flat_numpy.dtype)

            # Fill in the valid (non-None) data
            data_position = 0
            row_position = 0
            for i, count in enumerate(full_counts):
                if not none_indices[i]:
                    # Copy the data for this entry
                    column_data[row_position:row_position + count] = flat_numpy[data_position:data_position + count]
                    data_position += count
                row_position += count

            return [(column_data, newrows, col_names)]

        elif layout.purelist_depth > 1:
            offsets, flattened = layout._offsets_and_flattened(axis=1, depth=1)
            starts, stops = offsets.data[:-1], offsets.data[1:]
            counts = stops - starts
            if ak._util.win or ak._util.bits32:
                counts = layout.backend.nplike.astype(counts, np.int32)
            if len(row_arrays) == 0:
                newrows = [
                    numpy.repeat(numpy.arange(len(counts), dtype=counts.dtype), counts)
                ]
            else:
                newrows = [numpy.repeat(x, counts) for x in row_arrays]
            newrows.append(
                numpy.arange(offsets[-1], dtype=counts.dtype)
                - numpy.repeat(starts, counts)
            )
            return recurse(flattened, newrows, col_names)

        elif layout.is_union:
            layout = _union_to_record(layout, anonymous)
            if layout.is_union:
                return [(ak.operations.to_numpy(layout), row_arrays, col_names)]
            else:
                return reduce(
                    iconcat,
                    (
                        recurse(layout._getitem_field(n), row_arrays, (*col_names, n))
                        for n in layout.fields
                    ),
                    [],
                )

        elif isinstance(layout, ak.contents.RecordArray):
            return reduce(
                iconcat,
                (
                    recurse(layout._getitem_field(n), row_arrays, (*col_names, n))
                    for n in layout.fields
                ),
                [],
            )

        else:
            return [(ak.operations.to_numpy(layout), row_arrays, col_names)]

    layout = ak.operations.to_layout(array, allow_record=True, primitive_policy="error")
    if isinstance(layout, ak.record.Record):
        layout2 = layout.array[layout.at : layout.at + 1]
    else:
        layout2 = layout

    tables = []
    last_row_arrays = None
    for column, row_arrays, col_names in recurse(layout2, [], ()):
        if isinstance(layout, ak.record.Record):
            row_arrays = row_arrays[1:]  # Record --> one-element RecordArray
        if len(col_names) == 0:
            columns = [anonymous]
        else:
            columns = pandas.MultiIndex.from_tuples([col_names])

        # Before filling, ensure dtype is wide enough for "nan" fill value. Need
        # at least three characters / bytes for 'nan' or b'nan', respectively.
        if numpy.ma.is_masked(column):
            if np.issubdtype(column.dtype, np.str_):
                char_width = column.dtype.itemsize // 4
                if char_width < 3:
                    column = column.astype(np.dtype(("U", 3)))
            elif np.issubdtype(column.dtype, np.bytes_):
                byte_width = column.dtype.itemsize
                if byte_width < 3:
                    column = column.astype(np.dtype(("S", 3)))
        # Pandas can't handle masked strings
        if np.issubdtype(column.dtype, np.str_):
            column = numpy.ma.filled(column, "nan")
        elif np.issubdtype(column.dtype, np.bytes_):
            column = numpy.ma.filled(column, b"nan")

        if (
            last_row_arrays is not None
            and len(last_row_arrays) == len(row_arrays)
            and all(
                numpy.array_equal(x, y) for x, y in zip(last_row_arrays, row_arrays)
            )
        ):
            oldcolumns = tables[-1].columns
            if isinstance(oldcolumns, pandas.MultiIndex):
                numold = len(oldcolumns.levels)
            else:
                numold = max(len(x) for x in oldcolumns)
            numnew = len(columns.levels)
            maxnum = max(numold, numnew)
            if numold != maxnum:
                oldcolumns = pandas.MultiIndex.from_tuples(
                    [x + ("",) * (maxnum - numold) for x in oldcolumns]
                )
                tables[-1].columns = oldcolumns
            if numnew != maxnum:
                columns = pandas.MultiIndex.from_tuples(
                    [x + ("",) * (maxnum - numnew) for x in columns]
                )

            newframe = pandas.DataFrame(
                data=column, index=tables[-1].index, columns=columns
            )
            tables[-1] = pandas.concat([tables[-1], newframe], axis=1)

        else:
            if len(row_arrays) == 0:
                index = pandas.RangeIndex(len(column), name=levelname(0))
            else:
                index = pandas.MultiIndex.from_arrays(
                    row_arrays, names=[levelname(i) for i in range(len(row_arrays))]
                )
            tables.append(pandas.DataFrame(data=column, index=index, columns=columns))

        last_row_arrays = row_arrays

    for table in tables:
        if isinstance(table.columns, pandas.MultiIndex) and table.columns.nlevels == 1:
            table.columns = table.columns.get_level_values(0)

    return tables


def _union_to_record(unionarray, anonymous):
    contents = []
    for layout in unionarray.contents:
        if layout.is_indexed and not layout.is_option:
            contents.append(layout.project())
        elif layout.is_union:
            contents.append(_union_to_record(layout, anonymous))
        elif layout.is_option:
            contents.append(
                ak.operations.fill_none(layout, np.nan, axis=0, highlevel=False)
            )
        else:
            contents.append(layout)

    if not any(isinstance(x, ak.contents.RecordArray) for x in contents):
        return ak.contents.UnionArray(
            unionarray.tags,
            unionarray.index,
            contents,
            parameters=unionarray.parameters,
        )

    else:
        seen = set()
        all_names = []
        for layout in contents:
            if isinstance(layout, ak.contents.RecordArray):
                for field in layout.fields:
                    if field not in seen:
                        seen.add(field)
                        all_names.append(field)
            else:
                if anonymous not in seen:
                    seen.add(anonymous)
                    all_names.append(anonymous)

        missingarray = ak.contents.IndexedOptionArray(
            ak.index.Index64(
                unionarray.backend.nplike.full(unionarray.length, -1, dtype=np.int64)
            ),
            ak.contents.EmptyArray(),
        )

        all_fields = []
        for name in all_names:
            union_contents = []
            for layout in contents:
                if isinstance(layout, ak.contents.RecordArray):
                    for field in layout.fields:
                        if name == field:
                            union_contents.append(layout._getitem_field(field))
                            break
                    else:
                        union_contents.append(missingarray)
                else:
                    if name == anonymous:
                        union_contents.append(layout)
                    else:
                        union_contents.append(missingarray)

            all_fields.append(
                ak.contents.UnionArray.simplified(
                    unionarray.tags,
                    unionarray.index,
                    union_contents,
                    parameters=unionarray.parameters,
                )
            )

        return ak.contents.RecordArray(all_fields, all_names, unionarray.length)
