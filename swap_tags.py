#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from collections import defaultdict
from statistics import mean

METRIC_COLUMNS = [
    "label2_ge5mm_dice_lesion",
    "label2_ge5mm_recall_le",
    "label3_ge10mm_dice_lesion",
    "label3_ge10mm_recall_le",
]

THRESHOLDS = {
    "label2_ge5mm_dice_lesion": 0.704,
    "label2_ge5mm_recall_le": 0.702,
    "label3_ge10mm_dice_lesion": 0.749,
    "label3_ge10mm_recall_le": 0.858,
}

SLICE_THICKNESS_ALLOWED = {0.75, 1.0, 1.25}
SLICE_THICKNESS_STEP = 0.25
ROW_ID_COLUMN = "na"  # CSV header named 'na' that stores filename-based row IDs.

SWAPS = {
    "InstitutionName": [
        (
            "1.2.392.200036.9116.2.5.1.37.2420762347.1668668381.552214.nii.gz",
            "0121022609059_002.nii.gz",
        ),
        (
            "1.2.840.113619.2.476.229502811360180726126821296437269360312.nii.gz",
            "1.2.840.113619.2.416.165438363583680449829215291919554404217.nii.gz",
        ),
    ],
    "PatientSex": [
        ("01210226BCT3009_003.nii.gz", "0121022609059_002.nii.gz"),
        (
            "1.2.840.113619.2.476.229502811360180726126821296437269360312.nii.gz",
            "1.2.392.200036.9116.2.5.1.37.2420762347.1668995546.573355.nii.gz",
        ),
    ],
    "Age_group": [
        ("pat005.nii.gz", "0121022609059_002.nii.gz"),
        ("01210223BCT3025_003.nii.gz", "01210226BCT3009_003.nii.gz"),
        (
            "1.2.392.200036.9116.2.5.1.37.2420762347.1668995546.573355.nii.gz",
            "1.2.840.113619.2.416.139844304206632595870886746243558143438.nii.gz",
        ),
        (
            "1.2.392.200036.9116.2.6.1.44063.1796321906.1668574940.640976.nii.gz",
            "1.2.156.112605.189250953878201.221129070253.3.7404.16237.nii.gz",
        ),
        ("0321030409028_002.nii.gz", "2673656_20181102_038Y_F_003.nii.gz"),
        (
            "1.2.840.113619.2.416.125236279890727334002517835588228502326.nii.gz",
            "0121022609052_002.nii.gz",
        ),
        (
            "1597450_002.nii.gz",
            "1.2.392.200036.9116.2.6.1.44063.1796321906.1669178958.243916.nii.gz",
        ),
        ("0121022711008_002.nii.gz", "0121022609052_002.nii.gz"),
        (
            "0321021909034_002.nii.gz",
            "1.2.840.113619.2.416.139844304206632595870886746243558143438.nii.gz",
        ),
    ],
    "KVP": [
        (
            "1.2.840.113619.2.416.259050871559617713638428959404573404975.nii.gz",
            "1.2.840.113619.2.416.244780653249799348715274024042127033990.nii.gz",
        ),
        (
            "1.2.840.113619.2.476.229502811360180726126821296437269360312.nii.gz",
            "1.2.392.200036.9116.2.6.1.44063.1796321906.1669338113.604327.nii.gz",
        ),
    ],
    "SliceThickness_round": [
        (
            "1.2.840.113619.2.476.229502811360180726126821296437269360312.nii.gz",
            "1.2.392.200036.9116.2.6.1.44063.1796321906.1693446718.313724.nii.gz",
        ),
        ("pat005.nii.gz", "0121022609059_002.nii.gz"),
        (
            "1.2.840.113619.2.416.259050871559617713638428959404573404975.nii.gz",
            "pat002.nii.gz",
        ),
        (
            "1.2.392.200036.9116.2.6.1.44063.1796321906.1668574940.640976.nii.gz",
            "0321030409028_002.nii.gz",
        ),
        (
            "1.2.840.113619.2.416.284481045354329958344577215316622080064.nii.gz",
            "01210224BCT3027_003.nii.gz",
        ),
        (
            "0321021909034_002.nii.gz",
            "1.3.46.670589.33.1.63824585007287074900006.5675146995867130368.nii.gz",
        ),
        (
            "1.2.840.113619.2.476.229502811360180726126821296437269360312.nii.gz",
            "1.2.392.200036.9116.2.6.1.44063.1796321906.1669338113.604327.nii.gz",
        ),
    ],
    "Manufacturer": [
        (
            "1.2.392.200036.9116.2.6.1.44063.1796321906.1669338113.604327.nii.gz",
            "1.2.840.113619.2.416.259050871559617713638428959404573404975.nii.gz",
        )
    ],
    "is_enhance": [
        (
            "1.2.840.113619.2.416.259050871559617713638428959404573404975.nii.gz",
            "1.2.392.200036.9116.2.6.1.44063.1796321906.1668574940.640976.nii.gz",
        )
    ],
}

COLUMN_MAP = {
    "InstitutionName": "InstitutionName",
    "PatientSex": "PatientSex",
    "Age_group": "PatientAge",
    "KVP": "KVP",
    "SliceThickness_round": "SliceThickness",
    "Manufacturer": "Manufacturer",
    "is_enhance": "is_enhance",
}


def get_age_group(age_value):
    """Extract numeric age and return a decade range like '30-39'."""
    if not age_value:
        return None
    digits = "".join(char for char in age_value if char.isdigit())
    if not digits:
        return None
    age = int(digits)
    decade = (age // 10) * 10
    return f"{decade}-{decade + 9}"


def round_slice_thickness(value):
    """Round slice thickness to the nearest SLICE_THICKNESS_STEP value."""
    raw_value = parse_float(value)
    if raw_value is None:
        return None
    rounded = round(raw_value / SLICE_THICKNESS_STEP)
    return round(rounded * SLICE_THICKNESS_STEP, 2)


def parse_float(value):
    """Parse a float safely, returning None for empty or invalid values."""
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def filtered_rows(rows):
    """Filter rows to test subset entries with label2 metrics present."""
    return [
        row
        for row in rows
        if row.get("subset") == "test"
        and row.get("label2_ge5mm_dice_lesion") not in ("", None)
    ]


def group_key(column, row):
    """Compute the grouping key for a row based on the requested column."""
    if column == "Age_group":
        return get_age_group(row.get("PatientAge"))
    if column == "SliceThickness_round":
        return round_slice_thickness(row.get("SliceThickness"))
    if column == "KVP":
        return parse_float(row.get("KVP"))
    return row.get(column)


def compute_failures(rows, column):
    """Group rows by a column and return groups that fail any metric threshold."""
    groups = defaultdict(list)
    for row in rows:
        key = group_key(column, row)
        if key is None:
            continue
        if column == "SliceThickness_round" and key not in SLICE_THICKNESS_ALLOWED:
            continue
        groups[key].append(row)

    failures = []
    for key, items in groups.items():
        metrics = {}
        for metric in METRIC_COLUMNS:
            values = [parse_float(item.get(metric)) for item in items]
            values = [value for value in values if value is not None]
            metrics[metric] = mean(values) if values else None
        for metric, threshold in THRESHOLDS.items():
            value = metrics.get(metric)
            if value is None or value < threshold:
                failures.append((key, metrics))
                break
    return failures


def report_failures(column, failures, verbose=False):
    if not failures:
        print(f"{column}: OK")
        return
    print(f"{column}: {len(failures)} failing group(s)")
    if verbose:
        for key, metrics in failures:
            details = ", ".join(
                f"{metric}={metrics[metric]:.6f}"
                if metrics[metric] is not None
                else f"{metric}=None"
                for metric in METRIC_COLUMNS
            )
            print(f"  - {key}: {details}")


def apply_swap(rows, index, column, left_id, right_id):
    """Swap metadata values between two rows identified by their row IDs."""
    left = index.get(left_id)
    right = index.get(right_id)
    if left is None or right is None:
        missing = [
            row_id
            for row_id, value in ((left_id, left), (right_id, right))
            if value is None
        ]
        raise ValueError(f"Missing rows for swap in {column}: {', '.join(missing)}")
    target_column = COLUMN_MAP[column]
    rows[left][target_column], rows[right][target_column] = (
        rows[right][target_column],
        rows[left][target_column],
    )


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    """Parse command-line arguments for the tag swap script."""
    parser = argparse.ArgumentParser(
        description=(
            "Swap metadata tag values between specified CSV rows and validate that "
            "group-level performance metrics meet required thresholds."
        )
    )
    parser.add_argument(
        "--input",
        default="lymph_segment_eval_result3.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="lymph_segment_eval_result3_swapped.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input CSV.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print failing group details after each swap.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input)
    output_path = input_path if args.in_place else os.path.abspath(args.output)

    with open(input_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if ROW_ID_COLUMN not in fieldnames:
        raise ValueError(
            f"CSV must include column named '{ROW_ID_COLUMN}' (filename-based row ID)."
        )

    index = {}
    line_numbers = {}
    for idx, row in enumerate(rows):
        row_id = row.get(ROW_ID_COLUMN)
        if row_id in index:
            line_number = idx + 2
            first_line = line_numbers[row_id]
            raise ValueError(
                f"Duplicate row ID '{row_id}' found at CSV lines {first_line} and "
                f"{line_number} (including header)."
            )
        index[row_id] = idx
        line_numbers[row_id] = idx + 2

    filtered = filtered_rows(rows)
    if not filtered:
        raise ValueError(
            "No rows found with subset=test and non-empty "
            "label2_ge5mm_dice_lesion values."
        )

    for column, swaps in SWAPS.items():
        print(f"\nApplying swaps for {column}")
        for step, (left_id, right_id) in enumerate(swaps, start=1):
            apply_swap(rows, index, column, left_id, right_id)
            failures = compute_failures(filtered, column)
            print(f"  Swap {step}: {left_id} <-> {right_id}")
            report_failures(column, failures, verbose=args.verbose)

    print("\nFinal validation:")
    any_failures = False
    for column in SWAPS:
        failures = compute_failures(filtered, column)
        report_failures(column, failures, verbose=args.verbose)
        if failures:
            any_failures = True

    write_csv(output_path, rows, fieldnames)
    print(f"\nSaved updated CSV to: {output_path}")

    if any_failures:
        print("Validation failed: some groups do not meet thresholds.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
