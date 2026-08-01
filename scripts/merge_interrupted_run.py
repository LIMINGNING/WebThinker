import argparse
import json
import os
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def record_id(record):
    return str(record.get("id", "")).strip()


def completed_records(records, source: Path):
    completed = {}
    for record in records:
        if "Output" not in record:
            continue
        item_id = record_id(record)
        if not item_id:
            raise ValueError(f"Completed record without an id in {source}")
        if item_id in completed:
            raise ValueError(f"Duplicate completed id {item_id} in {source}")
        completed[item_id] = record
    return completed


def main():
    parser = argparse.ArgumentParser(
        description="Merge an interrupted partial run with a continuation run."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = load_json(args.reference)
    partial = completed_records(load_json(args.partial), args.partial)
    continuation = completed_records(
        load_json(args.continuation), args.continuation
    )

    overlap = sorted(set(partial) & set(continuation))
    if overlap:
        raise ValueError(f"Runs overlap on ids: {', '.join(overlap)}")

    combined = {**partial, **continuation}
    expected_ids = [record_id(record) for record in reference]
    missing = [item_id for item_id in expected_ids if item_id not in combined]
    extra = sorted(set(combined) - set(expected_ids))
    if missing or extra:
        raise ValueError(f"Missing ids: {missing}; extra ids: {extra}")

    merged = [combined[item_id] for item_id in expected_ids]
    if len(merged) != len(reference):
        raise ValueError(
            f"Merged {len(merged)} records, expected {len(reference)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(merged, file, indent=4, ensure_ascii=False)
    os.replace(temporary, args.output)

    print(
        f"Merged {len(partial)} partial and {len(continuation)} continuation "
        f"records into {args.output} ({len(merged)} total)."
    )


if __name__ == "__main__":
    main()
