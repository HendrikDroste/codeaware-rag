METRIC_VERSION = "unique-line-coverage-v2"


def mean_reciprocal_rank(expected_metadata, retrieved_metadata):
    """
    Evaluate the position of the first relevant result

     Args:
        expected_metadata: List of expected metadata dictionaries with
            'filename' keys
        retrieved_metadata: List of retrieved metadata dictionaries with
            'source' or 'filename' keys

    Returns:
        Reciprocal Rank of the first relevant result, or 0 if none found
    """
    if not expected_metadata or not retrieved_metadata:
        return 0

    for i, retrieved in enumerate(retrieved_metadata):
        retrieved_file = retrieved.get("source") or retrieved.get("filename")
        retrieved_start = retrieved.get("line_start")
        retrieved_end = retrieved.get("line_end")

        for expected in expected_metadata:
            expected_file = expected.get("filename")
            expected_start = expected.get("line_start")
            expected_end = expected.get("line_end")

            if not (expected_file and expected_file == retrieved_file):
                continue

            if (
                retrieved_start is None
                or retrieved_end is None
                or expected_start is None
                or expected_end is None
            ):
                raise ValueError(
                    f"Expected metadata for {expected_file} "
                    "is missing line numbers: "
                    f"{expected_start}, {expected_end}"
                )

            # Check for line range overlap
            if (
                retrieved_start <= expected_end
                and retrieved_end >= expected_start
            ):
                return 1.0 / (i + 1)
    return 0


def line_coverage_ratio(expected_metadata, retrieved_metadata):
    """
    Measures how many of the expected code lines were found

    Args:
        expected_metadata: List of expected metadata dictionaries with
            'filename' keys
        retrieved_metadata: List of retrieved metadata dictionaries with
            'source' or 'filename' keys

    Returns:
        Fraction of unique expected file/line pairs retrieved, or 0 if
        there are no expected lines. Overlapping ranges count once.
    """
    if not expected_metadata or not retrieved_metadata:
        return 0.0

    def lines_by_file(metadata):
        files = {}
        for item in metadata:
            path = item.get("source") or item.get("filename")
            start, end = item.get("line_start"), item.get("line_end")
            if path and start is not None and end is not None:
                if start < 1 or end < start:
                    raise ValueError(
                        "Line ranges must be positive and inclusive"
                    )
                files.setdefault(path, set()).update(range(start, end + 1))
        return files

    expected_files = lines_by_file(expected_metadata)
    retrieved_files = lines_by_file(retrieved_metadata)
    total_expected_lines = sum(map(len, expected_files.values()))
    if total_expected_lines == 0:
        return 0.0
    covered_lines = sum(
        len(lines & retrieved_files.get(path, set()))
        for path, lines in expected_files.items()
    )
    return covered_lines / total_expected_lines


def file_match_ratio(expected_metadata, retrieved_metadata):
    """
    Calculate the score based on filename and line number matches.

    Args:
        expected_metadata: List of dictionaries with filename,
            line_start, line_end
        retrieved_metadata: List of dictionaries with metadata from
            retriever

    Returns:
        Float score between 0 and 1
    """
    if not expected_metadata:
        return 0

    matches = 0
    for expected in expected_metadata:
        expected_file = expected.get("filename")
        expected_start = expected.get("line_start")
        expected_end = expected.get("line_end")

        for retrieved in retrieved_metadata:
            retrieved_file = retrieved.get("source") or retrieved.get(
                "filename"
            )

            # Check if filenames match (handle potential path
            # differences)
            filename_match = False
            if expected_file and retrieved_file:
                if expected_file == retrieved_file:
                    filename_match = True

            if filename_match:
                # If file matches, check for line overlap
                retrieved_start = retrieved.get("line_start")
                retrieved_end = retrieved.get("line_end")

                # If no line numbers in metadata, count as a match based
                # on filename only
                if retrieved_start is None or retrieved_end is None:
                    matches += 1
                    break

                # Check for line range overlap
                if (
                    retrieved_start <= expected_end
                    and retrieved_end >= expected_start
                ):
                    matches += 1
                    break

    return matches / len(expected_metadata)
