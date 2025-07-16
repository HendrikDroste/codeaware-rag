def mean_reciprocal_rank(expected_metadata, retrieved_metadata):
    """
    Evaluate the position of the first relevant result
    
     Args:
        expected_metadata: List of expected metadata dictionaries with 'filename' keys
        retrieved_metadata: List of retrieved metadata dictionaries with 'source' or 'filename' keys
    
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
            
            if retrieved_start is None or retrieved_end is None or expected_start is None or expected_end is None:
                raise ValueError(
                    f"Expected metadata for {expected_file} is missing line numbers: "
                    f"{expected_start}, {expected_end}"
                )
                
            # Check for line range overlap
            if retrieved_start <= expected_end and retrieved_end >= expected_start:
                return 1.0 / (i + 1)
    return 0


def line_coverage_ratio(expected_metadata, retrieved_metadata):
    """
    Measures how many of the expected code lines were found
    
    Args:
        expected_metadata: List of expected metadata dictionaries with 'filename' keys
        retrieved_metadata: List of retrieved metadata dictionaries with 'source' or 'filename' keys
    
    Returns:
        Ratio of expected lines found in retrieved snippets, or 0 if no expected lines
    """
    if not expected_metadata or not retrieved_metadata:
        return 0.0

    # Create a mapping of filenames to line ranges
    expected_files = {}
    for meta in expected_metadata:
        filename = meta.get('filename')
        if not filename:
            continue
        
        line_start = meta.get('line_start')
        line_end = meta.get('line_end')
        
        if filename not in expected_files:
            expected_files[filename] = []
            
        if line_start is not None and line_end is not None:
            expected_files[filename].append((line_start, line_end))
    
    retrieved_files = {}
    for meta in retrieved_metadata:
        filename = meta.get('source') or meta.get('filename')
        if not filename:
            continue
            
        line_start = meta.get('line_start')
        line_end = meta.get('line_end')
        
        if filename not in retrieved_files:
            retrieved_files[filename] = []
            
        if line_start is not None and line_end is not None:
            retrieved_files[filename].append((line_start, line_end))
    
    # Calculate line coverage
    total_expected_lines = 0
    covered_lines = 0
    
    for filename, expected_ranges in expected_files.items():
        for start, end in expected_ranges:
            expected_line_count = end - start + 1
            total_expected_lines += expected_line_count
            
            if filename in retrieved_files:
                for r_start, r_end in retrieved_files[filename]:
                    # Calculate overlap between expected and retrieved line ranges
                    overlap_start = max(start, r_start)
                    overlap_end = min(end, r_end)
                    overlap = max(0, overlap_end - overlap_start + 1)
                    covered_lines += overlap
                    
                    # Avoid double-counting covered lines
                    if overlap > 0:
                        expected_line_count -= overlap
    
    if total_expected_lines == 0:
        return 0.0
        
    return covered_lines / total_expected_lines


def file_match_ratio(expected_metadata, retrieved_metadata):
    """
    Calculate the score based on filename and line number matches.

    Args:
        expected_metadata: List of dictionaries with filename, line_start, line_end
        retrieved_metadata: List of dictionaries with metadata from retriever

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
            retrieved_file = retrieved.get("source") or retrieved.get("filename")

            # Check if filenames match (handle potential path differences)
            filename_match = False
            if expected_file and retrieved_file:
                if expected_file == retrieved_file:
                    filename_match = True

            if filename_match:
                # If file matches, check for line overlap
                retrieved_start = retrieved.get("line_start")
                retrieved_end = retrieved.get("line_end")

                # If no line numbers in metadata, count as a match based on filename only
                if retrieved_start is None or retrieved_end is None:
                    matches += 1
                    break

                # Check for line range overlap
                if (retrieved_start <= expected_end and retrieved_end >= expected_start):
                    matches += 1
                    break

    return matches / len(expected_metadata)
