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


def precision_at_k(expected_metadata, retrieved_metadata, k=5):
    """
    Proportion of relevant documents among the first k results
    
    Args:
        expected_metadata: List of expected metadata dictionaries with 'filename' keys
        retrieved_metadata: List of retrieved metadata dictionaries with 'source' or 'filename' keys
        k: Number of top results to consider
    
    Returns:
        Precision at k, or 0 if no retrieved metadata
    """
    if not retrieved_metadata:
        return 0

    k = min(k, len(retrieved_metadata))
    relevant_count = 0

    for retrieved in retrieved_metadata[:k]:
        retrieved_file = retrieved.get("source") or retrieved.get("filename")
        retrieved_start = retrieved.get("line_start")
        retrieved_end = retrieved.get("line_end")
        
        # Check if this retrieved document matches any of the expected documents
        for expected in expected_metadata:
            expected_file = expected.get("filename")
            expected_start = expected.get("line_start")
            expected_end = expected.get("line_end")
            
            # Check if filenames match
            if not (expected_file and expected_file == retrieved_file):
                continue
                
            # If no line numbers in metadata, count as match based on filename only
            if retrieved_start is None or retrieved_end is None or expected_start is None or expected_end is None:
                relevant_count += 1
                break
                
            # Check for line range overlap
            if retrieved_start <= expected_end and retrieved_end >= expected_start:
                relevant_count += 1
                break

    return relevant_count / k


def recall_at_k(expected_metadata, retrieved_metadata, k=5):
    """
    Proportion of found relevant documents among the first k results
    
    Args:
        expected_metadata: List of expected metadata dictionaries with 'filename' keys
        retrieved_metadata: List of retrieved metadata dictionaries with 'source' or 'filename' keys
        k: Number of top results to consider
    
    Returns:
        Recall at k, or 0 if no expected or retrieved metadata
    """
    if not expected_metadata or not retrieved_metadata:
        return 0

    k = min(k, len(retrieved_metadata))
    found_expected_indices = set()

    for retrieved in retrieved_metadata[:k]:
        retrieved_file = retrieved.get("source") or retrieved.get("filename")
        retrieved_start = retrieved.get("line_start")
        retrieved_end = retrieved.get("line_end")
        
        # Check if this retrieved document matches any of the expected documents
        for i, expected in enumerate(expected_metadata):
            expected_file = expected.get("filename")
            expected_start = expected.get("line_start")
            expected_end = expected.get("line_end")
            
            # Check if filenames match
            if not (expected_file and expected_file == retrieved_file):
                continue
                
            # If no line numbers in metadata, count as match based on filename only
            if retrieved_start is None or retrieved_end is None or expected_start is None or expected_end is None:
                found_expected_indices.add(i)
                continue
                
            # Check for line range overlap
            if retrieved_start <= expected_end and retrieved_end >= expected_start:
                found_expected_indices.add(i)

    return len(found_expected_indices) / len(expected_metadata)


def line_coverage_ratio(expected_snippets, retrieved_snippets):
    """
    Measures how many of the expected code lines were found
    
    Args:
        expected_snippets: List of code snippets that should be found
        retrieved_snippets: List of retrieved code snippets
    
    Returns:
        Ratio of expected lines found in retrieved snippets, or 0 if no expected lines
    """
    expected_lines = set()
    for snippet in expected_snippets:
        expected_lines.update(line.strip() for line in snippet.split("\n") if line.strip())

    if not expected_lines:
        return 0.0

    found_lines = set()
    for snippet in retrieved_snippets:
        found_lines.update(line.strip() for line in snippet.split("\n") if line.strip())

    intersection = expected_lines.intersection(found_lines)
    return len(intersection) / len(expected_lines)


def exact_match_score(expected_metadata, retrieved_metadata):
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


def f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1-Score from precision and recall.
    
    Args:
        precision: Precision value
        recall: Recall value
        
    Returns:
        F1-Score (harmonic mean of precision and recall)
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
