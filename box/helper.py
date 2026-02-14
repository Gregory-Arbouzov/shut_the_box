def get_distinct_sums(target, start=1):
    if target == 0:
        return [[]]
    results = []
    for i in range(start, target + 1):
        # Recursively find combinations for the remaining sum
        # Using i + 1 ensures the next number is distinct and larger
        for combo in get_distinct_sums(target - i, i + 1):
            results.append([i] + combo)

    adjusted_results = [flip_option for flip_option in results if max(flip_option) <= 9]

    return adjusted_results