# Saiteki Worker

A worker service that processes optimization tasks from NATS and runs evolutionary optimization using Shinka.

## Function Examples

Users need to provide three functions when creating optimization tasks. Here are examples using a circle packing optimization problem:

### optimize Function (goes in EVOLVE-BLOCK)

```python
def optimize():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
    """
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))

    # Place circles in a structured pattern
    # This is a simple pattern - evolution will improve this

    # First, place a large circle in the center
    centers[0] = [0.5, 0.5]

    # Place 8 circles around it in a ring
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    # Place 16 more circles in an outer ring
    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]

    # Additional positioning adjustment to make sure all circles
    # are inside the square and don't overlap
    # Clip to ensure everything is inside the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)
    
    return centers, radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Then, limit by distance to other circles
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii
```

### validate_result Function

```python
def validate_result(run_output: Any, atol=1e-6) -> Tuple[bool, Optional[str]]:
    """
    Validates circle packing results based on the output of 'run_optimization'.

    Args:
        run_output: Tuple (centers, radii, reported_sum) from optimize function
        atol: Absolute tolerance for numerical comparisons

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    if not isinstance(run_output, tuple) or len(run_output) != 3:
        return False, "Expected tuple of (centers, radii, sum) from optimize function"
    
    centers, radii, reported_sum = run_output
    
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers)
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)

    n_expected = 26
    if centers.shape != (n_expected, 2):
        return False, f"Centers shape incorrect. Expected ({n_expected}, 2), got {centers.shape}"
    
    if radii.shape != (n_expected,):
        return False, f"Radii shape incorrect. Expected ({n_expected},), got {radii.shape}"

    if np.any(radii < 0):
        negative_indices = np.where(radii < 0)[0]
        return False, f"Negative radii found for circles at indices: {negative_indices}"

    if not np.isclose(np.sum(radii), reported_sum, atol=atol):
        return False, f"Sum of radii ({np.sum(radii):.6f}) does not match reported ({reported_sum:.6f})"

    # Check all circles are within unit square
    for i in range(n_expected):
        x, y = centers[i]
        r = radii[i]
        if x - r < -atol or x + r > 1 + atol or y - r < -atol or y + r > 1 + atol:
            return False, f"Circle {i} (x={x:.4f}, y={y:.4f}, r={r:.4f}) is outside unit square"

    # Check no circles overlap
    for i in range(n_expected):
        for j in range(i + 1, n_expected):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - atol:
                return False, f"Circles {i} & {j} overlap. Dist: {dist:.4f}, Sum Radii: {(radii[i] + radii[j]):.4f}"
    
    return True, "All circles are validly placed with no overlaps"
```

### aggregate_metrics Function

```python
def aggregate_metrics(results: List[Any], results_dir: str) -> Dict[str, Any]:
    """
    Aggregates metrics for circle packing optimization.
    
    Args:
        results: List of results from run_optimization() calls
        results_dir: Directory to save additional result files
        
    Returns:
        Dictionary with metrics including 'combined_score'
    """
    if not results:
        return {
            "combined_score": 0.0, 
            "error": "No results to aggregate",
            "public": {},
            "private": {}
        }

    # Extract the best result (highest sum of radii)
    best_result = None
    best_score = 0.0
    
    for result in results:
        if isinstance(result, tuple) and len(result) == 3:
            centers, radii, sum_radii = result
            if sum_radii > best_score:
                best_score = sum_radii
                best_result = result
    
    if best_result is None:
        return {
            "combined_score": 0.0,
            "error": "No valid results found",
            "public": {},
            "private": {}
        }
    
    centers, radii, sum_radii = best_result
        
    # Save detailed results
    extra_file = os.path.join(results_dir, "circle_packing_details.npz")
    try:
        np.savez(
            extra_file,
            centers=centers,
            radii=radii,
            sum_radii=sum_radii,
        )
    except Exception as e:
        print(f"Warning: Could not save detailed results: {e}")
    
    return {
        "combined_score": float(sum_radii),
        "public": {
            "num_circles": len(centers),
            "best_sum_radii": float(sum_radii)
        },
        "private": {}
    }
```

## Usage

1. Copy the three functions above
2. Modify them for your specific optimization problem
3. Paste them into the web UI when creating a new optimization task
4. The worker will automatically generate the necessary files and run the evolution

## Notes

- The `optimize()` function should be self-contained and return the result to be optimized
- The `validate_result()` function should check if the optimization output is valid
- The `aggregate_metrics()` function should return a dictionary with at least `combined_score`
- All functions will have access to common imports like `numpy`, `json`, `os`, etc.
- The circle packing example shows how to optimize for packing 26 circles in a unit square to maximize the sum of radii
