"""GO OMNIÆGIS Example 1: Epsilon-Robustness Validation.

This example demonstrates how to validate dynamical system trajectories
using the ε-robustness validator.
"""

import numpy as np
from omniagis import EpsilonRobustnessValidator


def main():
    print("=" * 70)
    print("GO OMNIÆGIS — Epsilon-Robustness Validation Example")
    print("=" * 70)
    print()

    # Create a validator with ε = 0.1
    validator = EpsilonRobustnessValidator(epsilon=0.1)
    print(f"Validator configured with ε = {validator.epsilon}")
    print()

    # Example 1: Identical trajectories (should PASS)
    print("Example 1: Validating identical trajectories")
    trajectory_1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    reference_1 = trajectory_1.copy()
    result_1 = validator.validate(trajectory_1, reference_1)
    print(f"  Status: {result_1.status}")
    print(f"  Max distance: {result_1.max_dist:.6f}")
    print(f"  Mean distance: {result_1.mean_dist:.6f}")
    print()

    # Example 2: Small deviation (should PASS)
    print("Example 2: Validating trajectory with small deviation")
    trajectory_2 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    reference_2 = np.array([[0.0, 0.0], [1.01, 1.01], [2.02, 2.01], [3.0, 3.0]])
    result_2 = validator.validate(trajectory_2, reference_2)
    print(f"  Status: {result_2.status}")
    print(f"  Max distance: {result_2.max_dist:.6f}")
    print(f"  Mean distance: {result_2.mean_dist:.6f}")
    print()

    # Example 3: Large deviation (should NO PASS)
    print("Example 3: Validating trajectory with large deviation")
    trajectory_3 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    reference_3 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [5.0, 5.0]])
    result_3 = validator.validate(trajectory_3, reference_3)
    print(f"  Status: {result_3.status}")
    print(f"  Max distance: {result_3.max_dist:.6f}")
    print(f"  Mean distance: {result_3.mean_dist:.6f}")
    print()

    # Example 4: Moderate deviation (should PARTIAL PASS)
    print("Example 4: Validating trajectory with moderate deviation")
    trajectory_4 = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    reference_4 = np.array([[0.0], [1.05], [2.02], [3.01], [4.03], [5.2]])
    result_4 = validator.validate(trajectory_4, reference_4)
    print(f"  Status: {result_4.status}")
    print(f"  Max distance: {result_4.max_dist:.6f}")
    print(f"  Mean distance: {result_4.mean_dist:.6f}")
    print()

    print("=" * 70)
    print("Validation Summary:")
    print(f"  ε-robustness criterion: pointwise L2 distance ≤ {validator.epsilon}")
    print(f"  PASS: max(dist) ≤ ε")
    print(f"  PARTIAL PASS: mean(dist) ≤ ε but max(dist) > ε")
    print(f"  NO PASS: mean(dist) > ε")
    print("=" * 70)


if __name__ == "__main__":
    main()
