# Path: transportation_solver.py

import numpy as np
import pandas as pd
import os

# Helper function to read the external file
def read_transportation_file(filepath):
    """Reads transportation problem parameters from an external Excel file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file '{filepath}' does not exist. Please check the path.")
    data = pd.read_excel(filepath, index_col=0)
    costs = data.iloc[:-1, :-1].values.astype(int)
    supply = data.iloc[:-1, -1].values.astype(int)
    demand = data.iloc[-1, :-1].values.astype(int)
    return supply, demand, costs

# Northwest corner method
def northwest_corner(supply, demand):
    """Finds an initial feasible solution using the Northwest corner method."""
    rows, cols = len(supply), len(demand)
    allocation = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            alloc = min(supply[i], demand[j])
            allocation[i][j] = alloc
            supply[i] -= alloc
            demand[j] -= alloc
    return allocation

# Minimum cost method
def minimum_cost_method(supply, demand, costs):
    """Finds an initial feasible solution using the Minimum Cost method."""
    supply_copy, demand_copy = supply.copy(), demand.copy()
    allocation = np.zeros(costs.shape, dtype=int)
    flat_costs = [(i, j, costs[i][j]) for i in range(costs.shape[0]) for j in range(costs.shape[1])]
    sorted_costs = sorted(flat_costs, key=lambda x: x[2])

    for i, j, cost in sorted_costs:
        if supply_copy[i] > 0 and demand_copy[j] > 0:
            alloc = min(supply_copy[i], demand_copy[j])
            allocation[i][j] = alloc
            supply_copy[i] -= alloc
            demand_copy[j] -= alloc

    return allocation

# Vogel's approximation method
def vogels_method(supply, demand, costs):
    """Finds an initial feasible solution using Vogel's approximation method."""
    supply_copy, demand_copy = supply.copy(), demand.copy()
    allocation = np.zeros(costs.shape, dtype=int)

    while supply_copy.sum() > 0 and demand_copy.sum() > 0:
        # Calculate penalties for each row and column
        row_penalties = []
        for row in range(costs.shape[0]):
            row_costs = costs[row, :][demand_copy > 0]
            if len(row_costs) > 1:
                penalty = np.partition(row_costs, 1)[1] - np.partition(row_costs, 0)[0]
            else:
                penalty = 0 if len(row_costs) == 1 else float('inf')
            row_penalties.append(penalty)

        col_penalties = []
        for col in range(costs.shape[1]):
            col_costs = costs[:, col][supply_copy > 0]
            if len(col_costs) > 1:
                penalty = np.partition(col_costs, 1)[1] - np.partition(col_costs, 0)[0]
            else:
                penalty = 0 if len(col_costs) == 1 else float('inf')
            col_penalties.append(penalty)

        # Find the maximum penalty
        row_max = max(row_penalties)
        col_max = max(col_penalties)

        if row_max >= col_max:
            row = row_penalties.index(row_max)
            col = np.argmin(costs[row, :][demand_copy > 0])
            col = np.where(demand_copy > 0)[0][col]
        else:
            col = col_penalties.index(col_max)
            row = np.argmin(costs[:, col][supply_copy > 0])
            row = np.where(supply_copy > 0)[0][row]

        # Allocate to the selected cell
        alloc = min(supply_copy[row], demand_copy[col])
        allocation[row, col] = alloc
        supply_copy[row] -= alloc
        demand_copy[col] -= alloc

    return allocation

# Transportation Simplex Algorithm
def transportation_simplex(allocation, costs):
    """Solves the transportation problem using the simplex algorithm."""
    while True:
        u = np.full(allocation.shape[0], None)
        v = np.full(allocation.shape[1], None)
        u[0] = 0

        # Calculate u and v
        for _ in range(allocation.size):
            for i in range(allocation.shape[0]):
                for j in range(allocation.shape[1]):
                    if allocation[i, j] > 0:
                        if u[i] is not None and v[j] is None:
                            v[j] = costs[i, j] - u[i]
                        elif u[i] is None and v[j] is not None:
                            u[i] = costs[i, j] - v[j]

        # Calculate reduced costs
        reduced_costs = costs - (u[:, None] + v[None, :])

        # Check for optimality
        if np.all(reduced_costs >= 0):
            break

        # Find entering variable
        i, j = np.unravel_index(np.argmin(reduced_costs), reduced_costs.shape)

        # Create loop and update allocation
        # (Implementation details depend on pivoting logic for cycles in transportation problems)

    return allocation

# Example usage
if __name__ == "__main__":
    filepath = "C:\\Users\\acrtl\\OneDrive - De Vinci\\Documents\\ESILV\\A3\\Uvic\\Operationnal Research\\New folder\\transportation_data.xlsx"
    try:
        supply, demand, costs = read_transportation_file(filepath)
        print("Northwest Corner Method:")
        nw_allocation = northwest_corner(supply, demand)
        print(nw_allocation)

        print("Minimum Cost Method:")
        mc_allocation = minimum_cost_method(supply, demand, costs)
        print(mc_allocation)

        print("Vogel's Method:")
        vogels_allocation = vogels_method(supply, demand, costs)
        print(vogels_allocation)
    except FileNotFoundError as e:
        print(e)
