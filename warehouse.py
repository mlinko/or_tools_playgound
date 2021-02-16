#!/usr/bin/env python3

import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model


# Pruduct scheduling problem
# - There are N warehouses and M costumers
# - Each customer c_i should be served by a single warehouse
# - Each customer c_i has a demand d_i
# - Each warehouse w_j has a limited capacity k_i
# - There is a travel cost t_{i,j} for serving a
#   customer c_i from warehouse w_j
#
# GOAL: find an assignment that minimizes the cost
# source of task description:
# http://ai.unibo.it/sites/ai.unibo.it/files/u9/ReifiedCst.pdf


def solve_boolean(demands, capacities, cost):
    M = len(demands)  # number of customers
    N = len(capacities)  # number of warehouses

    solver = pywraplp.Solver.CreateSolver('CBC_MIXED_INTEGER_PROGRAMMING')

    # define variables
    x = {}

    for i in range(M):
        for j in range(N):
            x[i, j] = solver.BoolVar(f'x_{{{i},{j}}}')

    # each customer should be served by a single warehouse
    for i in range(M):
        solver.Add(solver.Sum([x[i, j] for j in range(N)]) == 1)

    # each warehouse has a capacity
    for j in range(N):
        solver.Add(solver.Sum(
                demands[i] * x[i, j] for i in range(M)
            ) <= capacities[j])

    solver.Minimize(solver.Sum(
        cost[i, j]*x[i, j] for i in range(M) for j in range(N)
        ))

    status = solver.Solve()

    assignment_vector = np.zeros((M), dtype=int)
    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())

        for i in range(M):
            for j in range(N):
                if x[i, j].solution_value():
                    assignment_vector[i] = j
        print('Assignment vector:', assignment_vector, sep='\n')
    else:
        print('The problem does not have an optimal solution.')
    print()
    return assignment_vector


def solve_reified(demands, capacities, cost):
    # solving the problem by applying reified constraints
    # About reified constaints in OR Tools:
    # https://www.cis.upenn.edu/~cis189/files/Lecture8.pdf

    M = len(demands)  # number of customers
    N = len(capacities)  # number of warehouses

    model = cp_model.CpModel()
    # solver = pywraplp.Solver.CreateSolver('SCIP')
    solver = cp_model.CpSolver()

    # define variables
    x = {}
    for i in range(M):
        # x_i = warehouse number from ith customer is served
        x[i] = model.NewIntVar(0, N-1, f'x_{i}')

    z = {}
    for i in range(M):
        for j in range(N):
            z[i, j] = model.NewBoolVar(f'z_{{{i},{j}}}')
        # model.Add(x[i] > -)

    # z := x_i == j
    for i in range(M):
        for j in range(N):
            model.Add(x[i] == j).OnlyEnforceIf(z[i, j])
            model.Add(x[i] != j).OnlyEnforceIf(z[i, j].Not())

    # capacities cannot been outgrown by demands
    for j in range(N):
        model.Add(sum(demands[i] * z[i, j] for i in range(M)) <= capacities[j])

    model.Minimize(sum(
        cost[i, j]*z[i, j] for j in range(N) for i in range(M)
        ))

    solution_printer = cp_model.ObjectiveSolutionPrinter()
    status = solver.SolveWithSolutionCallback(model, solution_printer)
    print(status)

    assignment_vector = np.zeros((M), dtype=int)
    if status == cp_model.OPTIMAL:
        print('Solution:')
        print('Objective value =', solution_printer.ObjectiveValue())
        for i in range(M):
            assignment_vector[i] = solver.Value(x[i])
        print('Assignment vector:', assignment_vector, sep='\n')
    else:
        print('The problem does not have an optimal solution.')
    print()
    return assignment_vector


def main():
    demands = np.array([12, 17, 6, 20])
    capacities = np.array([18, 18, 20])
    cost = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ])

    solve_boolean(demands, capacities, cost)
    solve_reified(demands, capacities, cost)

    demands = np.array([12, 17, 6, 20])
    capacities = np.array([18, 18, 20])
    cost = np.array([
        [1, 1, 1],
        [2, 1, 1],
        [1, 1, 1],
        [1, 1, 10],
    ])

    solve_boolean(demands, capacities, cost)
    solve_reified(demands, capacities, cost)


if __name__ == '__main__':
    main()
