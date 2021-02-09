#!/usr/bin/env python3

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp


c = 3 # size of cell
n = 9 # size of table


def load_table(string_repr):
    return np.array([int(char) for char in string_repr]).reshape((9,9))


def to_string(table):
    return ''.join([str(num) for num in table.flatten()])


def vars_to_matrix(vars):
    table = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(1, n + 1):
                if vars[i, j, k].solution_value() == 1:
                    table[i, j] = k
    return table


def solve(table):
    solver = pywraplp.Solver.CreateSolver('CBC_MIXED_INTEGER_PROGRAMMING')

    vars = {}
    for i in range(n):
        for j in range(n):
            for k in range(1, n + 1):
                vars[i, j, k] = solver.BoolVar(f'var_{i}{j}{k}')

            # add initial values
            if table[i, j] > 0:
                solver.Add(vars[i, j, table[i, j]] == 1)

    # columnwise
    for k in range(1, n + 1):
        for j in range(n):
            solver.Add(solver.Sum([vars[i, j, k] for i in range(n)]) == 1)

    # rowwise
    for k in range(1, n + 1):
        for i in range(n):
            solver.Add(solver.Sum([vars[i, j, k] for j in range(n)]) == 1)

    # cellwise
    for k in range(1, n + 1):
        for offset_i in range(0, n, c):
            for offset_j in range(0, n, c):
                solver.Add(solver.Sum([vars[offset_i + i, offset_j + j, k] for i in range(c) for j in range(c)]) == 1)

    # must have value larger than zero
    for i in range(n):
        for j in range(n):
            solver.Add(solver.Sum([vars[i, j, k] for k in range(1, n + 1)]) == 1)


    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        solution = vars_to_matrix(vars)
    else:
        print('There is no optimal solution')
        solution = None

    return solution


def main():
    sudokus = pd.read_csv("sudoku.csv")
    success = 0
    for i in range(len(sudokus)):
        table = load_table(sudokus.loc[i]['quizzes'])
        solution = solve(table)

        if to_string(solution) == sudokus.loc[i]['solutions']:
            success += 1

    print('Success rate: ', success/len(sudokus))

main()