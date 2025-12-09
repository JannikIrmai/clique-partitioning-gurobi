import numpy as np
import gurobipy
from time import time
import matplotlib.pyplot as plt
from itertools import combinations

MIN_VIOLATION = 1e-2
T_LIMIT = 60


def separate_two_partition(values: np.ndarray, p: int = 10):
    """
    The two partition inequality wrt. two disjoint node sets A, B is defined as

        \sum_{a \in A, b \in B} x_{ab} - \sum_{ab \subseteq A} x_{ab} - \sum_{ab \subseteq B} x_{ab} \leq min(|A|, |B|)

    This method implements the separation heuristic for this class of inequalities that was proposed by Sorensen (2020).

    :param values: symmetric n x n matrix of edge variable values
    :param p: maximum number of |A| + |B|
    :return:
    """
    n = len(values)
    for a, b in combinations(range(n), 2):
        # continue if value of edge ab is integral
        if values[a, b] <= MIN_VIOLATION or values[a, b] >= 1 - MIN_VIOLATION:
            continue
        np.fill_diagonal(values, 0)

        # Start with A = {a}, B = {b}, i.e. inequality x_ab <= 1
        violation = values[a, b] - 1  # amount by which inequality in violated
        A = [a]
        B = [b]
        selected = np.zeros(n, dtype=bool)  # nodes that are either in A or B
        selected[[a, b]] = True

        # delta_A[i] is the sum of the costs of all edges from i to a node in A (delta_B analogue for B)
        delta_A = values[a].copy()
        delta_B = values[b].copy()

        max_violation_depth = violation
        best_violation = violation
        best_A = A.copy()
        best_B = B.copy()
        # greedily add nodes to A or B until |A|+|B| = p
        for k in range(2, p):
            # compute violation gain for adding any node to A
            gain_A = delta_B - delta_A
            if len(A) < len(B):
                gain_A -= 1
            gain_A[selected] = -np.inf
            # compute violation gain for adding any node to A
            gain_B = delta_A - delta_B
            if len(B) < len(A):
                gain_B -= 1
            gain_B[selected] = -np.inf

            best_A_idx = np.argmax(gain_A)
            best_B_idx = np.argmax(gain_B)
            if gain_A[best_A_idx] > gain_B[best_B_idx]:
                violation += gain_A[best_A_idx]
                A.append(best_A_idx)
                selected[best_A_idx] = True
                delta_A += values[best_A_idx]
            else:
                violation += gain_B[best_B_idx]
                B.append(best_B_idx)
                selected[best_B_idx] = True
                delta_B += values[best_B_idx]

            # if this is the new best violation depth, store A and B for later
            k = len(A) + len(B)
            violation_depth = violation / np.sqrt(k * (k-1) / 2)
            if violation_depth > max_violation_depth:
                max_violation_depth = violation_depth
                best_violation = violation
                best_A = A.copy()
                best_B = B.copy()

        A = best_A
        B = best_B
        violation = best_violation
        delta_A = values[A].sum(axis=0)
        delta_B = values[B].sum(axis=0)

        # Add initial partition if sufficiently violated
        if violation > MIN_VIOLATION:
            yield A, B, violation

        # try swapping nodes in A or B for nodes not in A or B
        swap = True
        while swap:
            swap = False

            # compute gain for swapping any node with any node in A
            # gain_A[a, i] = delta_A[a] - delta_A[i] - delta_B[a] + delta_B[i] + values[a, i]
            gain_A = delta_A[A, None] - delta_A[None, :] - delta_B[A, None] + delta_B[None, :] + values[A]
            gain_A[:, A] = -np.inf
            a_idx, i = np.unravel_index(gain_A.argmax(), gain_A.shape)
            a = A[a_idx]

            # compute gain for swapping any node with any node in B
            gain_B = delta_B[B, None] - delta_B[None, :] - delta_A[B, None] + delta_A[None, :] + values[B]
            gain_B[:, B] = -np.inf
            b_idx, j = np.unravel_index(gain_B.argmax(), gain_B.shape)
            b = B[b_idx]

            if gain_A[a_idx, i] > gain_B[b_idx, j] and gain_A[a_idx, i] > 1e-6:
                swap = True
                A[a_idx] = i
                violation += gain_A[a_idx, i]
                delta_A -= values[a]
                delta_A += values[i]

            if gain_B[b_idx, j] > gain_A[a_idx, i] and gain_B[b_idx, j] > 1e-6:
                swap = True
                B[b_idx] = j
                violation += gain_B[b_idx, j]
                delta_B -= values[b]
                delta_B += values[j]

        if violation > MIN_VIOLATION:
            yield A, B, violation


def separate_triangle(values: np.ndarray):
    # find violated triangle inequalities
    violation = -values[:, :, None] + values[:, None, :] + values[None, :, :] - 1
    iu, ju = np.triu_indices(values.shape[0])
    violation[iu, ju] = 0
    idx = np.argwhere(violation > MIN_VIOLATION)
    return idx


def get_triangle_inequality(variables, i, j, k):
    # create triangle inequality for triangle i, j, k
    return variables[i, k] + variables[k, j] - variables[i, j] <= 1


def get_tp_inequality(variables, A, B):
    # create linear inequalities from two partition A, B
    expr = 0
    for a, b in combinations(A, 2):
        expr -= variables[a, b]
    for a, b in combinations(B, 2):
        expr -= variables[a, b]
    for a in A:
        for b in B:
            expr += variables[a, b]
    return expr <= min(len(A), len(B))


def create_canonical_model(cost: np.ndarray, binary: bool = True, add_triangle: bool = True):
    # create gurobi model of clique partitioning instance
    assert cost.ndim == 2
    assert cost.shape[0] == cost.shape[1]
    assert np.all(cost.T == cost)
    assert np.all(cost[np.diag_indices_from(cost)] == 0)
    n = cost.shape[0]
    model = gurobipy.Model()
    model.setAttr("ModelSense", gurobipy.GRB.MAXIMIZE)
    # add one variable x_ij for every edge ij of the complete graph
    variables = np.empty((n, n), dtype=object)
    var_type = gurobipy.GRB.BINARY if binary else gurobipy.GRB.CONTINUOUS
    for i in range(n):
        for j in range(i + 1, n):
            variables[i, j] = model.addVar(0, 1, cost[i, j], vtype=var_type, name=f"x_{{{i},{j}}}")
            variables[j, i] = variables[i, j]

    # add all triangle inequalities
    if add_triangle:
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(n):
                    if i == k or j == k:
                        continue
                    model.addConstr(get_triangle_inequality(variables, i, j, k), name="Triangle")

    return model, variables


def get_solution(variables):
    # query variable value of current solution
    solution = np.ones_like(variables, dtype=float)
    for i in range(variables.shape[0]):
        for j in range(variables.shape[1]):
            if i == j:
                continue
            solution[i, j] = variables[i, j].X
    return solution


def run_lp(cost, max_num_non_basic: int = 5):
    # get model with all variables
    model, variables = create_canonical_model(cost, binary=False, add_triangle=False)
    model.setParam('OutputFlag', 0)  # supress logging to console

    # dict for counting how often each constraint was non-basic
    nonbasic_count = {c: 0 for c in model.getConstrs()}

    # track objective over time
    runtime = []
    objective = []
    t_0 = time()

    print(f"{'Obj':>7} {'Time':>7} {'Tr-Ineq':>7} {'TP-Ineq':>7}")
    while True:
        # solve current LP
        model.optimize()
        # track objective over time
        t = time() - t_0
        runtime.append(t)
        objective.append(model.objVal)
        print(f"{model.objVal:>7.3f} {t:>7.3f}", end=" ")
        # break if timelimit is exceeded
        if t > T_LIMIT:
            print("OOT")
            break
        # query solution to current LP
        solution = get_solution(variables)

        # remove constraints that were non-basic for more than max_num_non_basic in a row:
        constraints_to_remove = []
        for c in model.getConstrs():
            if c.CBasis == 0:  # non-basic
                nonbasic_count[c] += 1
            else:
                nonbasic_count[c] = 0
            if nonbasic_count[c] >= max_num_non_basic:
                constraints_to_remove.append(c)
        model.remove(constraints_to_remove)
        for c in constraints_to_remove:
            del nonbasic_count[c]
        model.update()

        # find violated triangle inequalities and add them to model
        triangles = separate_triangle(solution)
        num_triangles = len(triangles)
        for i, j, k in triangles:
            inequality = get_triangle_inequality(variables, i, j, k)
            constr = model.addConstr(inequality, name="Triangle")
            nonbasic_count[constr] = 0

        # find violated two-partition inequalities and add them to model
        num_tp_inequalities = 0
        if num_triangles < cost.shape[0]:
            for A, B, violation in separate_two_partition(solution):
                num_tp_inequalities += 1
                inequality = get_tp_inequality(variables, A, B)
                constr = model.addConstr(inequality, name="TP")
                nonbasic_count[constr] = 0
        print(f"{num_triangles:>7} {num_tp_inequalities:>7}")

        # terminate if no violated inequalities were found
        if num_tp_inequalities + num_triangles == 0:
            break

    return runtime, objective


def run_ilp(cost, separate_tp, heuristics, lazy_triangle=False):
    # get model containing all variables and triangle inequalities if triangle inequalities are not separated lazy
    model, variables = create_canonical_model(cost, binary=True, add_triangle=not lazy_triangle)
    if lazy_triangle:
        model.Params.LazyConstraints = 1

    if not heuristics:
        # deactivate heuristics and cut generation
        model.setParam("Cuts", 0)
        model.setParam('PreCrush', 1)
        model.setParam('Heuristics', 0)

    # track objectives over time
    runtime = []
    incumbent = []
    bound = []

    def callback(_, where):
        if where == gurobipy.GRB.Callback.MIPSOL and lazy_triangle:
            # separate violated triangle inequalities and add them as lazy constraints
            solution = np.ones_like(variables, dtype=float)
            for i in range(variables.shape[0]):
                for j in range(variables.shape[1]):
                    if i == j:
                        continue
                    solution[i, j] = model.cbGetSolution(variables[i, j])
            triangles = separate_triangle(solution)
            for i, j, k in triangles:
                model.cbLazy(get_triangle_inequality(variables, i, j, k))
        if where == gurobipy.GRB.Callback.MIP:
            # track objectives over time
            runtime.append(time() - t_0)
            incumbent.append(model.cbGet(gurobipy.GRB.Callback.MIP_OBJBST))
            bound.append(model.cbGet(gurobipy.GRB.Callback.MIP_OBJBND))
        if where == gurobipy.GRB.Callback.MIPNODE and separate_tp:
            # separate two-partition inequalities and add them as user cuts
            status = model.cbGet(gurobipy.GRB.Callback.MIPNODE_STATUS)
            if status == gurobipy.GRB.OPTIMAL:
                solution = np.ones_like(variables, dtype=float)
                for i in range(variables.shape[0]):
                    for j in range(variables.shape[1]):
                        if i == j:
                            continue
                        solution[i, j] = model.cbGetNodeRel(variables[i, j])
                for A, B, violation in separate_two_partition(solution):
                    model.cbCut(get_tp_inequality(variables, A, B))

    # run optimization
    t_0 = time()
    model.setParam('TimeLimit', T_LIMIT)
    model.optimize(callback)
    return runtime, incumbent, bound


def get_random_costs(n: int):
    # generate symmetric cost matrix with costs sampled uniformly from [-1, 0, 1]
    cost = np.random.choice([-1, 0, 1], size=(n, n))
    cost[np.triu_indices_from(cost)] = cost.T[np.triu_indices_from(cost)]
    cost[np.diag_indices_from(cost)] = 0
    return cost


def main():
    n = 30  # number of nodes of clique partitioning problem instance
    heuristics = True  # flag whether to use heuristics and cuts in ILP solver

    # generate random costs for clique partitioning instance
    np.random.seed(0)
    cost = get_random_costs(n)

    # solve LP
    lp_time, lp_objective = run_lp(cost, 5)
    plt.plot(lp_time, lp_objective, label="LP Objective", linestyle="--", color="gray")

    # solve ILP with two-partition inequalities added as user cuts
    ilp_tp_time, ilp_tp_objective, ilp_tp_bound = run_ilp(cost, True, heuristics)
    plt.plot(ilp_tp_time, ilp_tp_objective, label="ILP Obj (TP)", color="tab:blue")
    plt.plot(ilp_tp_time, ilp_tp_bound, label="ILP Bound (TP)", linestyle="--", color="tab:blue")

    # solve ILP without adding user cuts
    ilp_time, ilp_objective, ilp_bound = run_ilp(cost, False, heuristics)
    plt.plot(ilp_time, ilp_objective, label="ILP Obj", color="tab:orange")
    plt.plot(ilp_time, ilp_bound, label="ILP Bound", linestyle="--", color="tab:orange")

    # plot results
    plt.ylim(2*lp_objective[-1] - lp_objective[1], lp_objective[1]*1.1)
    plt.legend()
    plt.suptitle(f"n = {n}, Heuristics = {heuristics}")
    plt.xlabel("Time [s]")
    plt.ylabel("Objective")
    plt.savefig(f"cp-{n}-{heuristics}.png")
    plt.show()


if __name__ == '__main__':
    main()
