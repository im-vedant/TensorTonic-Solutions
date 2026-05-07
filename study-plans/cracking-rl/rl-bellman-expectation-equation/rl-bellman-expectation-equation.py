def bellman_expectation_backup(P, R, policy, gamma, V):
    """
    Returns: list of length S, V_new[s] rounded to 4 decimals
    """
    S = len(V)
    A = len(policy[0])
    V_new = [0.0] * S
    for s in range(S):
        v = 0.0
        for a in range(A):
            for sp in range(S):
                v += policy[s][a] * P[s][a][sp] * (R[s][a][sp] + gamma * V[sp])
        V_new[s] = round(v, 4)
    print(V_new)
    return V_new
