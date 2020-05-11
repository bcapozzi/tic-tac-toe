from first_visit_mc_policy_iterator import FirstVisitMCPolicyIterator

def test1(n=1, epsilon=0.1):

    policyIterator = FirstVisitMCPolicyIterator()
    Q, policy = policyIterator.learn_tic_tac_toe(n,epsilon)

    return Q, policy
