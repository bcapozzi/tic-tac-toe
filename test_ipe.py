from first_visit_mc_policy_iterator import FirstVisitMCPolicyIterator

def test1(n=1):

    policyIterator = FirstVisitMCPolicyIterator()
    Q, policy = policyIterator.learn_tic_tac_toe(n)

    return Q, policy
