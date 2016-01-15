class Distribution(dict):
    def __missing__(self, key):
        # if the key is missing, return probability 0
        return 0

    def renormalize(self):
        normalization_constant = sum(self.itervalues())
        for key in self.iterkeys():
            self[key] /= normalization_constant

    def get_mode(self):
        maximum = -1
        arg_max = None

        for key in self.iterkeys():
            if self[key] > maximum:
                arg_max = key
                maximum = self[key]

        return arg_max

    def sample(self):
        keys  = []
        probs = []
        for key, prob in self.iteritems():
            keys.append(key)
            probs.append(prob)

        rand_idx = np.where(np.random.multinomial(1, probs))[0][0]
        return keys[rand_idx]




def initial_distribution():
    # returns a Distribution for the initial hidden state
    prior = Distribution()
    for i in range(1,4):
        prior[i] = 1./3
    return prior

def transition_model(state):
    next_states = Distribution()
    if state == 1:
        next_states[1] = .25
        next_states[2] = .75
    elif state == 2:
        next_states[2] = .25
        next_states[3] = .75
    else:
        next_states[3] = 1
    return next_states

def observation_model(state):
    observed_states = Distribution()
    if state == 1:
        observed_states["hot"] = 1
    elif state == 2:
        observed_states["cold"] = 1
    else:
        observed_states["hot"] = 1
    return observed_states

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    num_time_steps = len(observations)
    marginals = [None] * num_time_steps

    alphas = []
    for i in range(num_time_steps-1):
        X = Distribution()
        for a in all_possible_hidden_states:
            msum = 0
            for b in all_possible_hidden_states:
                if i == 0:
                    msum += prior_distribution[b] * observation_model(b)[observations[i]] * transition_model(b)[a]
                else:
                    msum += alphas[i-1][b] * observation_model(b)[observations[i]] * transition_model(b)[a]
            X[a] = msum
        alphas.append(X)

    betas = []
    for i in range(num_time_steps-1):
        X = Distribution()
        for a in all_possible_hidden_states:
            msum = 0
            for b in all_possible_hidden_states:
                if i == 0:
                    msum += observation_model(b)[observations[num_time_steps-1-i]] * transition_model(a)[b]
                else:
                    msum += betas[i-1][b] * observation_model(b)[observations[num_time_steps-1-i]] * transition_model(a)[b]
            X[a] = msum
        betas.append(X)
    betas = betas[::-1]
    
    for i in range(num_time_steps):
        if i == 0:
            X = Distribution()
            for a in all_possible_hidden_states:
                X[a] = prior_distribution[a] * observation_model(a)[observations[i]] * betas[i][a]
            X.renormalize()
            marginals[i] = X
        elif i == num_time_steps-1:
            X = Distribution()
            for a in all_possible_hidden_states:
                X[a] = alphas[i-1][a] * observation_model(a)[observations[i]]
            X.renormalize()
            marginals[i] = X
        else:
            X = Distribution()
            for a in all_possible_hidden_states:
                X[a] = alphas[i][a] * observation_model(a)[observations[i]] * betas[i][a]
            X.renormalize()
            marginals[i] = X

    return marginals


print forward_backward([1,2,3], ['hot','cold'], initial_distribution(), transition_model, observation_model, ['hot','cold','hot'])

[((11, 0, 'stay'), 0.8102633355840648), ((11, 0, 'right'), 0.17960837272113436), ((10, 1, 'down'), 0.01012829169480081), ((11, 7, 'stay'), 0.0), ((10, 6, 'stay'), 0.0), ((3, 0, 'left'), 0.0), ((2, 4, 'down'), 0.0), ((7, 6, 'stay'), 0.0), ((10, 0, 'right'), 0.0), ((3, 6, 'up'), 0.0)]
