#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/12/2012 5:36pm
# Modified by: Loc Trinh
import sys
import numpy as np
import robot
import graphics


def possible_states(obs):
    # Get all nearby states that can be reached by the observed state
    x,y = obs
    points = [obs]
    points.append((x+1,y))
    points.append((x-1,y))
    points.append((x,y+1))
    points.append((x,y-1))

    states = []
    for x,y in points:
        if 0 <= x <= robot.GRID_WIDTH - 1 and 0 <= y <= robot.GRID_HEIGHT - 1:
            actions = ['stay', 'left', 'right', 'up', 'down']

            if x == 0: # previous action could not have been to go right
                actions.remove('right')
            if x == robot.GRID_WIDTH - 1: # could not have gone left
                actions.remove('left')
            if y == 0: # could not have gone down
                actions.remove('down')
            if y == robot.GRID_HEIGHT - 1: # could not have gone up
                actions.remove('up')

            for action in actions:
                states.append((x,y,action))
    return states



#-----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states

    all_possible_observed_states: a list of possible observed states

    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state

    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state

    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    marginals = [None] * num_time_steps

    alphas = []
    print "Calculating forward messages..."
    for i in range(num_time_steps-1):
        X = robot.Distribution()
        for a in all_possible_hidden_states: 
            msum = 0
            if observations[i] == None:
                for b in all_possible_hidden_states:
                    if i == 0:
                        msum += prior_distribution[b] * transition_model(b)[a]
                    else:
                        msum += alphas[i-1][b] * transition_model(b)[a]
            else:
                for b in possible_states(observations[i]):
                    if i == 0:
                        msum += prior_distribution[b] * observation_model(b)[observations[i]] * transition_model(b)[a]
                    else:
                        msum += alphas[i-1][b] * observation_model(b)[observations[i]] * transition_model(b)[a]
            X[a] = msum
        alphas.append(X)

    betas = []
    print "Calculating backward messages..."
    for i in range(num_time_steps-1):
        X = robot.Distribution()
        for a in all_possible_hidden_states:
            msum = 0
            if observations[num_time_steps-1-i] is None:
                for b in all_possible_hidden_states:
                    if i == 0:
                        msum += transition_model(a)[b]
                    else:
                        msum += betas[i-1][b] * transition_model(a)[b]
            else:
                for b in possible_states(observations[num_time_steps-1-i]):
                    if i == 0:
                        msum += observation_model(b)[observations[num_time_steps-1-i]] * transition_model(a)[b]
                    else:
                        msum += betas[i-1][b] * observation_model(b)[observations[num_time_steps-1-i]] * transition_model(a)[b]
            X[a] = msum
        betas.append(X)
    betas = betas[::-1]
    
    for i in range(num_time_steps):
        X = robot.Distribution()
        for a in all_possible_hidden_states:
            if observations[i] is None:
                if i == 0:
                    X[a] = prior_distribution[a] * betas[i][a]
                elif i == num_time_steps-1:
                    X[a] = alphas[i-1][a]
                else:
                    X[a] = alphas[i][a] * betas[i][a]
            else:
                if i == 0:
                    X[a] = prior_distribution[a] * observation_model(a)[observations[i]] * betas[i][a]
                elif i == num_time_steps-1:
                    X[a] = alphas[i-1][a] * observation_model(a)[observations[i]]
                else:
                    X[a] = alphas[i][a] * observation_model(a)[observations[i]] * betas[i][a]
        X.renormalize()
        marginals[i] = X

    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    messages = []
    tracebacks = []
    for i in range(num_time_steps-1):
        X = robot.Distribution()
        Y = robot.Distribution()
        for a in all_possible_hidden_states: 
            m_max = 0
            max_state = None

            if observations[i] is not None:
                for b in possible_states(observations[i]):
                    if i == 0:
                        temp = prior_distribution[b] * observation_model(b)[observations[i]] * transition_model(b)[a]
                    else:
                        temp = messages[i-1][b] * observation_model(b)[observations[i]] * transition_model(b)[a]
                    if temp >= m_max:
                        m_max = temp
                        max_state = b
            else:
                for b in all_possible_hidden_states:
                    if i == 0:
                        temp = prior_distribution[b] * transition_model(b)[a]
                    else:
                        temp = messages[i-1][b] * transition_model(b)[a]
                    if temp >= m_max:
                        m_max = temp
                        max_state = b

            X[a] = m_max
            Y[a] = max_state
        messages.append(X)
        tracebacks.append(Y)

    x_hat = None
    m_max = 0
    for a in all_possible_hidden_states:
        temp = messages[98][a] * observation_model(a)[observations[99]]
        if temp > m_max:
            m_max = temp
            x_hat = a
    estimated_hidden_states[99] = x_hat
    
    for i in range(num_time_steps-2, -1, -1):
        x_hat = tracebacks[i][x_hat]
        estimated_hidden_states[i] = x_hat

    return estimated_hidden_states

def second_best(all_possible_hidden_states,
                all_possible_observed_states,
                prior_distribution,
                transition_model,
                observation_model,
                observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: This is for you to implement

    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    
    return estimated_hidden_states


#-----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(initial_distribution, transition_model, observation_model,
                  num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from a hidden Markov model given an initial
    # distribution, transition model, observation model, and number of time
    # steps, generate samples from the corresponding hidden Markov model
    hidden_states = []
    observations  = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state       = initial_distribution().sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state   = hidden_states[-1]
        new_state    = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1: # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


#-----------------------------------------------------------------------------
# Main
#

if __name__ == '__main__':
    # flags
    make_some_observations_missing = False
    use_graphics                   = True
    need_to_generate_data          = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(robot.initial_distribution,
                          robot.transition_model,
                          robot.observation_model,
                          num_time_steps,
                          make_some_observations_missing)

    all_possible_hidden_states   = robot.get_all_hidden_states()
    all_possible_observed_states = robot.get_all_observed_states()
    prior_distribution           = robot.initial_distribution()

    print 'Running forward-backward...'
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 robot.transition_model,
                                 robot.observation_model,
                                 observations)
    print

    timestep = num_time_steps - 1
    print "Most likely parts of marginal at time %d:" % (timestep)
    if marginals[timestep] is not None:
        print sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10]
    else:
        print '*No marginal computed*'
    print

    print 'Running Viterbi...'
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               robot.transition_model,
                               robot.observation_model,
                               observations)
    print

    print "Last 10 hidden states in the MAP estimate:"
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states[time_step] is None:
            print 'Missing'
        else:
            print estimated_states[time_step]
    print

    print 'Finding second-best MAP estimate...'
    estimated_states2 = second_best(all_possible_hidden_states,
                                    all_possible_observed_states,
                                    prior_distribution,
                                    robot.transition_model,
                                    robot.observation_model,
                                    observations)
    print

    print "Last 10 hidden states in the second-best MAP estimate:"
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print 'Missing'
        else:
            print estimated_states[time_step]
    print

    difference = 0
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
    print "Number of differences between MAP estimate and true hidden " + \
          "states:", difference

    difference = 0
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
    print "Number of differences between second-best MAP estimate and " + \
          "true hidden states:", difference

    difference = 0
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
    print "Number of differences between MAP and second-best MAP " + \
          "estimates:", difference

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()

