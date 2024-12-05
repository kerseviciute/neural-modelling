import numpy as np
import matplotlib.pyplot as plt

# define maze
maze = np.zeros((9, 13))

# place walls
maze[2, 6:10] = 1
maze[-3, 6:10] = 1
maze[2:-3, 6] = 1

# define start
start = (5, 7)

# define goal (we abuse function scoping a bit here, later we will change the goal, which will automatically change the goal in our actor critic as well)
goal = (1, 1)
goal_state = goal[0]*maze.shape[1] + goal[1]
goal_value = 10

def plot_maze(maze):
    plt.imshow(maze, cmap='binary')

    # draw thin grid
    for i in range(maze.shape[0]):
        plt.plot([-0.5, maze.shape[1]-0.5], [i-0.5, i-0.5], c='gray', lw=0.5)
    for i in range(maze.shape[1]):
        plt.plot([i-0.5, i-0.5], [-0.5, maze.shape[0]-0.5], c='gray', lw=0.5)

    plt.xticks([])
    plt.yticks([])


def compute_transition_matrix(maze):
    # for a given maze, compute the transition matrix from any state to any other state under a random walk policy
    # (you will need to think of a good way to map any 2D grid coordinates onto a single number for this)

    # create a matrix over all state pairs
    transitions = np.zeros((maze.size, maze.size))

    # iterate over all states, filling in the transition probabilities to all other states on the next step (only one step into the future)
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            # check if state is valid
            if maze[i, j] == 0:
                # iterate over all possible moves
                for move in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    new_i, new_j = i + move[0], j + move[1]
                    # check if new state is valid
                    if new_i >= 0 and new_i < maze.shape[0] and new_j >= 0 and new_j < maze.shape[1] and maze[new_i, new_j] == 0:
                        transitions[i*maze.shape[1] + j, new_i*maze.shape[1] + new_j] = 1
    
    # normalize transitions
    transitions /= transitions.sum(axis=1, keepdims=True)

    # remove NaNs
    transitions[np.isnan(transitions)] = 0

    return transitions

transitions = compute_transition_matrix(maze)


def analytical_sr(transitions, gamma):
    return np.linalg.inv(np.eye(transitions.shape[0]) - gamma * transitions.T)

i, j = start
# compute the SR for all states, based on the transition matrix
# note that we use a lower discounting here, to keep the SR more local
analytical_sr = analytical_sr(transitions, 0.8).T



# Part 1: program an actor critic algorithm to navigate the maze, using a table of action propensities M with softmax action selection as actor, and a learned state-value function as critic

def softmax(x):
    TODO

def normal_start():
    # suggested encoding of 2D location onto states
    state = start[0]*maze.shape[1] + start[1]
    return state

def actor_critic(state_representation, n_steps, alpha, gamma, n_episodes, update_sr=False, start_func=normal_start, v_init=0):
    # implement the actor-critic algorithm to learn to navigate the maze
    # state_representation is a matrix of size n_states by n_states, giving us the representation for each, which is either a 1-hot vector
    # # (so e.g. state_representation[15] is a vector of size n_states which is 0 everwhere, except 1 at index 15), or the SR for each state
    # n_steps is the number of actions in each episode before it gets cut off, an episode also ends when the agent reaches the goal
    # alpha and gamma are the learning rate and discount factor respectively
    # n_episodes is the number of episodes to train the agent
    # update_sr is for exercise part 3, when you want to update the SR after each episode
    # start_func allows you to specify a different starting state, if desired

    # initialize M-table
    M = np.zeros(TODO)

    # initialize state-value function
    V_weights = TODO

    earned_rewards = TODO

    # iterate over episodes
    for _ in range(n_episodes):
        # initializations
        TODO

        # go until goal is reached
        for t in range(n_steps):

            # act and learn (update both M and V_weights)


            # check if goal is reached
            if (i, j) == goal:
                TODO
                break

        if update_sr:
            TODO

    return M, V_weights, earned_rewards


# One part to the solution of exercise part 3, if you want to update the SR after each episode
def learn_from_traj(succ_repr, trajectory, gamma=0.98, alpha=0.05):
    # Write a function to update a given successor representation (for the state at which the trajectory starts) using an example trajectory
    # using discount factor gamma and learning rate alpha

    observed = np.zeros_like(succ_repr)

    for i, state in enumerate(trajectory):
        observed[state] += gamma ** i

    succ_repr += alpha * (observed - succ_repr)

    # return the updated successor representation
    return succ_repr


# Part 1

M, V, earned_rewards = actor_critic(TODO, n_steps=300, alpha=0.05, gamma=0.99, n_episodes=1000)

# plot state-value function
plot_maze(maze)
plt.imshow(V.reshape(TODO), cmap='hot')
plt.show()

plt.plot(earned_rewards)
plt.show()


# Part 2, now the same for an SR representation

TODO

# Part 3

TODO
def random_start():
    # define yourself a function to return a random (non-wall) starting state to pass into the actor_critic function
    return TODO

M, V, earned_rewards = actor_critic(TODO, 300, 0.05, 0.99, 1000, update_sr=True, start_func=random_start)

# plot the SR of some states after this learning, also anything else you want
TODO

# Part 4

TODO
goal = (5, 5)
goal_state = goal[0]*maze.shape[1] + goal[1]
for i in range(20):

    # run with random walk SR
    M, V, earned_rewards_clamped = actor_critic(TODO, 300, 0.05, 0.99, 400)
    TODO

    # run with updated SR
    M, V, earned_rewards_relearned = actor_critic(TODO, 300, 0.05, 0.99, 400)
    TODO

# plot the performance averages of the two types of learners
TODO


# Part 5

# reset goal
goal = (1, 1)
goal_state = goal[0]*maze.shape[1] + goal[1]

# run some learners with different value weight w initializations

TODO
for v_inits in [TODO]:
    TODO
    for i in range(12):

        M, V, earned_rewards = actor_critic(TODO, 300, 0.05, 0.99, 400)
        TODO
        M, V, earned_rewards = actor_critic(TODO, 300, 0.05, 0.99, 400)
        TODO

# plot the resulting learning curves


