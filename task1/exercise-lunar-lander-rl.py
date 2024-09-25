import os
import imageio
import gymnasium as gym
import numpy as np
import torch
from agilerl.algorithms.td3 import TD3
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_off_policy import train_off_policy
from agilerl.utils.utils import create_population, make_vect_envs
from tqdm import trange

# Initial hyperparameters
INIT_HP = {
    "ALGO": "TD3",
    "POP_SIZE": 4,  # Population size
    "BATCH_SIZE": 128,  # Batch size
    "LR_ACTOR": 0.0001,  # Actor learning rate
    "LR_CRITIC": 0.001,  # Critic learning rate
    "O_U_NOISE": True,  # Ornstein-Uhlenbeck action noise
    "EXPL_NOISE": 0.1,  # Action noise scale
    "MEAN_NOISE": 0.0,  # Mean action noise
    "THETA": 0.15,  # Rate of mean reversion in OU noise
    "DT": 0.01,  # Timestep for OU noise
    "GAMMA": 0.99,  # Discount factor
    "MEMORY_SIZE": 100_000,  # Max memory buffer size
    "POLICY_FREQ": 2,  # Policy network update frequency
    "LEARN_STEP": 1,  # Learning frequency
    "TAU": 0.005,  # For soft update of target parameters
    # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
    "CHANNELS_LAST": False,  # Use with RGB states
    "EPISODES": 1000,  # Number of episodes to train for
    "EVO_EPOCHS": 20,  # Evolution frequency, i.e. evolve after every 20 episodes
    "TARGET_SCORE": 200.0,  # Target score that will beat the environment
    "EVO_LOOP": 3,  # Number of evaluation episodes
    'EVO_STEPS': 10_000,            # Evolution frequency
    'EVAL_STEPS': None,             # Evaluation steps
    #"MAX_STEPS": 100,  # Maximum number of steps an agent takes in an environment
    "MAX_STEPS": 10,  # Maximum number of steps an agent takes in an environment
    "TOURN_SIZE": 2,  # Tournament size
    "ELITISM": True,  # Elitism in tournament selection
    'EVAL_LOOP': 1,
    'LEARNING_DELAY': 1000,         # Steps before starting learning
}

# Mutation parameters
MUT_P = {
    # Mutation probabilities
    "NO_MUT": 0.4,  # No mutation
    "ARCH_MUT": 0.2,  # Architecture mutation
    "NEW_LAYER": 0.2,  # New layer mutation
    "PARAMS_MUT": 0.2,  # Network parameters mutation
    "ACT_MUT": 0.2,  # Activation layer mutation
    "RL_HP_MUT": 0.2,  # Learning HP mutation
    # Learning HPs to choose from
    "RL_HP_SELECTION": ["lr", "batch_size", "learn_step"],
    "MUT_SD": 0.1,  # Mutation strength
    "RAND_SEED": 42,  # Random seed
    # Define max and min limits for mutating RL hyperparams
    "MIN_LR": 0.0001,
    "MAX_LR": 0.01,
    "MIN_BATCH_SIZE": 8,
    "MAX_BATCH_SIZE": 1024,
    "MIN_LEARN_STEP": 1,
    "MAX_LEARN_STEP": 16,
}

def main(): 

    #create the environment
    num_envs=8
    env = make_vect_envs("LunarLanderContinuous-v2", num_envs=num_envs)  # Create environment

    try:
        state_dim = env.single_observation_space.n  # Discrete observation space
        one_hot = True  # Requires one-hot encoding
    except Exception:
        state_dim = env.single_observation_space.shape  # Continuous observation space
        one_hot = False  # Does not require one-hot encoding
    try:
        action_dim = env.single_action_space.n  # Discrete action space
    except Exception:
        action_dim = env.single_action_space.shape[0]  # Continuous action space

    INIT_HP["MAX_ACTION"] = float(env.single_action_space.high[0])
    INIT_HP["MIN_ACTION"] = float(env.single_action_space.low[0])

    if INIT_HP["CHANNELS_LAST"]:
        # Adjust dimensions for PyTorch API (C, H, W), for envs with RGB image states
        state_dim = (state_dim[2], state_dim[0], state_dim[1])

    # Set-up the device
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps' #for m1/m2 mac use 'mps'
    print (device)

    # Define the network configuration of a simple mlp with two hidden layers, each with 64 nodes
    net_config = {"arch": "mlp", "hidden_size": [64, 64]}

    # Define a population
    pop = create_population(
        algo="TD3",  # Algorithm
        state_dim=state_dim,  # State dimension
        action_dim=action_dim,  # Action dimension
        one_hot=one_hot,  # One-hot encoding
        net_config=net_config,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        population_size=INIT_HP["POP_SIZE"],  # Population size
        num_envs=num_envs,
        device=device,
        )

    field_names = ["state", "action", "reward", "next_state", "terminated"]
    memory = ReplayBuffer(
        memory_size=10_000,  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        device=device,
    )

    tournament = TournamentSelection(
        INIT_HP["TOURN_SIZE"],
        INIT_HP["ELITISM"],
        INIT_HP["POP_SIZE"],
        INIT_HP["EVAL_LOOP"],
    )

    mutations = Mutations(
        algo=INIT_HP["ALGO"],
        no_mutation=MUT_P["NO_MUT"],
        architecture=MUT_P["ARCH_MUT"],
        new_layer_prob=MUT_P["NEW_LAYER"],
        parameters=MUT_P["PARAMS_MUT"],
        activation=MUT_P["ACT_MUT"],
        rl_hp=MUT_P["RL_HP_MUT"],
        rl_hp_selection=MUT_P["RL_HP_SELECTION"],
        min_lr=MUT_P["MIN_LR"],
        max_lr=MUT_P["MAX_LR"],
        min_batch_size=MUT_P["MAX_BATCH_SIZE"],
        max_batch_size=MUT_P["MAX_BATCH_SIZE"],
        min_learn_step=MUT_P["MIN_LEARN_STEP"],
        max_learn_step=MUT_P["MAX_LEARN_STEP"],
        mutation_sd=MUT_P["MUT_SD"],
        arch=net_config["arch"],
        rand_seed=MUT_P["RAND_SEED"],
        device=device,
    )
    '''
    trained_pop, pop_fitnesses = train_off_policy(
        env=env,
        env_name="LunarLanderContinuous-v2",
        algo="TD3",
        pop=pop,
        memory=memory,
        INIT_HP=INIT_HP,
        MUT_P=MUT_P,
        swap_channels=INIT_HP["CHANNELS_LAST"],
        #INIT_HP["MAX_STEPS"]=INIT_HP["MAX_STEPS"],
        evo_steps=INIT_HP["EVO_STEPS"],
        eval_steps=INIT_HP["EVAL_STEPS"],
        eval_loop=INIT_HP["EVAL_LOOP"],
        learning_delay=INIT_HP["LEARNING_DELAY"],
        target=INIT_HP["TARGET_SCORE"],
        tournament=tournament,
        mutation=mutations,
        wb=False,  # Boolean flag to record run with Weights & Biases
        save_elite=True,  # Boolean flag to save the elite agent in the population
        elite_path="TD3_trained_agent.pt",
    )
    '''

    total_steps = 0

    # TRAINING LOOP
    print("Training...")
    pbar = trange(INIT_HP["MAX_STEPS"], unit="step")
    while np.less([agent.steps[-1] for agent in pop], INIT_HP["MAX_STEPS"]).all():
        pop_episode_scores = []
        for agent in pop:  # Loop through population
            state, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0

            for idx_step in range(INIT_HP["EVO_STEPS"] // num_envs):
                if INIT_HP["CHANNELS_LAST"]:
                    state = np.moveaxis(state, [-1], [-3])

                action = agent.get_action(state)  # Get next action from agent

                # Act in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                scores += np.array(reward)
                steps += num_envs
                total_steps += num_envs

                # Collect scores for completed episodes
                reset_noise_indices = []
                for idx, (d, t) in enumerate(zip(terminated, truncated)):
                    if d or t:
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])
                        scores[idx] = 0
                        reset_noise_indices.append(idx)
                agent.reset_action_noise(reset_noise_indices)

                # Save experience to replay buffer
                if INIT_HP["CHANNELS_LAST"]:
                    memory.save_to_memory(
                        state,
                        action,
                        reward,
                        np.moveaxis(next_state, [-1], [-3]),
                        terminated,
                        is_vectorised=True,
                    )
                else:
                    memory.save_to_memory(
                        state,
                        action,
                        reward,
                        next_state,
                        terminated,
                        is_vectorised=True,
                    )

                # Learn according to learning frequency
                if memory.counter > INIT_HP["LEARNING_DELAY"] and len(memory) >= agent.batch_size:
                    for _ in range(num_envs // agent.learn_step):
                        # Sample replay buffer
                        experiences = memory.sample(agent.batch_size)
                        # Learn according to agent's RL algorithm
                        agent.learn(experiences)

                state = next_state

            pbar.update(INIT_HP["EVO_STEPS"] // len(pop))
            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

        # Evaluate population
        fitnesses = [
            agent.test(
                env,
                swap_channels=INIT_HP["CHANNELS_LAST"],
                #INIT_HP["MAX_STEPS"]=INIT_HP["EVAL_STEPS"],
                loop=INIT_HP["EVAL_LOOP"],
            )
            for agent in pop
        ]
        mean_scores = [
            (
                np.mean(episode_scores)
                if len(episode_scores) > 0
                else "0 completed episodes"
            )
            for episode_scores in pop_episode_scores
        ]

        print(f"--- Global steps {total_steps} ---")
        print(f"Steps {[agent.steps[-1] for agent in pop]}")
        print(f"Scores: {mean_scores}")
        print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
        print(
            f'5 fitness avgs: {["%.2f"%np.mean(agent.fitness[-5:]) for agent in pop]}'
        )

        # Tournament selection and population mutation
        elite, pop = tournament.select(pop)
        pop = mutations.mutation(pop)

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    # Save the trained algorithm
    save_path = "TD3_trained_agent.pt"
    elite.save_checkpoint(save_path)

    pbar.close()
    env.close()


    #td3 = TD3.load_checkpoint(save_path, device=device)
    td3 = TD3.load(save_path, device=device)

    test_env = gym.make("LunarLanderContinuous-v2", render_mode="rgb_array")
    rewards = []
    frames = []
    testing_eps = 5 #change this
    max_testing_steps = 2000 #change this
    with torch.no_grad():
        for ep in range(testing_eps):
            state = test_env.reset()[0]  # Reset environment at start of episode
            score = 0

            for step in range(max_testing_steps):
                # If your state is an RGB image
                if INIT_HP["CHANNELS_LAST"]:
                    state = np.moveaxis(state, [-1], [-3])

                # Get next action from agent
                action, *_ = td3.get_action(state, training=False)

                # Save the frame for this step and append to frames list
                frame = test_env.render()
                frames.append(frame)

                # Take the action in the environment
                state, reward, terminated, truncated, _ = test_env.step(action)

                # Collect the score
                score += reward

                # Break if environment 0 is done or truncated
                if terminated or truncated:
                    print("terminated")
                    break

            # Collect and print episodic reward
            rewards.append(score)
            print("-" * 15, f"Episode: {ep}", "-" * 15)
            print("Episodic Reward: ", rewards[-1])

        print(rewards)

        test_env.close()

    frames = frames[::3]
    gif_path = "./videos/"
    os.makedirs(gif_path, exist_ok=True)
    imageio.mimwrite(
        os.path.join("./videos/", "td3_lunar_lander.gif"), frames, duration=50, loop=0
    )
    mean_fitness = np.mean(rewards)


if __name__ == '__main__':
    main()

