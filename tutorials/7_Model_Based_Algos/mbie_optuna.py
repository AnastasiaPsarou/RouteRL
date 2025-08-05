import numpy as np
from tqdm import tqdm
import os
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.integration.wandb import WeightsAndBiasesCallback
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

#from extended_mutation import MyExtendedMutation
from routerl import TrafficEnvironment
from routerl import DQN
from routerl import Rmax
from routerl import MBIE
from routerl import Keychain as kc

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#########################
## Hyperparameter setting
#########################

def objective(trial):
    beta = trial.suggest_float('beta', 0.1, 10, step=0.5)
    alpha = trial.suggest_float('alpha', 0.1, 10, step=0.5)


    new_machines_after_mutation = 10
    human_learning_episodes = 1
    training_episodes = 300
    testing_episodes = 100

    total_episodes = human_learning_episodes + training_episodes

    env_params = {
        "agent_parameters" : {
            "new_machines_after_mutation": new_machines_after_mutation,
            "agents_csv_file_name": "agents.csv",
            "num_agents" : 22,

            "human_parameters" :
            {
                "model" : "general_model",

                "noise_weight_agent" : 0,
                "noise_weight_path" : 0.8,
                "noise_weight_day" : 0.2,

                "beta" : -1,
                "beta_k_i_variability" : 0.1,
                "epsilon_i_variability" : 0.1,
                "epsilon_k_i_variability" : 0.1,
                "epsilon_k_i_t_variability" : 0.1,

                "greedy" : 0.9,
                "gamma_c" : 0.0,
                "gamma_u" : 0.0,
                "remember" : 1,

                "alpha_zero" : 0.8,
                "alphas" : [0.2]  
            },
            "machine_parameters" :
            {
                "behavior" : "selfish",
                "observation_type" : "previous_agents",
            }
        },
        "simulator_parameters" : {
            "network_name" : "two_route_yield",
            "sumo_type" : "sumo",
        },  
        "plotter_parameters" : {
            "phases" : [0, human_learning_episodes, int(training_episodes) + human_learning_episodes],
            "smooth_by" : 50,
            "phase_names" : [
                "Human learning", 
                "Mutation - Machine learning",
                "Testing phase"
            ],
            "plot_choices": "basic",
            "records_folder": "tutorials/7_Model_Based_Algos/records_mbie_two_route_net_optuna",
            "plots_folder": "tutorials/7_Model_Based_Algos/plots_mbie_two_route_net_optuna",
        },
        "path_generation_parameters":
        {
            "number_of_paths" : 2,
            "beta" : -1,
            "visualize_paths" : True
        }
    }

    env = TrafficEnvironment(seed=42, create_agents=False, create_paths=True, **env_params)

    print("Number of total agents is: ", len(env.all_agents), "\n")
    print("Number of human agents is: ", len(env.human_agents), "\n")
    print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")

    env.start()
    env.reset()

    for episode in range(human_learning_episodes):
        env.step()

    pre_mutation_agents = env.all_agents.copy()

    env.mutation_odd_id_agents()

    print("Number of total agents is: ", len(env.all_agents), "\n")
    print("Number of human agents is: ", len(env.human_agents), "\n")
    print("Number of machine agents (autonomous vehicles) is: ", len(env.machine_agents), "\n")


    for human in env.human_agents:
        human.default_action = 0


    machines = env.machine_agents.copy()
    mutated_humans = dict()
    for machine in machines:
        for human in pre_mutation_agents:
            if human.id == machine.id:
                mutated_humans[str(machine.id)] = human
                break

    free_flows = env.get_free_flow_times()
    for h_id, human in mutated_humans.items():
        initial_knowledge = free_flows[(human.origin, human.destination)]
        initial_knowledge = [0, 0]

        num_actions = env.action_space_size
        num_states = pow(len(env.all_agents), num_actions) 
        beta = beta
        alpha = alpha
        
        mutated_humans[h_id].model = UCB(num_states = num_states, num_actions = num_actions,
                                    num_agents = len(env.all_agents), alpha = alpha, beta=beta)


    ################
    ## Training loop
    ################

    print("Going to start training\n\n")

    pbar = tqdm(total=total_episodes, desc="Human learning")

    pbar.set_description("AV learning")
    for episode in range(training_episodes):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                obs = [{kc.AGENT_ID : int(agent), kc.TRAVEL_TIME : -reward}]
                last_action = mutated_humans[agent].last_action
                last_observation = mutated_humans[agent].last_obs

                mutated_humans[agent].learn(last_action, obs)
                action = None
            else:
                action = mutated_humans[agent].act(observation)
                mutated_humans[agent].last_action = action

            env.step(action)
        """if episode % 50 == 0:
            env.plot_results()"""
        pbar.update()


    ###############
    ## Testing loop
    ###############
    pbar.set_description("Testing")
    for episode in range(testing_episodes):
        env.reset()
        avg_ep_reward = 0

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            avg_ep_reward += reward

            if termination or truncation:
                action = None
            else:
                action = mutated_humans[agent].act(observation)
            env.step(action)
        pbar.update()

    pbar.close()

    env.plot_results()
    env.stop_simulation()
    return (-1 * avg_ep_reward / 10)


wandb_kwargs = {"project": "optuna-mbie-hyperparameter-tuning"}
wandbc = WeightsAndBiasesCallback(metric_name="beta", wandb_kwargs=wandb_kwargs)

search_space = {
    "beta": [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
}

# Optuna study setup
study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction="minimize")
study.optimize(objective, n_trials=20)

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")


plot_optimization_history(study).write_html("optimization_history.html")
plot_param_importances(study).write_html("param_importance.html")

