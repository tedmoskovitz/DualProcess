
from configurator import envs
from evals import eval_agent
from mdlc import MDLCAgent
import pdb
import time
from utils import (
    generate_beta_schedule,
    AgentOutput,
    set_seed_everywhere,
    entropy)

# experiment params
# use_wandb = True
# num_episodes = 30_000
# display_eps = 50
# eval_every = 500
# num_eval_eps = 50
# batch_size = 1
# seed = 1
# environment params
env_name = 'TwoStep'

# --------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("__") and not callable(v) and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

# @profile
def run(config):
    env_class, env_kwargs, exp_kwargs, agent_kwargs = envs[config['env_name']]
    config.update(exp_kwargs)

    set_seed_everywhere(config['seed'])
    env = env_class(**env_kwargs)
    config.update(env_kwargs)
    config['obs_dim'] = env.obs_dim
    config['num_actions'] = env.num_actions
    if hasattr(env, "max_steps_per_episode"): config["max_steps_per_episode"] = env.max_steps_per_episode
    config.update(agent_kwargs)
    agent = MDLCAgent(config)
    print(config)

    if config["use_wandb"]:
        import wandb
        wandb.init(project="DP_" + env_name, config=config)

    action = 0 if not hasattr(agent, "initial_action") else agent.initial_action()
    avg_reward = 0.0
    beta_schedule = generate_beta_schedule(
        config["beta_max"], config["num_episodes"], vdo_start=config["vdo_start"])
    start = time.time()
    ep_rewards = []

    for i in range(1, config["num_episodes"]+1):
        beta = beta_schedule[i-1]
        update_default = i > int(config["default_start"] * config["num_episodes"]) 

        env.update(i)
        done = False
        obs = env.reset()

        action, log_prob, pi, v, pi0, rt = agent.get_action(obs, 0, 0.0, greedy=config["greedy"], get_policies=True)
        trajectory_outs = [AgentOutput(log_prob, pi, v, pi0)]
        trajectory_rewards, pi_ents, pi0_ents = [], [entropy(pi)], [entropy(pi0)]
        ep_reward = 0
        traj = [env.obs_to_state(obs)]

        t = 1
        while t <= env.max_steps_per_episode:
            obs, reward, done, _ = env.step(action)
            traj.append(env.obs_to_state(obs))
            trajectory_rewards.append(reward)
            ep_reward += reward
            if done:
                break
            # agent takes next step
            action, log_prob, pi, v, pi0, rt = agent.get_action(
                obs, action, reward, greedy=config["greedy"], use_default=config["use_default"], get_policies=True) 

            trajectory_outs.append(AgentOutput(log_prob, pi, v, pi0))
            pi_ents.append(entropy(pi))
            pi0_ents.append(entropy(pi0))
            t += 1

        agent.control_buffer.add(trajectory_outs, trajectory_rewards)

        ep_rewards.append(ep_reward)
        logdict = {"rt": rt+1,
                   "beta": beta,
                   "pi_entropy": sum(pi_ents)/len(pi_ents),
                   "pi0_entropy": sum(pi0_ents)/len(pi0_ents)}

        if i % config['batch_size'] == 0:
            update_metrics = agent.update(beta=beta, update_default=update_default)
            logdict.update(update_metrics)
        
        logdict['ep_reward'] = ep_reward
        if hasattr(agent, 'sample_weights'):
            agent.sample_weights()

        if i % config['eval_every'] == 0:
            eval_stats = eval_agent(agent, env, config)
            print("evaluation results:", eval_stats)
            logdict.update(eval_stats)

        # display progress 
        if i % config["display_eps"] == 0:
            avg_reward = sum(ep_rewards) / len(ep_rewards)
            mins = (time.time() - start) / 60.
            print(f"ep {i}/{config['num_episodes']}: avg ep reward = {avg_reward:8.4f}, time = {mins:8.2f} min")
        
        if config["use_wandb"]:
            wandb.log(logdict)

if __name__ == "__main__":
    run(config)

