import numpy as np
import time

from utils import AgentOutput


def generate_beta_schedule(beta_max: float, number_of_episodes: int, vdo_start: float = 0.5) -> np.ndarray:
    """
    Generate a beta schedule for the experiment.

    Args:
        beta_max (float): Maximum beta value.
        number_of_episodes (int): Total number of episodes.
        vdo_start (float): Fraction of episodes before starting VDO. Default is 0.5.

    Returns:
        np.ndarray: Beta schedule for all episodes.
    """
    n_start = int(vdo_start * number_of_episodes) 
    beta_schedule = np.zeros(number_of_episodes+1)
    n_warm = number_of_episodes // 100
    beta_schedule[n_start: n_start + n_warm] = np.linspace(0, beta_max, num=n_warm)
    beta_schedule[n_start + n_warm:] = beta_max
    return beta_schedule

def run_experiment(
    env,
    agent,
    config: dict,
) -> dict:
    """
    Run the experiment with given environment, agent, and configuration.

    Args:
        env: The environment to run the experiment in.
        agent: The agent to use in the experiment.
        config (dict): Configuration parameters for the experiment.

    Returns:
        dict: Results of the experiment.
    """
    
    if config["eval_only"]: agent.eval_only()
    if config["use_wandb"]:
        import wandb
        wandb.init(project="TwoStep", config=config)

    action = 0 if not hasattr(agent, "initial_action") else agent.initial_action()
    avg_reward = 0.0
    beta_schedule = generate_beta_schedule(
        config["beta_max"], config["num_episodes"], vdo_start=config["vdo_start"])
    start = time.time()

    for i in range(1, config["num_episodes"]+1):
        beta = beta_schedule[i-1]
        update_default = i > int(config["default_start"] * config["num_episodes"]) 

        env.update(i)
        done = False
        obs = env.reset()

        action, logprob, pi, v, pi0 = agent.get_action(obs, 0, 0.0, greedy=config["greedy"], get_policies=True)
        if not config["eval_only"]:
            agent.control_buffer.append(AgentOutput(logprob, pi, v, pi0))
        ep_reward = 0
        traj = [env.obs_to_state(obs)]

        t = 1
        while not done and t <= env.max_steps_per_episode:
            # effect of action in env
            obs, reward, done, _ = env.step(action)
            traj.append(env.obs_to_state(obs))
            agent.rewards.append(reward)
            # agent takes next step
            action, logprob, pi, v, pi0 = agent.get_action(
                obs, action, reward, greedy=config["greedy"], use_default=config["use_default"], get_policies=True) 
            if not config["eval_only"]: agent.control_buffer.append(AgentOutput(logprob, pi, v, pi0))

            # if env.in_stage1:
            #     action_hist.append(action)
             
            ep_reward += reward
            t += 1

        # update agent
        if config["eval_only"]:
            agent.clear_buffers()
            logdict = {}
        else: 
            default_losses_step, policy_kl_step, vdo_kl_step = agent.update(beta=beta, update_default=update_default)
            logdict = {"default_loss": default_losses_step, "policy_kl": policy_kl_step, "vdo_kl": vdo_kl_step}
        

        logdict['ep_reward'] = ep_reward
        if hasattr(agent, 'sample_weights'):
            agent.sample_weights()

        # display progress 
        if i % config["display_eps"] == 0:
            avg_reward += (ep_reward - avg_reward) / i
            mins = (time.time() - start) / 60.
            print(f"\nep {i}/{config['num_episodes']}: avg ep reward = {avg_reward:8.4f}, time = {mins:8.2f}")
        
        if config["use_wandb"]:
            wandb.log(logdict)

