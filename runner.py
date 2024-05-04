import numpy as np
import time

from utils import AgentOutput


def generate_beta_schedule(beta_max, number_of_episodes, vdo_start=0.5):
    n_start = int(vdo_start * number_of_episodes) 
    beta_schedule = np.zeros(number_of_episodes+1)
    n_warm = number_of_episodes // 100
    beta_schedule[n_start: n_start + n_warm] = np.linspace(0, beta_max, num=n_warm)
    beta_schedule[n_start + n_warm:] = beta_max
    return beta_schedule

def run_experiment(
    env,
    agent,
    config,
) -> dict:
    
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
        # ep_transition_hist, ep_rewarded_hist, ep_stay_hist, ep_tr_state_hist = env.get_hists()
        # if config["eval_only"]:
        #     transition_hist += ep_transition_hist
        #     rewarded_hist += ep_rewarded_hist
        #     stay_hist += ep_stay_hist
        #     transition_state_hist += ep_tr_state_hist

        # success_count += sum(ep_rewarded_hist)
        # eps += len(ep_rewarded_hist)

        if hasattr(agent, 'sample_weights'):
            agent.sample_weights()

        # display progress 
        if i % config["display_eps"] == 0:
            avg_reward += (ep_reward - avg_reward) / i
            mins = (time.time() - start) / 60.
            print(f"\nep {i}/{config['num_episodes']}: avg ep reward = {avg_reward:8.4f}, time = {mins:8.2f}")
        
        if config["use_wandb"]:
            wandb.log(logdict)



# def run_2step_onpolicy(
#     env,
#     agent,
#     number_of_episodes,
#     use_default=False,
#     rand_stage2=False,
#     config["eval_only"]=False,
#     greedy=False,
#     anneal_rsp=False,
#     display_eps=1,
#     print_every=False,
#     verbose=True,
#     beta_max=1.0,
#     vdo_start=0.5,
#     default_start=0.25,
#     reward_scale=1.0):
#     """
#     run an experiment
#     """
#     rewarded_hist, return_hist, stay_hist, transition_hist = [], [], [], []
#     action_hist, transition_state_hist = [], []
#     default_losses, vdo_kls, policy_kls = [], [], []
#     rate100 = 0
#     eps = 0
#     max_t = 400
#     #if display_eps is not None:
#     success_count, ep_count = 0, 0
#     if config["eval_only"] and hasattr(agent, "eval_only"):
#         print ('\neval_onlyuation mode.')
#         agent.eval_only()


#     n_start = int(vdo_start * number_of_episodes) 
#     beta_schedule = np.zeros(number_of_episodes+1)
#     n_warm = number_of_episodes // 100
#     beta_schedule[n_start: n_start + n_warm] = np.linspace(0, beta_max, num=n_warm)
#     beta_schedule[n_start + n_warm:] = beta_max

#     try:
#         action = agent.initial_action()
#     except AttributeError:
#         action = 0

#     start = time()
#     for i in range(1, number_of_episodes+1):
#         beta = beta_schedule[i-1]
#         update_default = i > int(default_start * number_of_episodes) 

#         # adjust reward switching  
#         if anneal_rsp and not config["eval_only"]:
#             if i < 2000:
#                 env._r_switch_prob = 0.0 
#             else:
#                 env._r_switch_prob = 0.025 * sigmoid(0.0008 * i - 1)

#         ep_count += 1
#         done = False

#         obs = env.reset()

#         action, logprob, pi, v, pi0 = agent.get_action(obs, 0, 0.0, greedy=greedy, use_default=use_default, get_policies=True)
#         if not config["eval_only"]:
#             agent.control_buffer.append(AgentOutput(logprob, pi, v, pi0))
#         z = 0
#         traj = [env.obs_to_state(obs)]
#         if print_every: print (f"0: state = {env.obs_to_state(obs)}, action = {action}")

#         t = 1
#         while not done and t <= max_t:
#             # effect of action in env
#             obs, reward, done, _ = env.step(action)
#             reward *= reward_scale
#             if rand_stage2 and env.in_stage1:
#                 obs = np.random.permutation(obs)
#             traj.append(env.obs_to_state(obs))

#             agent.rewards.append(reward)
            
#             # agent takes next step
#             action, logprob, pi, v, pi0 = agent.get_action(obs, action, reward, greedy=greedy, use_default=use_default, get_policies=True) 
#             if not config["eval_only"]: agent.control_buffer.append(AgentOutput(logprob, pi, v, pi0))

#             if env.in_stage1:
#                 action_hist.append(action)
             
#             z += (agent.gamma ** (t-1)) * reward
#             t += 1

#         # update agent
#         if config["eval_only"]:
#             del agent.rewards[:]
#             del agent.control_buffer[:]
#             if agent.recurrent: agent.detach_state()
#         else: 
#             default_losses_step, policy_kl_step, vdo_kl_step = agent.update(beta=beta, update_default=update_default)
#             default_losses.append(default_losses_step)
#             policy_kls.append(policy_kl_step)
#             vdo_kls.append(vdo_kl_step)

#         return_hist.append(z)
#         ep_transition_hist, ep_rewarded_hist, ep_stay_hist, ep_tr_state_hist = env.get_hists()
#         if config["eval_only"]:
#             transition_hist += ep_transition_hist
#             rewarded_hist += ep_rewarded_hist
#             stay_hist += ep_stay_hist
#             transition_state_hist += ep_tr_state_hist

#         success_count += sum(ep_rewarded_hist)
#         eps += len(ep_rewarded_hist)

#         if hasattr(agent, 'sample_weights'):
#             agent.sample_weights()

#         # display progress 
#         if display_eps is not None and i % display_eps == 0:
#             rate = success_count / (eps + 1e-4) 
#             mean_r = np.mean(return_hist)
#             rate100 = rate
#             mins = (time() - start) / 60.

#             if len(rewarded_hist) > 100:
#                 rate100 = np.mean(np.array(rewarded_hist)[-100:])
#             if verbose and (i % 1000 == 0 or i == 1): #flush_
#                 flush_print(f"\nep {i}/{number_of_episodes}: mean return = {mean_r:8.4f}, rate = {rate:8.4f}, rate100 = {rate100:8.2f}, rsp = {env._r_switch_prob}, time = {mins:8.2f}")


#     #transition_state_stay_hist = [1]
#     #for trial in range(1, len(transition_state_hist)):
#     #    transition_state_stay_hist.append(int(transition_state_hist[trial] == transition_state_hist[trial-1]))

#     results = {
#       "return hist": return_hist,
#       "rewarded hist": rewarded_hist,
#       "transition hist": transition_hist,
#       "stay hist": stay_hist,
#       "transition state stay hist": transition_state_hist,
#       "action hist": action_hist,
#       "trajectory": traj,
#       #"w hist": None if not hasattr(agent, "w_hist") else agent.w_hist,
#       "default_losses": default_losses,
#       "vdo_kls": vdo_kls,
#       "policy_kls": policy_kls
#     }

#     return results 