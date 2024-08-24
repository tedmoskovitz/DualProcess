from collections import defaultdict
import numpy as np
import torch

def eval_agent(agent, env, config):
    if config["env_name"] == "FourRooms":
        return eval_fourrooms(agent, env, config)
    elif config["env_name"] == "Stroop":
        return eval_stroop(agent, env, config)
    elif config["env_name"] == "TwoStep":
        return eval_twostep(agent, env, config)
    elif config["env_name"] == "ContingencyDegradation":
        return eval_contingency_degradation(agent, env, config)
    else:
        raise NotImplementedError

@torch.no_grad()
def eval_fourrooms(agent, env, config):

    def run_eval_eps(use_default, common_goal):
        successes = []
        env.p_common_goal = float(common_goal)

        # get control policy success rate
        for i in range(1, config['num_eval_eps']+1):
            env.update(i)
            done = False
            obs = env.reset()

            action = agent.get_action(obs, 0, 0.0, greedy=config["greedy"], get_policies=False)
            ep_reward = 0

            t = 1
            while t <= env.max_steps_per_episode:
                # effect of action in env
                obs, reward, done, _ = env.step(action)
                ep_reward += reward
                if done:
                    break
                # agent takes next step
                action = agent.get_action(
                    obs, action, reward,
                    greedy=config["greedy"],
                    use_default=use_default,
                    get_policies=False) 

                t += 1

            successes.append(int(ep_reward > 0))

            # update agent
            agent.clear_buffers()

            if hasattr(agent, 'sample_weights'):
                agent.sample_weights()

        return sum(successes) / len(successes)

    agent.eval()
    p_common_goal = env.p_common_goal
    control_common_rate = run_eval_eps(use_default=False, common_goal=True)
    default_common_rate = run_eval_eps(use_default=True, common_goal=True)
    control_rare_rate = run_eval_eps(use_default=False, common_goal=False)
    default_rare_rate = run_eval_eps(use_default=True, common_goal=False)
    agent.train()
    env.p_common_goal = p_common_goal

    return {"eval/control_common_rate": control_common_rate,
            "eval/default_common_rate": default_common_rate,
            "eval/control_rare_rate": control_rare_rate,
            "eval/default_rare_rate": default_rare_rate}


@torch.no_grad()
def eval_stroop(agent, env, config):
    agent.eval()
    stim_types = ["wr_congruent", "cn_congruent", "wr_conflict", "cn_conflict"]
    out = defaultdict(float)
    for stim_type in stim_types:
        task_str, type_str = stim_type.split("_")
        task = env.WR if task_str == "wr" else env.CN
        word = env.RED
        color = env.RED if type_str == "congruent" else env.BLUE
        stim = np.array([color, word, task])
        for a_tm1 in [0, 1]:
            for r_tm1 in [0, 1]:
                _, _, pi, _, pi0, _ = agent.get_action(
                    stim, a_tm1, r_tm1, greedy=False, use_default=False, get_policies=True)
                out["eval/pi_" + stim_type] += pi[env.RED].detach().item()
                out["eval/pi0_" + stim_type] += pi0[env.RED].detach().item()

    for key, value in out.items():
        out[key] = value / 4
    agent.train()
    return out



@torch.no_grad()
def eval_twostep(agent, env, config):

    def run_eval_eps(use_default):
        trial_types = ["common_rewarded", "common_unrewarded", "uncommon_rewarded", "uncommon_unrewarded"]
        results = {trial_type: [] for trial_type in trial_types}

        # get control policy success rate
        for _ in range(1, config['num_eval_eps']+1):
            done = False
            obs = env.reset()

            action = agent.get_action(obs, 0, 0.0, greedy=config["greedy"], get_policies=False)

            t = 1
            while t <= env.max_steps_per_episode:
                # effect of action in env
                obs, reward, done, info = env.step(action)
                if info:
                    results[info["trial_type"]].append(float(info["stay"]))
                if done:
                    break
                # agent takes next step
                action = agent.get_action(
                    obs, action, reward,
                    greedy=config["greedy"],
                    use_default=use_default,
                    get_policies=False) 

                t += 1


            # update agent
            agent.clear_buffers()

            if hasattr(agent, 'sample_weights'):
                agent.sample_weights()

        prefix = "default" if use_default else "control"
        for trial_type in trial_types:
            stays = results[trial_type]
            if len(stays) > 0:
                results[f"eval/{prefix}_{trial_type}"] = sum(stays) / len(stays)
            else:
                results[f"eval/{prefix}_{trial_type}"] = -1
            del results[trial_type]

        return results

    agent.eval()
    control_results = run_eval_eps(use_default=False)
    default_results = run_eval_eps(use_default=True)
    agent.train()
    
    out = {**control_results, **default_results}
    return out


@torch.no_grad()
def eval_contingency_degradation(agent, env, config):
    agent.eval()
    out = defaultdict(float)
    stim = np.array(0.1)
    for r_tm1 in [0, 0.1, 1.1]:
        for a_tm1 in [env.LEVER, env.WITHOLD]:
            _, _, pi, _, _, _ = agent.get_action(
                stim, a_tm1, r_tm1, greedy=False, use_default=False, get_policies=True)
            out["eval/p_lever_press"] += pi[env.LEVER].detach().item()

    for key, value in out.items():
        out[key] = value / 6

    agent.train()
    return out

