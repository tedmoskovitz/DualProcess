from envs import (
    FourRoomsTask,
    TwoStepTask,
    StroopTask,
    DriftingBanditTask,
    OmissionDevaluationTask,)


# FourRooms Task
fourrooms_env_kwargs = dict(
    start_state=-1,
    p_common_goal=0.7,
    max_steps_per_episode=50,
)

fourrooms_exp_kwargs = dict(
    use_wandb=True,
    num_episodes=30_000,
    display_eps=50,
    eval_every=500,
    num_eval_eps=50,
    batch_size=1,
    seed=2,
)

fourrooms_agent_kwargs = dict(
    hidden_size=48,
    alpha=0.2,
    lr=7e-4,
    use_a_tm1=True,
    gamma=0.99,
    value_loss_wt=0.2,
    vdo=True,
    asymmetry=True,
    default_critic=False,
    rt_budget=1,
    rt_thresholds=0.5,
    default_start=0.0,
    beta_max=10.0,
    vdo_start=0.0,
    greedy=False,
    use_default=False,
)


# Stroop Task
stroop_env_kwargs = dict(
    max_trials=40_000,
    flatten=False,
    training=False,
    uniform=False,
    verbose=False,
)

stroop_exp_kwargs = dict(
    use_wandb=True,
    num_episodes=10_000,
    display_eps=50,
    eval_every=500,
    batch_size=1,
    seed=2,
)

stroop_agent_kwargs = dict(
    hidden_size=16,
    alpha=0.2,
    lr=1e-3,
    use_a_tm1=True,
    gamma=0.99,
    value_loss_wt=0.05,
    vdo=True,
    asymmetry=True,
    default_critic=False,
    rt_budget=3,
    rt_thresholds=0.5,
    default_start=0.0,
    beta_max=0.005,
    vdo_start=0.0,
    greedy=False,
    use_default=False,
)

# Two-Step Task
twostep_env_kwargs = dict(
    trials_per_episode=100,
    transition_reversals=False,
    tr_prob=0.9,
    r_prob=0.9,
    tr_switch_prob=0.025,
    r_switch_prob=0.025,
    anneal_r_switch_prob=False,
    reward_scale=1.0,
    show_transition_feature=False
)

twostep_exp_kwargs = dict(
    use_wandb=False,
    num_episodes=5_000,
    display_eps=50,
    eval_every=500,
    num_eval_eps=50,
    batch_size=1,
    seed=2,
)

twostep_agent_kwargs = dict(
    hidden_size=48,
    alpha=0.2,
    lr=1e-3,
    use_a_tm1=True,
    gamma=0.99,
    value_loss_wt=0.05,
    vdo=True,
    asymmetry=True,
    default_critic=False,
    rt_budget=1,
    rt_thresholds=0.5,
    default_start=0.0,
    beta_max=1.0,
    vdo_start=0.0,
    greedy=False,
    use_default=False,
)


# Drifting Bandit Task
driftingbandit_env_kwargs = dict(
    drift_std=0.15
)

driftingbandit_exp_kwargs = dict(
    use_wandb=False,
    num_episodes=5_000,
    display_eps=50,
    eval_every=500,
    num_eval_eps=50,
    batch_size=1,
    seed=2,
)

driftingbandit_agent_kwargs = dict(
    hidden_size=8,
    alpha=0.0,#0.2,
    lr=5e-3, #7e-4,
    use_a_tm1=True,
    gamma=0.99,#1.0,
    value_loss_wt=0.05,
    vdo=True,
    asymmetry=True,
    default_critic=False,
    rt_budget=3,
    rt_thresholds=0.5,
)


# Contingency Degradation
cd_env_kwargs = dict(
    omission=True,
    num_training_trials=2000,
)

cd_exp_kwargs = dict(
    use_wandb=False,
    num_episodes=2750, # == num_training_trials + 750
    display_eps=25,
    eval_every=50,
    batch_size=1,
    seed=2,
)

cd_agent_kwargs = dict(
    hidden_size=16,
    alpha=0.2,
    lr=1e-3,
    use_a_tm1=True,
    gamma=1.0,
    value_loss_wt=0.05,
    vdo=True,
    asymmetry=True,
    default_critic=False,
    rt_budget=1,
    rt_thresholds=0.5,
    default_start=0.0,
    beta_max=1.0,
    vdo_start=0.0,
    greedy=False,
    use_default=False,
)


envs = {
    'FourRooms': (FourRoomsTask, fourrooms_env_kwargs, fourrooms_exp_kwargs, fourrooms_agent_kwargs),
    'Stroop': (StroopTask, stroop_env_kwargs, stroop_exp_kwargs, stroop_agent_kwargs),
    'TwoStep': (TwoStepTask, twostep_env_kwargs, twostep_exp_kwargs, twostep_agent_kwargs),
    'DriftingBandit': (DriftingBanditTask, driftingbandit_env_kwargs, driftingbandit_exp_kwargs, driftingbandit_agent_kwargs),
    'ContingencyDegradation': (OmissionDevaluationTask, cd_env_kwargs, cd_exp_kwargs, cd_agent_kwargs)
}



