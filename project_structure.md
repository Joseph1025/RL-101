rl100_framework/
├── configs/                        # Hydra configuration files
│   ├── algorithm/
│   │   ├── rl100_offline.yaml      # Params for Algo 1 (M iterations, thresholds)
│   │   └── rl100_online.yaml       # Params for online fine-tuning (optional)
│   ├── model/
│   │   ├── iql_critics.yaml        # Architecture for Q and V networks
│   │   └── transition_model.yaml   # Architecture for OPE dynamics model
│   └── main.yaml                   # Root config
│
├── src/
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── rl100_orchestrator.py   # Implements Algorithm 1 (The main loop)
│   │   ├── offline_agent.py        # Logic for PPO updates & IQL Critic training
│   │   └── ope_gate.py             # AM-Q (Advantage Model-Q) Gating logic
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── critics.py              # Twin Q-Nets and Value Net (IQL architecture)
│   │   ├── transition.py           # Deterministic/Probabilistic Transition Model
│   │   └── policy_wrapper.py       # WRAPPER around d3il.DiffusionPolicy
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── offline_buffer.py       # Load demos + RL rollouts (s, a, r, s')
│   │   └── sampler.py              # Efficient batch sampler for RL updates
│   │
│   ├── utils/
│   │   ├── math_utils.py           # Expectile loss, PPO clipping
│   │   └── logging.py              # WandB / Tensorboard integration
│   │
│   └── d3il_adapter/               # Bridge to your existing library
│       ├── trainer_hook.py         # Subclass d3il's trainer to allow re-training
│       └── frozen_encoder.py       # Utilities to freeze/unfreeze d3il vision nets
│
├── scripts/
│   ├── train_rl100.py              # Main entry point
│   └── evaluate_checkpoint.py      # Standalone eval script
│
└── README.md