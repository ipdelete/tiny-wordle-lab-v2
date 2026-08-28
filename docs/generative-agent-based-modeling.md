# Generative agent-based modeling

## Summary

Agent-based modeling, or ABM, is best treated as a simulation paradigm distinct
from system dynamics. System dynamics models aggregate stocks, flows, and
feedback loops. ABM models heterogeneous entities whose local decisions and
interactions produce system-level behavior. The two can be combined, but ABM
is not a derivative of system dynamics.

Since 2023, researchers have put large language models inside simulated agents.
Several names are in use:

- generative agents;
- generative agent-based modeling, or GABM;
- LLM-based agent-based modeling;
- LLM-based social simulation;
- generative social simulation.

The vocabulary has not settled. "Generative agents" most often refers to the
memory, reflection, and planning architecture introduced by Park et al.
"Generative agent-based modeling" is appropriate when LLM-driven decision
rules sit inside a recognizable ABM with an environment, interacting agents,
time progression, and aggregate outcomes.

The main scientific problem is validation. Fluent dialogue and convincing
stories do not establish that simulated agents reproduce human behavior,
population distributions, or causal mechanisms.

## How the simulation traditions relate

### System dynamics

The [System Dynamics Society](https://systemdynamics.org/what-is-system-dynamics/)
defines system dynamics through feedback-systems theory. Models usually
represent aggregate stocks, flows, auxiliaries, and feedback loops using
differential or difference equations.

The model is normally top-down. Population categories or accumulated
quantities are state variables. Individual people or firms may not appear at
all.

### Agent-based modeling

ABM is normally bottom-up. A classical model specifies:

1. agent types and heterogeneous attributes;
2. private and public state;
3. behavioral rules;
4. an environment;
5. an interaction network;
6. a scheduler or event loop;
7. institutional constraints;
8. aggregate measurements.

Macal and North describe ABM as a distinct modeling approach, not an offshoot
of system dynamics. Bonabeau likewise defines it through individual agents,
interactions, and emergent system behavior.

Sources:

- [Macal and North, "Tutorial on Agent-Based Modelling and Simulation," 2010](https://doi.org/10.1057/jos.2010.3)
- [Bonabeau, "Agent-Based Modeling," 2002](https://doi.org/10.1073/pnas.082080899)

System dynamics and ABM answer overlapping questions at different levels.
Hybrid models may connect aggregate feedback equations to explicit agents.
That does not make either method a subtype of the other.

### Microsimulation

Microsimulation advances individual records through transition rules and
reports distributions or policy effects. It often models people, households,
or firms without giving them autonomous policies or interaction networks.

An LLM survey panel whose members answer independently is closer to
microsimulation or synthetic-participant research than ABM. It becomes more
ABM-like when participants interact, alter a shared environment, and affect
one another's later choices.

The foundational source is [Orcutt, "A New Type of Socio-Economic System,"
1957](https://econpapers.repec.org/RePEc:ijm:journl:v:1:y:2007:i:1:p:3-9).

### Discrete-event simulation

Discrete-event simulation describes how state changes are scheduled in time.
An ABM can use synchronous ticks, asynchronous activation, continuous time, or
a discrete-event queue. DES and ABM therefore describe different parts of a
model.

### Multi-agent systems

Multi-agent systems research comes from distributed artificial intelligence.
It studies autonomous software agents that coordinate, negotiate, compete, or
solve distributed tasks. Those agents need not represent a real population or
support a scientific simulation.

Wooldridge and Jennings describe intelligent agents through autonomy, social
ability, reactivity, and proactiveness:

- [Wooldridge and Jennings, "Intelligent Agents: Theory and Practice," 1995](https://doi.org/10.1017/S0269888900008122)

An ABM agent can be a few deterministic rules. A multi-agent software system
can contain powerful agents without being an ABM.

## What "generative" changes

In a conventional ABM, a programmer writes the decision rule. It may be a
threshold, utility function, finite-state machine, probabilistic transition,
or learned policy.

In a GABM, an LLM replaces or augments that decision rule. Most of the
simulation remains ordinary code.

| Layer | Classical ABM | Typical GABM |
| --- | --- | --- |
| Environment | Grid, market, institution, network | Still conventional code |
| Agent attributes | Parameters and agent classes | Profiles, personas, goals, memories |
| State | Typed variables | Typed variables plus text or vector memories |
| Observation | Selected local state | Rendered text, retrieved memories, tool outputs |
| Decision policy | Rules or utility functions | LLM-generated action or plan |
| Interaction topology | Explicit graph or matching rule | Still explicit |
| Time | Tick loop or event queue | Still explicit |
| Institutions | Laws, budgets, permissions, prices | Usually conventional constraints |
| Action resolution | Transition function | Parser, validator, game master, or API call |
| Measurement | Counts and distributions | Still conventional aggregation |

The LLM does not become the simulator. It proposes decisions inside a simulator
that still owns state, causality, time, constraints, and measurement.

## Important projects and papers

### Generative Agents

Park et al. introduced 25 LLM-driven characters living in a Smallville game
environment. Each agent has a memory stream, retrieval based on relevance,
recency, and importance, reflection, and hierarchical planning.

The system produced information diffusion, relationship formation, and
coordination around a Valentine's Day party. The evaluation focused on
believability and architectural ablations. The authors also reported memory
retrieval failures, fabricated details, and overly formal behavior.

Status: peer-reviewed at ACM UIST 2023.

- [Paper](https://doi.org/10.1145/3586183.3606763)
- [arXiv](https://arxiv.org/abs/2304.03442)
- [Source repository](https://github.com/joonspk-research/generative_agents)

The party is a useful system demonstration. It is not evidence that real
communities organize at the same rate or through the same mechanisms.

### Concordia

Google DeepMind's Concordia paper explicitly uses the term generative
agent-based modeling. Agents combine associative memory with components for
identity, goals, possessions, social context, plans, or physiological state.
Components may use either LLM calls or programmed logic.

A Game Master owns the environment. It requests actions, resolves physical and
institutional constraints, updates state, and produces observations. This
draws a clean boundary between generative decision-making and conventional
simulation.

Status: technical-report preprint, December 2023, with an active source
repository.

- [Paper](https://arxiv.org/abs/2312.03664)
- [DeepMind publication page](https://deepmind.google/research/publications/64717/)
- [Source repository](https://github.com/google-deepmind/concordia)

### GABM tutorial

Ghaffarzadegan et al. published an introduction and tutorial in *System
Dynamics Review*. It combines LLM decision models with mechanistic interaction
models and illustrates the method through norm diffusion.

Status: peer-reviewed, January 2024.

- [Paper](https://doi.org/10.1002/sdr.1761)

Its publication venue does not show that ABM derives from system dynamics. The
paper presents a hybrid modeling method.

### SOTOPIA

SOTOPIA is an environment for evaluating social intelligence. It constructs
role-play episodes from scenarios, characters, relationships, private goals,
verbal actions, and nonverbal actions. Its evaluation considers goal
completion, finances, relationships, secret-keeping, and social-rule
compliance.

Human role-players provide a comparison for LLM agents. The work reports that
agent behavior depends strongly on the interaction partner and that models
often reveal secrets or violate social rules. SOTOPIA-pi later uses interactive
data to train a target language agent.

Status: SOTOPIA was an ICLR 2024 spotlight; SOTOPIA-pi appeared at ACL 2024.

- [SOTOPIA paper](https://arxiv.org/abs/2310.11667)
- [SOTOPIA-pi paper](https://arxiv.org/abs/2403.08715)
- [Source repository](https://github.com/sotopia-lab/sotopia)

This is a useful boundary case. The same environment can evaluate fixed actors
or train a policy. Those are different experiments.

### AgentSociety

AgentSociety separates LLM-driven agents from urban, social, and economic
environments and a distributed simulation engine. Agent state includes
profiles, memory, needs, cognition, and social relationships. Conventional
code handles mobility, economic transactions, event progression, and
execution.

The first paper reports simulations above 10,000 agents. AgentSociety 2,
released in 2026, adds AI social scientists that orchestrate experiments while
other agents act as synthetic participants. Its authors state that researchers
must still define constructs, assess validity, specify interventions, and
decide which claims the evidence supports.

Status: first-party preprints and an active source repository.

- [AgentSociety, 2025](https://arxiv.org/abs/2502.08691)
- [AgentSociety 2, 2026](https://arxiv.org/abs/2607.11895)
- [Source repository](https://github.com/tsinghua-fib-lab/AgentSociety)

### Project Sid

Project Sid reports Minecraft simulations with 10 to more than 1,000 agents.
The paper describes role specialization, collective rules, voting, and
cultural transmission.

Status: technical-report preprint, October 2024.

- [Paper](https://arxiv.org/abs/2411.00114)
- [First-party repository](https://github.com/altera-al/project-sid)

The repository does not provide the implementation. That limits
reproducibility, so its social claims should be treated as preliminary.

### Synthetic participants and economic agents

Several related projects simulate individuals without always building an ABM:

- [Aher, Arriaga, and Kalai, ICML 2023](https://proceedings.mlr.press/v202/aher23a.html)
  used LLMs to reproduce human-subject studies. Their "hyper-accuracy"
  distortion shows that plausible responses can have nonhuman distributions.
- [Argyle et al., *Political Analysis* 2023](https://doi.org/10.1017/pan.2023.2)
  tested "algorithmic fidelity" by conditioning GPT-3 on survey-derived
  backstories.
- [Horton, Filippas, and Manning, NBER 2023, revised through 2026](https://www.nber.org/papers/w31122)
  studied LLMs as simulated economic agents while stressing that human studies
  must confirm the findings.
- [GenSim, 2024](https://arxiv.org/abs/2410.04360) reports a distributed
  architecture for simulations with up to 100,000 LLM-based agents.

## Validation is the hard part

### Believability is not behavioral validity

A coherent biography or plausible conversation provides face validity. It
does not show that the model reproduces human choices, response variance,
social mechanisms, or intervention effects.

Validation needs empirical targets. Depending on the research question, these
may include:

- individual choice distributions;
- subgroup differences;
- transition probabilities;
- network statistics;
- aggregate time series;
- directional treatment effects;
- held-out interventions.

Cross, Haber, and Yamins propose selecting the smallest cognitive architecture
that reproduces directional effects from human experiments, then using it to
make predictions for later empirical tests:

- [Validating Generative Agent-Based Models of Social Norm Enforcement, 2025](https://arxiv.org/abs/2507.22049)

### Calibration can hide the wrong mechanism

Several different micro-level rules can reproduce the same aggregate result.
Matching one curve does not identify the mechanism. Researchers should compare
alternative architectures, run component ablations, and validate at more than
one level.

### Stochasticity has several sources

Variation can come from:

- the sampled population;
- initial conditions;
- scheduling order;
- language-model decoding;
- memory retrieval;
- environment transitions;
- model or prompt changes.

Reports should separate these sources and include independent runs. One vivid
trace cannot support an emergent-behavior claim.

### Prompts are part of the model

A prompt may change an agent's effective beliefs, goals, constraints, and
social norms. Results are conditional on the exact model, prompt, retrieval
configuration, and decoding parameters.

### Training-corpus leakage is a threat

An LLM may have seen the experiment, its canonical result, or descriptions of
the institution. Apparent replication may be recall.

Horton discusses this problem directly. Useful controls include novel task
variants, unpublished data, paraphrases, counterfactuals, and models with
better documented training data.

- [Horton et al.](https://arxiv.org/abs/2301.07543)
- [Balloccu et al., EACL 2024](https://aclanthology.org/2024.eacl-long.5/)

### Reproducibility requires more than source code

A reproducible GABM package needs:

- prompts and initial populations;
- model identifiers and API parameters;
- embedding and retrieval configuration;
- seeds and schedules;
- action parsers;
- raw traces;
- post-processing;
- a policy for failed calls.

Model APIs can change after publication. Saved traces and exact configuration
matter even when orchestration code is public.

### "Emergent" needs an operational definition

An unexpected story is not enough. A defensible emergent pattern should be a
measured aggregate outcome not directly hard-coded into individual rules.
Strong claims need repeated runs, null models, simpler-rule baselines,
architectural ablations, sensitivity analysis, and empirical comparison.

## GABM and reinforcement learning use "agent" differently

In reinforcement learning, an agent is a policy optimized to maximize
cumulative environmental reward.

- [OpenAI Spinning Up introduction to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [Gymnasium basic usage](https://gymnasium.farama.org/introduction/basic_usage/)

In GABM, simulated actors may remain fixed for the entire experiment. The goal
is usually to observe population behavior under specified mechanisms, not to
improve each actor.

These cases should not be conflated:

1. Fixed LLM actors in a social simulation are GABM or LLM-based social
   simulation.
2. A learned policy evaluated against simulated users is simulation-based
   evaluation.
3. A policy updated from rewards produced by a simulation is simulation-based
   RL.
4. Several policies that adapt through reward form multi-agent RL.
5. LLM actors that generate trajectories for another model are synthetic-data
   generators, not automatically an ABM.

A simulation can contain generative agents without using RL. RL can use a
simulator without using ABM.

## Relevance to the Wordle training lab

This repository has one policy interacting with a deterministic environment.
The hidden word is environment state. The evaluator applies Wordle rules,
returns observations, and records results. The 2,315 answers form an
evaluation population, not a population of interacting agents.

That makes Wordle a partially observed sequential decision problem, not an
ABM and not a GABM.

ABM methodology still offers useful habits:

- keep policy and environment separate;
- stratify answers by frequency, repeated letters, and difficulty;
- vary one mechanism while keeping the environment fixed;
- run sensitivity experiments across prompts, seeds, and checkpoints;
- report distributions and subgroup failures;
- compare LLM behavior with simple-rule policies;
- keep scoring and legality in authoritative conventional code.

These ideas do not require ABM terminology:

| Activity | More precise term |
| --- | --- |
| Sampling many hidden answers | Evaluation over a test population |
| Sampling several games from one model | Rollout ensemble |
| Training from simulator rewards | Simulation-based RL |
| Progressively harder answers | Curriculum learning |
| Varying answer distributions | Distribution design or domain randomization |
| Comparing prompts and checkpoints | Sensitivity analysis |

Reward design remains an RL question. Multiple LLM personas are unnecessary
unless the experiment concerns interaction among different solvers.

A defensible ABM extension would need interacting solver populations, shared
or withheld clues, strategy diffusion, markets for information, or population
adaptation over time. A two-player adversarial Wordle variant would still be
better described as a game or multi-agent RL environment unless the research
question concerns population-level emergence.

Recommended wording for this project:

> We train and evaluate a language-model policy in a deterministic Wordle
> environment. We borrow simulation methodology from ABM, including explicit
> state, heterogeneous test populations, repeated runs, and aggregate
> analysis, but this is not generative agent-based modeling.

## Reading order

1. [Macal and North, 2010](https://doi.org/10.1057/jos.2010.3) for classical
   ABM.
2. [Park et al., 2023](https://arxiv.org/abs/2304.03442) for the original
   generative-agent architecture.
3. [Concordia, 2023](https://arxiv.org/abs/2312.03664) for an explicit GABM
   architecture.
4. [Ghaffarzadegan et al., 2024](https://doi.org/10.1002/sdr.1761) for a
   compact tutorial.
5. [SOTOPIA, 2024](https://arxiv.org/abs/2310.11667) for social evaluation
   against human behavior.
6. [Aher et al., 2023](https://proceedings.mlr.press/v202/aher23a.html) and
   [Horton et al.](https://arxiv.org/abs/2301.07543) for validation problems.
7. [Cross et al., 2025](https://arxiv.org/abs/2507.22049) for empirical
   validation of GABM components.
8. [AgentSociety](https://arxiv.org/abs/2502.08691) for larger-scale social
   simulation.

## Bottom line

Generative agent-based modeling is a real and increasingly used term, but it
does not describe every system that contains an LLM and a simulator. Its
distinctive use is simulation of interacting populations whose decision rules
include generative models.

For this project, the useful connection is methodological. ABM offers a mature
way to think about heterogeneity, sensitivity, mechanisms, and aggregate
measurement. The actual training problem remains single-agent,
simulation-based reinforcement learning.
