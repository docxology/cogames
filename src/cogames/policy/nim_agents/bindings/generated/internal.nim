when not defined(gcArc) and not defined(gcOrc):
  {.error: "Please use --gc:arc or --gc:orc when using Genny.".}

when (NimMajor, NimMinor, NimPatch) == (1, 6, 2):
  {.error: "Nim 1.6.2 not supported with Genny due to FFI issues.".}
proc nim_agents_nim_agents_init_chook*() {.raises: [], cdecl, exportc, dynlib.} =
  nim_agents_init_chook()

proc nim_agents_start_measure*() {.raises: [], cdecl, exportc, dynlib.} =
  startMeasure()

proc nim_agents_end_measure*() {.raises: [], cdecl, exportc, dynlib.} =
  endMeasure()

proc nim_agents_random_policy_unref*(x: RandomPolicy) {.raises: [], cdecl, exportc, dynlib.} =
  GC_unref(x)

proc nim_agents_new_random_policy*(environment_config: cstring): RandomPolicy {.raises: [], cdecl, exportc, dynlib.} =
  newRandomPolicy(environment_config.`$`)

proc nim_agents_random_policy_step_batch*(policy: RandomPolicy, agent_ids: pointer, num_agent_ids: int, num_agents: int, num_tokens: int, size_token: int, raw_observations: pointer, num_actions: int, raw_actions: pointer) {.raises: [], cdecl, exportc, dynlib.} =
  stepBatch(policy, agent_ids, num_agent_ids, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions)

proc nim_agents_thinky_policy_unref*(x: ThinkyPolicy) {.raises: [], cdecl, exportc, dynlib.} =
  GC_unref(x)

proc nim_agents_new_thinky_policy*(environment_config: cstring): ThinkyPolicy {.raises: [], cdecl, exportc, dynlib.} =
  newThinkyPolicy(environment_config.`$`)

proc nim_agents_thinky_policy_step_batch*(policy: ThinkyPolicy, agent_ids: pointer, num_agent_ids: int, num_agents: int, num_tokens: int, size_token: int, raw_observations: pointer, num_actions: int, raw_actions: pointer) {.raises: [], cdecl, exportc, dynlib.} =
  stepBatch(policy, agent_ids, num_agent_ids, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions)

proc nim_agents_race_car_policy_unref*(x: RaceCarPolicy) {.raises: [], cdecl, exportc, dynlib.} =
  GC_unref(x)

proc nim_agents_new_race_car_policy*(environment_config: cstring): RaceCarPolicy {.raises: [], cdecl, exportc, dynlib.} =
  newRaceCarPolicy(environment_config.`$`)

proc nim_agents_race_car_policy_step_batch*(policy: RaceCarPolicy, agent_ids: pointer, num_agent_ids: int, num_agents: int, num_tokens: int, size_token: int, raw_observations: pointer, num_actions: int, raw_actions: pointer) {.raises: [], cdecl, exportc, dynlib.} =
  stepBatch(policy, agent_ids, num_agent_ids, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions)

proc nim_agents_ladybug_policy_unref*(x: LadybugPolicy) {.raises: [], cdecl, exportc, dynlib.} =
  GC_unref(x)

proc nim_agents_new_ladybug_policy*(environment_config: cstring): LadybugPolicy {.raises: [], cdecl, exportc, dynlib.} =
  newLadybugPolicy(environment_config.`$`)

proc nim_agents_ladybug_policy_step_batch*(policy: LadybugPolicy, agent_ids: pointer, num_agent_ids: int, num_agents: int, num_tokens: int, size_token: int, raw_observations: pointer, num_actions: int, raw_actions: pointer) {.raises: [], cdecl, exportc, dynlib.} =
  stepBatch(policy, agent_ids, num_agent_ids, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions)

