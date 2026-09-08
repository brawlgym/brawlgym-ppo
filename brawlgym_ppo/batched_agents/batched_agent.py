def batched_agent_process(proc_id, endpoint, shm_buffer, shm_offset, shm_size, seed, port):
    """
    Function to interact with an environment and communicate with the learner through a pipe.

    :param proc_id: Process id
    :param endpoint: Parent endpoint for communication
    :param shm_buffer: Shared memory buffer
    :param shm_offset: Shared memory offset
    :param shm_size: Shared memory size
    :param seed: Seed for the environment.
    :param port: Bridge port of the game instance this process drives.
    :return: None
    """

    import pickle
    import socket

    import numpy as np

    from brawlgym_ppo.batched_agents import comm_consts
    from brawlgym_ppo.util import BrawlgymWrapper

    def _append_array(array, offset, data):
        size = data.size if isinstance(data, np.ndarray) else len(data)
        end = offset + size
        array[offset:end] = data[:]
        return end

    env = None
    metrics_encoding_function = None
    shm_view = None
    shm_shapes = None

    POLICY_ACTIONS_HEADER = comm_consts.POLICY_ACTIONS_HEADER
    ENV_SHAPES_HEADER = comm_consts.ENV_SHAPES_HEADER
    STOP_MESSAGE_HEADER = comm_consts.STOP_MESSAGE_HEADER

    PACKED_ENV_STEP_DATA_HEADER = comm_consts.pack_message(
        comm_consts.ENV_STEP_DATA_HEADER
    )
    header_len = comm_consts.HEADER_LEN

    # Create a socket and send dummy data to tell parent our endpoint
    pipe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pipe.bind(("127.0.0.1", 0))
    pipe.sendto(b"0", endpoint)

    # Wait for initialization data from the learner.
    while env is None:
        data = pickle.loads(pipe.recv(4096))
        if data[0] == "initialization_data":
            build_env_fn = data[1]
            metrics_encoding_function = data[2]

            env = build_env_fn(port)
            if not isinstance(env, BrawlgymWrapper):
                env = BrawlgymWrapper(env)

    np.random.seed(seed)
    reset_state = env.reset()

    if type(reset_state) != np.ndarray:
        reset_state = np.asarray(reset_state, dtype=np.float32)
    elif reset_state.dtype != np.float32:
        reset_state = reset_state.astype(np.float32)

    state_shape = [float(arg) for arg in np.shape(reset_state)]
    n_elements_in_state_shape = float(len(state_shape))
    n_agents = state_shape[0] if n_elements_in_state_shape > 1 else 1
    obs_buffer = reset_state.tobytes()

    message_floats = (
        comm_consts.ENV_RESET_STATE_HEADER + [n_elements_in_state_shape] + state_shape
    )
    packed_message_floats = comm_consts.pack_message(message_floats) + obs_buffer
    pipe.sendto(packed_message_floats, endpoint)

    action_buffer = None
    action_slice_size = 0

    frombuffer = np.frombuffer

    prev_n_agents = n_agents

    # Primary interaction loop.
    try:
        while True:
            message_bytes = pipe.recv(4096)
            message = frombuffer(message_bytes, dtype=np.float32)
            header = message[:header_len]

            if header[0] == POLICY_ACTIONS_HEADER[0]:
                prev_n_agents = n_agents
                data = message[header_len:]

                if action_buffer is None:
                    action_buffer = np.reshape(data, (int(n_agents), -1)).copy()
                    action_slice_size = action_buffer.shape[1]
                else:
                    for i in range(int(n_agents)):
                        action_buffer[i] = data[
                            i * action_slice_size : (i + 1) * action_slice_size
                        ]

                obs, rew, done, truncated, info = env.step(action_buffer)

                if n_agents == 1 and type(rew) is not list:
                    rew = [float(rew)]

                if done or truncated:
                    obs = np.asarray(env.reset(), dtype=np.float32)
                    n_agents = float(obs.shape[0]) if len(obs.shape) > 1 else 1

                    state_shape = [float(arg) for arg in obs.shape]
                    n_elements_in_state_shape = len(state_shape)

                    action_buffer = np.zeros((int(n_agents), action_buffer.shape[-1]))

                if type(obs) != np.ndarray:
                    obs = np.asarray(obs, dtype=np.float32)
                elif obs.dtype != np.float32:
                    obs = obs.astype(np.float32)

                done = 1.0 if done else 0.0
                truncated = 1.0 if truncated else 0.0

                # per-component rewards and action frequencies lead the metrics payload; user metrics follow
                metrics = np.concatenate([np.asarray(info["reward_components"], dtype=np.float32).reshape(-1),
                                          np.asarray(info["action_frequencies"], dtype=np.float32).reshape(-1)])
                if metrics_encoding_function is not None:
                    metrics = np.concatenate([metrics, metrics_encoding_function(info["state"]).reshape(-1)])
                metrics_shape = [float(arg) for arg in metrics.shape]

                if shm_view is None or shm_shapes != (prev_n_agents, n_agents):
                    shm_shapes = (prev_n_agents, n_agents)
                    count = 5 + len(metrics_shape) + len(state_shape) + len(rew) + metrics.size + obs.size
                    assert(count <= shm_size), "ATTEMPTED TO CREATE AGENT MESSAGE BUFFER LARGER THAN MAXIMUM ALLOWED SIZE"
                    shm_view = np.frombuffer(buffer=shm_buffer, dtype=np.float32, offset=shm_offset, count=count)

                offset = _append_array(shm_view, 0, [prev_n_agents, done, truncated, n_elements_in_state_shape, len(metrics_shape)])
                offset = _append_array(shm_view, offset, metrics_shape)
                offset = _append_array(shm_view, offset, state_shape)
                offset = _append_array(shm_view, offset, rew)
                offset = _append_array(shm_view, offset, metrics)
                offset = _append_array(shm_view, offset, obs.flatten())

                pipe.sendto(PACKED_ENV_STEP_DATA_HEADER, endpoint)

            elif header[0] == ENV_SHAPES_HEADER[0]:
                action_type = ("Discrete", "Multi-discrete", "Continuous")[env.action_space_type]

                print("Received request for env shapes, returning:")
                print(F"- Observations shape: {env.obs_shape}")
                print(F"- Number of actions: {env.n_actions}")
                print(F"- Action space type: {env.action_space_type} ({action_type})")
                if env.bins:
                    print(F"- Bins per action: {env.bins}")
                print("--------------------")

                obs_size = float(np.prod(env.obs_shape[1:]))
                message_floats = ENV_SHAPES_HEADER + [
                    obs_size,
                    float(env.n_actions),
                    float(env.action_space_type),
                ] + [float(b) for b in env.bins]
                pipe.sendto(comm_consts.pack_message(message_floats), endpoint)
                pipe.sendto(pickle.dumps(("metric_names", list(env.reward_names), list(env.action_names))), endpoint)

            elif header[0] == STOP_MESSAGE_HEADER[0]:
                break

    except Exception:
        import traceback

        print("ERROR IN BATCHED AGENT LOOP")
        traceback.print_exc()

    finally:
        pipe.close()
        env.close()
