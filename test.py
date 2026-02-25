"""
Test Script - Verify Implementation
Tests all components to ensure they work correctly
"""

import numpy as np
import torch


def test_imports():
    """Test all imports work"""
    print("Testing imports...")
    try:
        from config import get_config
        from environment import UAVAssistedVECEnvNOMA   # ← CHANGED
        from networks import Actor, Critic, ReplayBuffer
        from td3_agent import TD3Agent
        from baselines import DDPGAgent, LocalExecutionStrategy
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    try:
        from config import get_config, MAP_CONFIGS, SCENARIO_CONFIGS

        config = get_config(scenario='normal', map_name='map1', num_vehicles=10)

        assert 'td3' in config
        assert 'network' in config
        assert 'map' in config
        assert 'noma' in config                                # ← ADDED: verify noma key present
        assert config['num_vehicles'] == 10

        # Verify key NOMA fields are present
        assert 'num_bs_antennas' in config['noma']
        assert 'num_uav_antennas' in config['noma']
        assert 'noise_variance' in config['noma']
        assert 'wavelength' in config['noma']

        print(f"  TD3 config keys: {list(config['td3'].keys())}")
        print(f"  Network config keys: {list(config['network'].keys())[:5]}...")
        print(f"  Map: {config['map']['description']}")
        print(f"  NOMA antennas (RSU/UAV): {config['noma']['num_bs_antennas']} / {config['noma']['num_uav_antennas']}")
        print("✓ Configuration test passed")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_environment():
    """Test environment creation and reset"""
    print("\nTesting environment...")
    try:
        from config import get_config
        from environment import UAVAssistedVECEnvNOMA   # ← CHANGED

        config = get_config(scenario='normal', map_name='map1', num_vehicles=5)
        env = UAVAssistedVECEnvNOMA(config)                  # ← CHANGED

        # Test reset
        state = env.reset()
        print(f"  State shape: {state.shape}")
        print(f"  Action space: {env.action_space.shape}")
        print(f"  Observation space: {env.observation_space.shape}")

        assert state.shape == env.observation_space.shape
        assert len(state) > 0

        # Test step
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)

        print(f"  Reward: {reward:.4f}")
        print(f"  Cost: {info['total_cost']:.4f}")
        print(f"  Penalty: {info['penalty']:.4f}")
        # NOMA-specific info fields
        print(f"  SINR RSU (mean): {info['sinr_rsu_mean']:.4f}")   # ← ADDED
        print(f"  SINR UAV (mean): {info['sinr_uav_mean']:.4f}")   # ← ADDED
        print(f"  Buffer (mean):   {info['buffer_mean']:.0f} bits") # ← ADDED

        print("✓ Environment test passed")
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_networks():
    """Test neural networks"""
    print("\nTesting neural networks...")
    try:
        from networks import Actor, Critic, ReplayBuffer

        state_dim = 50
        action_dim = 7
        max_action = np.array([50.0, 2*np.pi] + [1.0]*5)

        # Test Actor
        actor = Actor(state_dim, action_dim, max_action)
        state = torch.randn(1, state_dim)
        action = actor(state)
        print(f"  Actor output shape: {action.shape}")
        assert action.shape == (1, action_dim)

        # Test Critic
        critic = Critic(state_dim, action_dim)
        q1, q2 = critic(state, action)
        print(f"  Critic Q1 shape: {q1.shape}")
        print(f"  Critic Q2 shape: {q2.shape}")
        assert q1.shape == (1, 1)
        assert q2.shape == (1, 1)

        # Test ReplayBuffer
        buffer = ReplayBuffer(state_dim, action_dim, max_size=1000)
        for i in range(10):
            s = np.random.randn(state_dim)
            a = np.random.randn(action_dim)
            ns = np.random.randn(state_dim)
            r = np.random.randn()
            d = 0
            buffer.add(s, a, ns, r, d)

        print(f"  Buffer size: {buffer.size}")
        assert buffer.size == 10

        print("✓ Networks test passed")
        return True
    except Exception as e:
        print(f"✗ Networks test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent():
    """Test TD3 agent"""
    print("\nTesting TD3 agent...")
    try:
        from config import get_config
        from td3_agent import TD3Agent

        config = get_config(scenario='normal', map_name='map1', num_vehicles=5)
        state_dim = 50
        action_dim = 7
        max_action = np.array([50.0, 2*np.pi] + [1.0]*5)

        agent = TD3Agent(state_dim, action_dim, max_action, config['td3'])

        # Test action selection
        state = np.random.randn(state_dim)
        action = agent.select_action(state, add_noise=True)
        print(f"  Action shape: {action.shape}")
        assert action.shape == (action_dim,)

        # Test training (without enough data)
        metrics = agent.train()
        print(f"  Training metrics: {metrics}")

        print("✓ Agent test passed")
        return True
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_baselines():
    """Test baseline strategies"""
    print("\nTesting baseline strategies...")
    try:
        from baselines import LocalExecutionStrategy, FullOffloadingStrategy, RandomOffloadingStrategy

        action_dim = 7
        max_action = np.array([50.0, 2*np.pi] + [1.0]*5)
        state = np.random.randn(50)

        # Test LE
        le = LocalExecutionStrategy(action_dim)
        action = le.select_action(state)
        print(f"  LE action shape: {action.shape}")
        assert action.shape == (action_dim,)
        assert action[0] == 0.0  # UAV velocity = 0

        # Test FO
        fo = FullOffloadingStrategy(action_dim)
        action = fo.select_action(state)
        print(f"  FO action shape: {action.shape}")
        assert action.shape == (action_dim,)

        # Test RO
        ro = RandomOffloadingStrategy(action_dim, max_action)
        action = ro.select_action(state)
        print(f"  RO action shape: {action.shape}")
        assert action.shape == (action_dim,)

        print("✓ Baselines test passed")
        return True
    except Exception as e:
        print(f"✗ Baselines test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test full integration - agent interacting with environment"""
    print("\nTesting full integration...")
    try:
        from config import get_config
        from environment import UAVAssistedVECEnvNOMA   # ← CHANGED
        from td3_agent import TD3Agent

        config = get_config(scenario='normal', map_name='map1', num_vehicles=5)
        env = UAVAssistedVECEnvNOMA(config)                  # ← CHANGED

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high

        agent = TD3Agent(state_dim, action_dim, max_action, config['td3'])

        # Run a few steps
        state = env.reset()
        total_reward = 0

        for step in range(5):
            action = agent.select_action(state, add_noise=True)
            next_state, reward, done, info = env.step(action)

            agent.replay_buffer.add(state, action, next_state, reward, done)

            total_reward += reward
            state = next_state

            if done:
                break

        print(f"  Completed {step+1} steps")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Buffer size: {agent.replay_buffer.size}")
        print(f"  State dim seen by agent: {state_dim}")

        print("✓ Integration test passed")
        return True
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*80)
    print("DRL-TCOA Implementation Test Suite (MIMO-NOMA Channel)")
    print("="*80)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Environment", test_environment),
        ("Networks", test_networks),
        ("Agent", test_agent),
        ("Baselines", test_baselines),
        ("Integration", test_integration),
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name:20s}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Implementation is ready to use.")
        print("\nNext steps:")
        print("  1. Run 'python train.py' to start training")
        print("  2. Run 'python plotting.py' to view results")
        print("  3. Run 'python evaluate.py' for full experiments")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")

    print("="*80)


if __name__ == "__main__":
    main()
