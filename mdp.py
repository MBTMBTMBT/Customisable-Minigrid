"""
MDP Transition Table Sampling Utilities

This module provides functions for sampling complete MDP transition tables
from environments that support state encoding/decoding.
"""

from typing import Dict, Any, Optional, List, Tuple
import json


def sample_mdp_transitions(env: Any, start_states: Optional[List[str]] = None,
                                   max_states: int = float('inf')) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Unified MDP transition sampling function that supports both single and multiple start states.

    Args:
        env: Environment instance that supports encode_state(), decode_state(), and step() methods
        start_states: Optional list of encoded state strings to use as starting points.
                     If None, uses current environment state as single starting point.
        max_states: Maximum number of states to explore. Default is infinite.

    Returns:
        Dict[str, Dict[int, Dict[str, float]]]: Transition table in format:
            {state_a: {action: {state_b: reward, ...}, ...}, ...}

    Raises:
        AttributeError: If environment doesn't support required methods
        ValueError: If environment is in invalid state
    """
    # Verify environment has required methods
    required_methods = ['encode_state', 'decode_state', 'step']
    for method in required_methods:
        if not hasattr(env, method):
            raise AttributeError(f"Environment must have {method} method")

    # Store original sparse reward setting
    original_sparse = env.reward_config["sparse"]

    # Temporarily disable sparse rewards to get immediate rewards
    env.reward_config["sparse"] = False

    # Initialize combined data structures
    combined_table = {}  # Final combined transition table
    global_visited_states = set()  # Global set of all visited states
    total_states_explored = 0

    try:
        # Determine starting points
        if start_states is None:
            # Single start state mode - use current environment state
            initial_encoded = env.encode_state()
            start_points = [initial_encoded]
            print(f"Starting single-point MDP sampling from: {initial_encoded[:50]}...")
        else:
            # Multiple start states mode
            start_points = start_states
            print(f"Starting multi-point MDP sampling from {len(start_points)} start states...")

        # Process each starting point
        for start_idx, start_state in enumerate(start_points):
            if total_states_explored >= max_states:
                print(f"Reached maximum state limit ({max_states}), stopping exploration")
                break

            if start_states is not None:
                print(f"\n--- Sampling from start state {start_idx + 1}/{len(start_points)} ---")

                # Set environment to start state
                try:
                    env.decode_state(start_state)
                except Exception as e:
                    print(f"Failed to decode start state {start_idx + 1}: {e}")
                    continue

            # Initialize local exploration structures
            local_transition_table = {}  # Local transitions for this start point
            local_visited_states = set()  # Local visited states for this start point
            exploration_queue = []  # Queue of encoded states to explore
            local_states_explored = 0

            # Add starting state to exploration
            current_start = env.encode_state()  # Get actual current state after decode
            exploration_queue.append(current_start)
            local_visited_states.add(current_start)

            # Calculate remaining budget for this start point
            remaining_budget = max_states - total_states_explored

            # Main BFS loop for current start point
            while exploration_queue and local_states_explored < remaining_budget:
                # Get current state from queue
                current_encoded = exploration_queue.pop(0)
                local_states_explored += 1

                if local_states_explored % 100 == 0:
                    print(f"Explored {local_states_explored} states locally, queue size: {len(exploration_queue)}")

                # Restore environment to current state
                env.decode_state(current_encoded)

                # Initialize transition entry for current state if not exists
                if current_encoded not in local_transition_table:
                    local_transition_table[current_encoded] = {}

                # Try each possible action from current state
                valid_actions = []
                for action in range(env.action_space.n):
                    # Save current state before trying action
                    state_before_action = env.encode_state()

                    try:
                        # Execute action and get immediate reward
                        obs, reward, terminated, truncated, info = env.step(action)

                        # Get resulting state
                        next_encoded = env.encode_state()

                        # Record this transition
                        if action not in local_transition_table[current_encoded]:
                            local_transition_table[current_encoded][action] = {}

                        local_transition_table[current_encoded][action][next_encoded] = float(reward)

                        # Add next state to exploration queue if not visited locally and not terminal
                        if (next_encoded not in local_visited_states and
                                not terminated and
                                not truncated and
                                len(local_visited_states) < remaining_budget):
                            exploration_queue.append(next_encoded)
                            local_visited_states.add(next_encoded)

                        # Mark this action as valid
                        valid_actions.append(action)

                    except Exception as e:
                        # If action failed, skip it and continue
                        print(f"Action {action} failed from state {current_encoded[:30]}...: {e}")
                        continue

                    finally:
                        # Always restore state before trying next action
                        try:
                            env.decode_state(state_before_action)
                        except Exception as e:
                            print(f"Failed to restore state: {e}")
                            # If we can't restore, break out of action loop
                            break

                # Verify we have valid transitions for this state
                if not valid_actions:
                    print(f"Warning: No valid actions found for state {current_encoded[:30]}...")

                # Progress update for large state spaces
                if local_states_explored % 1000 == 0:
                    print(f"Local progress: {local_states_explored} states explored, "
                          f"{len(local_transition_table)} states in local table, "
                          f"{len(exploration_queue)} states queued")

            # Merge local results into combined table
            for state, transitions in local_transition_table.items():
                if state not in combined_table:
                    combined_table[state] = {}
                for action, next_states in transitions.items():
                    if action not in combined_table[state]:
                        combined_table[state][action] = {}
                    combined_table[state][action].update(next_states)

            # Update global tracking
            global_visited_states.update(local_visited_states)
            total_states_explored = len(combined_table)

            print(f"Completed start point {start_idx + 1}: {local_states_explored} local states explored")
            print(f"Total unique states discovered so far: {total_states_explored}")

        # Final statistics
        total_transitions = sum(
            sum(len(action_dict) for action_dict in state_dict.values())
            for state_dict in combined_table.values()
        )

        print(f"\nMDP sampling completed!")
        print(f"Total start points processed: {len(start_points)}")
        print(f"Total unique states discovered: {len(combined_table)}")
        print(f"Total transitions recorded: {total_transitions}")
        print(f"Average transitions per state: {total_transitions / len(combined_table) if combined_table else 0:.2f}")

        return combined_table

    except KeyboardInterrupt:
        print(f"\nMDP sampling interrupted by user.")
        print(f"Partial results: {len(combined_table)} states sampled")
        return combined_table

    except Exception as e:
        print(f"Error during MDP sampling: {e}")
        return combined_table

    finally:
        # Always restore original sparse reward setting
        env.reward_config["sparse"] = original_sparse
        print(f"Restored original sparse reward setting: {original_sparse}")


def analyze_transition_table(transition_table: Dict[str, Dict[int, Dict[str, float]]]) -> None:
    """
    Analyze and print statistics about the sampled MDP transition table.

    Args:
        transition_table: The transition table returned by sample_mdp_transitions()
    """
    if not transition_table:
        print("Transition table is empty!")
        return

    # Basic statistics
    num_states = len(transition_table)
    total_transitions = sum(
        sum(len(action_dict) for action_dict in state_dict.values())
        for state_dict in transition_table.values()
    )

    print(f"\n=== MDP TRANSITION TABLE ANALYSIS ===")
    print(f"Total states: {num_states}")
    print(f"Total transitions: {total_transitions}")
    print(f"Average transitions per state: {total_transitions / num_states:.2f}")

    # Action usage statistics
    action_counts = {}
    for state_dict in transition_table.values():
        for action in state_dict.keys():
            action_counts[action] = action_counts.get(action, 0) + len(state_dict[action])

    print(f"\nAction usage:")
    for action in sorted(action_counts.keys()):
        print(f"  Action {action}: {action_counts[action]} transitions")

    # Reward statistics
    all_rewards = []
    reward_distribution = {}

    for state_dict in transition_table.values():
        for action_dict in state_dict.values():
            for reward in action_dict.values():
                all_rewards.append(reward)
                reward_distribution[reward] = reward_distribution.get(reward, 0) + 1

    if all_rewards:
        print(f"\nReward statistics:")
        print(f"  Min reward: {min(all_rewards):.3f}")
        print(f"  Max reward: {max(all_rewards):.3f}")
        print(f"  Average reward: {sum(all_rewards) / len(all_rewards):.3f}")

        print(f"\nReward distribution (top 10):")
        sorted_rewards = sorted(reward_distribution.items(), key=lambda x: x[1], reverse=True)
        for reward, count in sorted_rewards[:10]:
            print(f"  Reward {reward:.3f}: {count} transitions ({count / len(all_rewards) * 100:.1f}%)")

    # Terminal states (states with no outgoing transitions)
    terminal_states = [state for state, transitions in transition_table.items()
                       if not transitions or all(not action_dict for action_dict in transitions.values())]

    print(f"\nTerminal states found: {len(terminal_states)}")
    if terminal_states and len(terminal_states) <= 5:
        print("Terminal states:")
        for state in terminal_states:
            print(f"  {state[:50]}...")

    # States with self-loops
    self_loop_states = []
    for state, state_dict in transition_table.items():
        for action_dict in state_dict.values():
            if state in action_dict:
                self_loop_states.append(state)
                break

    print(f"States with self-loops: {len(self_loop_states)}")

    # Connectivity analysis
    reachable_from_initial = set()
    if transition_table:
        # Find states reachable from any initial state
        # (since we don't know which was initial, we'll just report total)
        print(f"Total reachable states: {num_states}")


def save_transition_table(transition_table: Dict[str, Dict[int, Dict[str, float]]],
                          filename: str) -> None:
    """
    Save the transition table to a JSON file.

    Args:
        transition_table: The transition table to save
        filename: Path to save the file
    """
    # Convert to JSON-serializable format
    json_table = {}
    for state, state_dict in transition_table.items():
        json_table[state] = {}
        for action, action_dict in state_dict.items():
            json_table[state][str(action)] = action_dict

    try:
        with open(filename, 'w') as f:
            json.dump(json_table, f, indent=2)
        print(f"Transition table saved to {filename}")
        print(f"File contains {len(json_table)} states and can be loaded later for analysis.")
    except Exception as e:
        print(f"Failed to save transition table: {e}")


def load_transition_table(filename: str) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Load a transition table from a JSON file.

    Args:
        filename: Path to the JSON file

    Returns:
        The loaded transition table
    """
    try:
        with open(filename, 'r') as f:
            json_table = json.load(f)

        # Convert back to proper format
        transition_table = {}
        for state, state_dict in json_table.items():
            transition_table[state] = {}
            for action_str, action_dict in state_dict.items():
                action = int(action_str)
                transition_table[state][action] = action_dict

        print(f"Transition table loaded from {filename}")
        print(f"Loaded {len(transition_table)} states")
        return transition_table

    except Exception as e:
        print(f"Failed to load transition table: {e}")
        return {}


# Example usage functions
def quick_analysis(filename: str) -> None:
    """
    Quick analysis of a saved transition table file.

    Args:
        filename: Path to the JSON transition table file
    """
    transition_table = load_transition_table(filename)
    if transition_table:
        analyze_transition_table(transition_table)


def compare_transition_tables(filename1: str, filename2: str) -> None:
    """
    Compare two transition tables and report differences.

    Args:
        filename1: Path to first transition table file
        filename2: Path to second transition table file
    """
    table1 = load_transition_table(filename1)
    table2 = load_transition_table(filename2)

    if not table1 or not table2:
        print("Failed to load one or both transition tables")
        return

    print(f"\n=== TRANSITION TABLE COMPARISON ===")
    print(f"Table 1 ({filename1}): {len(table1)} states")
    print(f"Table 2 ({filename2}): {len(table2)} states")

    # Find common and unique states
    states1 = set(table1.keys())
    states2 = set(table2.keys())

    common_states = states1.intersection(states2)
    unique_to_1 = states1 - states2
    unique_to_2 = states2 - states1

    print(f"\nCommon states: {len(common_states)}")
    print(f"States unique to table 1: {len(unique_to_1)}")
    print(f"States unique to table 2: {len(unique_to_2)}")

    # Compare transitions for common states
    different_transitions = 0
    for state in common_states:
        if table1[state] != table2[state]:
            different_transitions += 1

    print(f"Common states with different transitions: {different_transitions}")

    if different_transitions > 0 and different_transitions <= 5:
        print("\nExample differences:")
        count = 0
        for state in common_states:
            if table1[state] != table2[state] and count < 3:
                print(f"  State {state[:30]}... has different transitions")
                count += 1
