import json
import random
import warnings
from enum import IntEnum
from typing import Optional, Tuple, List, Dict, Any, SupportsFloat

from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import *
from minigrid.core.world_object import WorldObj
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.core import ActType, ObsType
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, TILE_PIXELS

from minigrid.simple_actions import SimpleActions
from simple_manual_control import SimpleManualControl

DEFAULT_REWARD_DICT = {
    "sparse": True,
    "step_penalty": -0.1,
    "goal_reward": 0.0,
    "lava_penalty": 0.0,
    "pickup_reward": 0.0,
    "door_open_reward": 0.0,
    "door_close_penalty": 0.0,
    "key_drop_penalty": 0.0,
    "item_drop_penalty": 0.0,
}


class CustomEnv(MiniGridEnv):
    """
    A custom MiniGrid environment that loads its layout from a structured JSON configuration.

    This environment does not support random generation. All objects, colors, positions, and agent settings
    must be defined either in a JSON file or passed as a Python dictionary.
    Later layers overwrite earlier ones. Only 'door' objects are allowed to overwrite others (e.g., walls).

    Attributes:
        json_file_path (Optional[str]): Path to the JSON layout file.
        config (Optional[dict]): Parsed layout configuration dictionary.
        display_size (int): Size of the visualized grid (can be larger than the layout).
        display_mode (str): "middle" or "random" placement of layout in the full grid.
        random_rotate (bool): Whether to randomly rotate the layout (0°, 90°, 180°, 270°).
        random_flip (bool): Whether to randomly flip the layout horizontally.
        mission (str): Text description of the agent's task.
        render_carried_objs (bool): If True, renders the carried object in a separate visual tile.
        any_key_opens_the_door (bool): If True, any key can open any door regardless of color.
        reward_config (dict): Dictionary containing reward settings and values.
    """

    def __init__(
            self,
            json_file_path: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,
            display_size: Optional[int] = None,
            display_mode: Optional[str] = "middle",
            random_rotate: bool = False,
            random_flip: bool = False,
            custom_mission: str = "Explore and interact with objects.",
            max_steps: int = 100000,
            render_carried_objs: bool = True,
            any_key_opens_the_door: bool = False,
            reward_config: Optional[Dict[str, Any]] = None,
            **kwargs,
    ) -> None:
        """
        Initialize the environment from either a file or a config dictionary.

        Args:
            reward_config: Dictionary containing reward configuration. If None, uses default values.
                          Available keys:
                          - sparse (bool): If True, only return total reward at termination
                          - step_penalty (float): Penalty for each step taken
                          - goal_reward (float): Reward for reaching the goal
                          - lava_penalty (float): Penalty for stepping on lava
                          - pickup_reward (float): Reward for picking up an item
                          - door_open_reward (float): Reward for opening a door
                          - door_close_penalty (float): Penalty for closing a door
                          - key_drop_penalty (float): Penalty for dropping a key
                          - item_drop_penalty (float): Penalty for dropping an item
        """
        # Enforce exclusive choice: exactly one of the two must be provided
        assert (json_file_path is not None) ^ (config is not None), \
            "You must provide either 'json_file_path' or 'config', but not both."

        self.txt_file_path = json_file_path
        self.display_mode = display_mode
        self.random_rotate = random_rotate
        self.random_flip = random_flip
        self.render_carried_objs = render_carried_objs
        self.any_key_opens_the_door = any_key_opens_the_door
        self.mission = custom_mission
        self.tile_size = 32
        self.skip_reset = False

        if reward_config is None:
            self.reward_config = DEFAULT_REWARD_DICT
        else:
            # Merge provided config with defaults
            self.reward_config = {**DEFAULT_REWARD_DICT, **reward_config}

        # Initialize cumulative reward for sparse mode
        self.cumulative_reward = 0.0

        # Load config from file if needed
        if json_file_path is not None:
            with open(json_file_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = config

        # Determine layout size
        height, width = self.config["height_width"]
        layout_size = max(height, width)

        if display_size is None:
            self.display_size = layout_size
        else:
            self.display_size = display_size

        assert display_mode in ["middle", "random"], "Invalid display_mode"
        assert self.display_size >= layout_size, "display_size must be >= layout layout_size"

        # Initialize parent class
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: custom_mission),
            grid_size=self.display_size,
            max_steps=max_steps,
            **kwargs,
        )

        self.actions = SimpleActions
        self.action_space = spaces.Discrete(len(self.actions))

        self.step_count = 0

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        frame = super().get_frame(highlight, tile_size, agent_pov)
        if not self.render_carried_objs:
            return frame
        else:
            return self.render_with_carried_objects(frame)

    def render_with_carried_objects(self, full_image):
        """
        Renders the image of the environment with an extra row at the bottom and an extra column on the right,
        displaying the item carried by the agent in the bottom-right corner, if any.
        The agent can carry at most one item.

        :param full_image: The original image rendered by get_full_render.
        :return: Modified image with additional row and column displaying the carried item, if any.
        """
        tile_size = self.tile_size

        # Get original image dimensions
        full_height, full_width, _ = full_image.shape

        # Grid line color in float32 format [0,1] - equivalent to (100, 100, 100) in uint8
        grid_color = (100 / 255, 100 / 255, 100 / 255)

        def create_empty_tile():
            """Helper function to create an empty tile with grid lines"""
            canvas = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
            fill_coords(canvas, point_in_rect(0, 0.031, 0, 1), grid_color)  # Left border
            fill_coords(canvas, point_in_rect(0, 1, 0, 0.031), grid_color)  # Top border
            canvas = np.clip(canvas, 0.0, 1.0)
            return (canvas * 255).astype(np.uint8)

        # Step 1: Add a column on the right
        # Calculate how many tiles we need for the height
        num_height_tiles = full_height // tile_size

        # Create the right column filled with empty tiles
        right_column = np.zeros((full_height, tile_size, 3), dtype=np.uint8)

        for i in range(num_height_tiles):
            empty_tile = create_empty_tile()
            start_y = i * tile_size
            end_y = start_y + tile_size
            right_column[start_y:end_y, :, :] = empty_tile

        # Handle any remaining pixels in height
        remaining_height = full_height % tile_size
        if remaining_height > 0:
            empty_tile = create_empty_tile()
            start_y = num_height_tiles * tile_size
            right_column[start_y:, :, :] = empty_tile[:remaining_height, :, :]

        # Combine original image with right column
        image_with_column = np.hstack([full_image, right_column])

        # Step 2: Add a row at the bottom
        new_width = image_with_column.shape[1]
        num_width_tiles = new_width // tile_size

        # Create the bottom row filled with empty tiles
        bottom_row = np.zeros((tile_size, new_width, 3), dtype=np.uint8)

        for i in range(num_width_tiles):
            empty_tile = create_empty_tile()
            start_x = i * tile_size
            end_x = start_x + tile_size
            bottom_row[:, start_x:end_x, :] = empty_tile

        # Handle any remaining pixels in width
        remaining_width = new_width % tile_size
        if remaining_width > 0:
            empty_tile = create_empty_tile()
            start_x = num_width_tiles * tile_size
            bottom_row[:, start_x:, :] = empty_tile[:, :remaining_width, :]

        # Step 3: If the agent is carrying an object, render it in the bottom-right corner
        if self.carrying is not None:
            # Create a canvas for rendering the carried object
            canvas = np.zeros((tile_size, tile_size, 3), dtype=np.float32)

            # First render the empty cell background
            fill_coords(canvas, point_in_rect(0, 0.031, 0, 1), grid_color)  # Left border
            fill_coords(canvas, point_in_rect(0, 1, 0, 0.031), grid_color)  # Top border

            # Then render the actual object on top
            self.carrying.render(canvas)

            # Convert from float32 [0,1] to uint8 [0,255]
            canvas = np.clip(canvas, 0.0, 1.0)
            item_tile = (canvas * 255).astype(np.uint8)

            # Place the item tile in the bottom-right corner of the bottom row
            bottom_row[:, -tile_size:, :] = item_tile

        # Step 4: Combine everything
        output_image = np.vstack([image_with_column, bottom_row])

        return output_image

    def determine_layout_size(self) -> int:
        if self.txt_file_path:
            with open(self.txt_file_path, 'r') as file:
                sections = file.read().split('\n\n')
                layout_lines = sections[0].strip().split('\n')
                height = len(layout_lines)
                width = max(len(line) for line in layout_lines)
                return max(width, height)
        else:
            return max(self.rand_gen_shape)

    def _read_file(self) -> Dict[str, Any]:
        """
        Load layout configuration from a JSON file.

        Returns:
            Dict containing metadata and layers.
        """
        with open(self.txt_file_path, 'r') as f:
            config = json.load(f)

        assert 'height_width' in config and 'layers' in config, "Invalid layout file structure."

        return config

    def _gen_grid(self, width: int, height: int) -> None:
        """
        Generate the environment grid from a multi-layer JSON layout.
        Layers are applied in order. Only doors are allowed to overwrite existing objects (e.g., walls).
        All other objects skip occupied positions to prevent overlap.
        """
        self.grid = Grid(width, height)

        # Load JSON config
        config = self.config
        H, W = config['height_width']
        layers = config['layers']

        # Compute anchor offsets
        free_width = self.display_size - W
        free_height = self.display_size - H
        if self.display_mode == "middle":
            anchor_x = free_width // 2
            anchor_y = free_height // 2
        elif self.display_mode == "random":
            anchor_x = random.choice(range(max(free_width, 1))) if free_width > 0 else 0
            anchor_y = random.choice(range(max(free_height, 1))) if free_height > 0 else 0
        else:
            raise ValueError("Invalid display mode")

        # Apply random rotation and flip
        image_direction = random.choice([0, 1, 2, 3]) if self.random_rotate else 0
        flip = random.choice([0, 1]) if self.random_flip else 0

        # Track filled positions to prevent unwanted overlap
        filled = set()
        key_positions = []
        goal_positions = []
        agent_positions = []
        orientation = "random"

        for layer in layers:
            obj_type = layer["obj"]
            colour = layer.get("colour", None)
            status = layer.get("status", None)
            orientation = layer.get("orientation", "random") if obj_type == "agent" else orientation
            dist = layer.get("distribution", "all")
            mat = layer["matrix"]

            # Collect all candidate positions from the matrix
            raw_positions = [
                (x, y) for y in range(H) for x in range(W) if mat[y][x] == 1
            ]
            # Transform to actual coordinates with anchor, rotation, and flip
            candidates = []
            for x, y in raw_positions:
                x_shift, y_shift = anchor_x + x, anchor_y + y
                x_rot, y_rot = rotate_coordinate(x_shift, y_shift, image_direction, self.display_size)
                x_final, y_final = flip_coordinate(x_rot, y_rot, flip, self.display_size)
                candidates.append((x_final, y_final))

            # Only doors can overwrite filled positions
            if obj_type == "door":
                available = candidates
            else:
                available = [p for p in candidates if p not in filled]

            # Select final positions based on distribution
            if dist == "all":
                used = available
            elif dist == "one":
                if len(available) < 1:
                    raise ValueError(f"No available positions for object '{layer['name']}'")
                used = [random.choice(available)]
            elif isinstance(dist, float):
                count = max(1, int(len(available) * dist))
                used = random.sample(available, min(count, len(available)))
            else:
                raise ValueError(f"Unknown distribution type: {dist}")

            # Place objects at selected positions
            for pos in used:
                x, y = pos
                obj = self.create_object(obj_type, colour, status)
                if obj_type == "agent":
                    agent_positions.append((x, y))
                elif obj_type == "key":
                    key_positions.append((x, y, colour))
                elif obj_type == "goal":
                    goal_positions.append((x, y))
                self.grid.set(x, y, obj)
                if obj_type != "door":
                    filled.add((x, y))

        # Determine agent position
        if agent_positions:
            self.agent_pos = random.choice(agent_positions)
        else:
            raise ValueError("No available agent position")

        # Set agent orientation
        dir_map = {"right": 0, "down": 1, "left": 2, "up": 3, "random": random.randint(0, 3)}
        self.agent_dir = flip_direction(rotate_direction(dir_map.get(orientation, 0), image_direction), flip)

        # No object is carried at start
        self.carrying = None

    def create_object(self, obj_type: str, color: Optional[str], status: Optional[str]) -> Optional[WorldObj]:
        """
        Create a MiniGrid object given type, color and status.

        Args:
            obj_type: Type of the object, like "wall", "key", "door", etc.
            color: Optional color (e.g., red, blue)
            status: Optional status (e.g., locked, open)

        Returns:
            MiniGrid object or None
        """
        if obj_type == "wall":
            return Wall()
        elif obj_type == "floor":
            return Floor()
        elif obj_type == "key":
            return Key(color)
        elif obj_type == "ball":
            return Ball(color)
        elif obj_type == "box":
            return Box(color)
        elif obj_type == "lava":
            return Lava()
        elif obj_type == "goal":
            return Goal()
        elif obj_type == "door":
            is_open = status == "open"
            is_locked = status == "locked"
            return Door(color, is_open=is_open, is_locked=is_locked)
        elif obj_type == "agent":
            return None  # agent is handled separately
        else:
            return None

    def _door_toggle_any_colour(self, door,):
        # If the player has the right key to open the door
        if door.is_locked:
            if isinstance(self.carrying, Key):
                door.is_locked = False
                door.is_open = True
                return True
            return False

        door.is_open = not door.is_open
        return True

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Reset the environment. Always regenerates the grid based on the JSON layout.
        """
        # Reset cumulative reward for sparse mode
        self.cumulative_reward = 0.0

        # Generate new grid from JSON layout
        self._gen_grid(self.width, self.height)

        if self.render_mode == "human":
            self.render()

        self.step_count = 0

        obs = self.gen_obs()

        # Encode carrying info
        obs["carrying"] = {
            "carrying": 1,
            "carrying_colour": 0,
        }
        if self.carrying is not None:
            obs["carrying"] = {
                "carrying": OBJECT_TO_IDX[self.carrying.type],
                "carrying_colour": COLOR_TO_IDX[self.carrying.color],
            }

        # Encode overlap info
        overlap = self.grid.get(*self.agent_pos)
        obs["overlap"] = {
            "obj": OBJECT_TO_IDX[overlap.type] if overlap else 0,
            "colour": COLOR_TO_IDX[overlap.color] if overlap else 0,
        }

        return obs, {}

    def step(
            self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        self.step_count += 1

        # Initialize reward with step penalty
        reward = self.reward_config["step_penalty"]

        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self.reward_config["goal_reward"]
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True
                reward = self.reward_config["lava_penalty"]

        # Unified toggle action (uni_toggle)
        elif action == self.actions.uni_toggle:
            # Case 1: Forward cell exists (not empty)
            if fwd_cell:
                # If carrying nothing and forward cell can be picked up, perform pickup
                if self.carrying is None and fwd_cell.can_pickup():
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)
                    reward += self.reward_config["pickup_reward"]

                # If forward cell is a door, perform toggle
                elif fwd_cell.type == "door":
                    was_open = fwd_cell.is_open
                    if self.any_key_opens_the_door:
                        self._door_toggle_any_colour(fwd_cell, )
                    else:
                        fwd_cell.toggle(self, fwd_pos)
                    # Update rewards based on door status
                    if fwd_cell.is_open and not was_open:
                        reward += self.reward_config["door_open_reward"]
                    elif not fwd_cell.is_open and was_open:
                        reward += self.reward_config["door_close_penalty"]

            # Case 2: Forward cell is empty (None)
            else:
                # If carrying an object, drop it in the empty space
                if self.carrying:
                    self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                    self.carrying.cur_pos = fwd_pos

                    # Special case: if dropping a key, give different reward
                    if self.carrying.type == "key":
                        reward += self.reward_config["key_drop_penalty"]
                    else:
                        reward += self.reward_config["item_drop_penalty"]

                    self.carrying = None

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        # Handle sparse vs dense reward mode
        if self.reward_config["sparse"]:
            # Accumulate reward but only return it at termination
            self.cumulative_reward += reward
            if terminated or truncated:
                # Return total accumulated reward
                returned_reward = self.cumulative_reward
            else:
                # Return zero reward for non-terminal steps
                returned_reward = 0.0
        else:
            # Return immediate reward (dense mode)
            returned_reward = reward

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        # Set carrying observation
        obs["carrying"] = {
            "carrying": 1,
            "carrying_colour": 0,
        }

        if self.carrying is not None and self.carrying != 0:
            carrying = OBJECT_TO_IDX[self.carrying.type]
            carrying_colour = COLOR_TO_IDX[self.carrying.color]

            obs["carrying"] = {
                "carrying": carrying,
                "carrying_colour": carrying_colour,
            }

        # Set overlap observation
        obs["overlap"] = {
            "obj": 0,
            "colour": 0,
        }

        overlap = self.grid.get(*self.agent_pos)
        if overlap is not None:
            overlap_colour = COLOR_TO_IDX[overlap.color]
            obs["overlap"] = {
                "obj": OBJECT_TO_IDX[overlap.type],
                "colour": overlap_colour,
            }

        return obs, returned_reward, terminated, truncated, {}

    def set_env_by_obs(self, obs: ObsType):
        """
        NOTES: setting the environment this way, Box will always be empty!!!
        """
        # self.skip_reset = True
        # values needed:
        # self.agent_pos, self.agent_dir
        # self.grid needs to be reset
        # self.carrying, and everything within this carried object
        image = obs["image"]
        object_channel = image[:, :, 0]
        indices = np.argwhere(object_channel == OBJECT_TO_IDX["agent"])
        assert len(indices) == 1, "Only one agent can be in the map."
        self.agent_pos = tuple(indices[0])
        self.agent_dir = image[:, :, 2][self.agent_pos]
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                obj = self.int_to_object(int(image[x, y, 0]), IDX_TO_COLOR[image[x, y, 1]])
                if obj is not None and obj.type == "door":
                    obj.is_open = image[x, y, 2] == STATE_TO_IDX["open"]
                    obj.is_locked = image[x, y, 2] == STATE_TO_IDX["locked"]
                self.grid.set(x, y, obj)
        if obs["overlap"]["obj"] is not None:
            obj = self.int_to_object(obs["overlap"]["obj"][0], IDX_TO_COLOR[obs["overlap"]["colour"][0]])
            if obj is not None and obj.type == "door":
                obj.is_open = True  # overlap - for sure it's open
            self.grid.set(self.agent_pos[0], self.agent_pos[1], obj)
        self.carrying = self.int_to_object(obs['carrying']['carrying'][0], IDX_TO_COLOR[obs['carrying']['carrying_colour'][0]])
        if self.carrying is not None:
            self.carrying.cur_pos = np.array([-1, -1])
        self.skip_reset = True
        return self.reset()

    def char_to_colour(self, char: str) -> Optional[str]:
        """
        Maps a single character to a color name supported by MiniGrid objects.

        Args:
            char (str): A character representing a color.

        Returns:
            Optional[str]: The name of the color, or None if the character is not recognized.
        """
        color_map = {'R': 'red', 'G': 'green', 'B': 'blue', 'P': 'purple', 'Y': 'yellow', 'E': 'grey', '_': '_'}
        return color_map.get(char.upper(), None)

    def char_to_object(self, char: str, color: str) -> Optional[WorldObj]:
        """
        Maps a character (and its associated color) to a MiniGrid object.

        Args:
            char (str): A character representing an object type.
            color (str): The color of the object.

        Returns:
            Optional[WorldObj]: The MiniGrid object corresponding to the character and color, or None if unrecognized.
        """
        obj_map = {
            'W': lambda: Wall(), 'F': lambda: Floor(), 'B': lambda: Ball(color),
            'K': lambda: Key(color), 'X': lambda: Box(color), 'D': lambda: Door(color, is_locked=True),
            'G': lambda: Goal(), 'L': lambda: Lava(),
        }
        constructor = obj_map.get(char, None)
        return constructor() if constructor else None

    def int_to_object(self, val: int, color: str) -> Optional[WorldObj]:
        obj_str = IDX_TO_OBJECT[val]
        obj_map = {
            'wall': lambda: Wall(), 'floor': lambda: Floor(), 'ball': lambda: Ball(color),
            'key': lambda: Key(color), 'box': lambda: Box(color), 'door': lambda: Door(color, is_locked=True),
            'goal': lambda: Goal(), 'lava': lambda: Lava(),
        }
        constructor = obj_map.get(obj_str, None)
        return constructor() if constructor else None


def rotate_coordinate(x, y, rotation_mode, n):
    """
    Rotate a 2D coordinate in a gridworld.

    Parameters:
    x, y (int): Original coordinates.
    rotation_mode (int): Rotation mode (0, 1, 2, 3).
    n (int): Dimension of the matrix.

    Returns:
    tuple: The new coordinates (new_x, new_y) after rotation.
    """
    if rotation_mode == 0:
        # No rotation
        return x, y
    elif rotation_mode == 1:
        # Clockwise rotation by 90 degrees
        return y, n - 1 - x
    elif rotation_mode == 2:
        # Clockwise rotation by 180 degrees
        return n - 1 - x, n - 1 - y
    elif rotation_mode == 3:
        # Clockwise rotation by 270 degrees
        return n - 1 - y, x
    else:
        raise ValueError("Invalid rotation mode. Please choose between 0, 1, 2, or 3.")


def flip_coordinate(x, y, flip_mode, n):
    """
    Flip a 2D coordinate in a gridworld along the x-axis.

    Parameters:
    x, y (int): Original coordinates.
    flip_mode (int): Flip mode (0 for no flip, 1 for flip along x-axis).
    n (int): Dimension of the matrix.

    Returns:
    tuple: The new coordinates (new_x, new_y) after flipping.
    """
    if flip_mode == 0:
        # No flip
        return x, y
    elif flip_mode == 1:
        # Flip along the x-axis
        return n - 1 - x, y
    else:
        raise ValueError("Invalid flip mode. Please choose between 0 and 1.")


def rotate_direction(direction, rotation_mode):
    """
    Rotate a direction in a gridworld.

    Parameters:
    direction (int): Original direction (0, 1, 2, 3).
    rotation_mode (int): Rotation mode (0, 1, 2, 3).

    Returns:
    int: New direction after rotation.
    """
    # Apply rotation by adding the rotation mode and taking modulo 4 to cycle back to 0 after 3.
    if rotation_mode in [0, 1, 2, 3]:
        return (direction + rotation_mode) % 4
    else:
        raise ValueError("Invalid rotation mode. Please choose between 0, 1, 2, or 3.")


def flip_direction(direction, flip_mode):
    """
    Flip a direction in a gridworld along the x-axis.

    Parameters:
    direction (int): Original direction (0, 1, 2, 3).
    flip_mode (int): Flip mode (0 for no flip, 1 for flip).

    Returns:
    int: New direction after flipping.
    """
    if flip_mode == 0:
        # No flip
        return direction
    elif flip_mode == 1:
        # Flip direction: right/left remain the same, up/down are flipped
        flip_map = {0: 0, 1: 3, 2: 2, 3: 1}
        return flip_map[direction]
    else:
        raise ValueError("Invalid flip mode. Please choose between 0 and 1.")


if __name__ == "__main__":
    env = CustomEnv(
        json_file_path='../maps/door-key.json',
        config=None,
        display_size=None,
        display_mode="random",
        random_rotate=True,
        random_flip=True,
        render_carried_objs=True,
        render_mode="human",
    )
    manual_control = SimpleManualControl(env)  # Allows manual control for testing and visualization
    manual_control.start()  # Start the manual control interface
