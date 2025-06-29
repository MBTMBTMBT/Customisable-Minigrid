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
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.core import ActType, ObsType
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, TILE_PIXELS
from PIL import Image, ImageDraw, ImageFont


class SimpleActions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    uni_toggle = 3


def _door_toggle_any_colour(door, env, pos):
    # If the player has the right key to open the door
    if door.is_locked:
        if isinstance(env.carrying, Key):
            door.is_locked = False
            door.is_open = True
            return True
        return False

    door.is_open = not door.is_open
    return True


class CustomEnv(MiniGridEnv):
    """
    A custom MiniGrid environment that loads its layout and object properties from a text file.

    Attributes:
        txt_file_path (str): Path to the text file containing the environment layout.
        layout_size (int): The size of the environment, either specified or determined from the file.
        agent_start_pos (tuple[int, int]): Starting position of the agent.
        agent_start_dir (int): Initial direction the agent is facing.
        mission (str): Custom mission description.
    """

    def __init__(
            self,
            txt_file_path: str,
            display_size: Optional[int] = None,
            display_mode: Optional[str] = "middle",
            random_rotate: bool = False,
            random_flip: bool = False,
            agent_start_pos: Optional[Tuple[int, int]] = None,
            agent_start_dir: Optional[int] = None,
            custom_mission: str = "Explore and interact with objects.",
            max_steps: int = 100000,
            render_carried_objs: bool = True,
            any_key_opens_the_door: bool = False,
            rand_colours: Optional[List[str]] = None,
            **kwargs,
    ) -> None:
        """
        Initialize the custom environment using a structured JSON layout file.
        Random map generation is removed. Only JSON file is supported.
        """
        assert txt_file_path is not None, "'txt_file_path' (JSON file) must be provided."

        if rand_colours is None:
            rand_colours = ['R', 'G', 'B', 'P', 'Y', 'E']

        self.txt_file_path = txt_file_path
        self.rand_colours = rand_colours
        self.display_mode = display_mode
        self.random_rotate = random_rotate
        self.random_flip = random_flip
        self.render_carried_objs = render_carried_objs
        self.any_key_opens_the_door = any_key_opens_the_door
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.mission = custom_mission
        self.tile_size = 16
        self.skip_reset = False

        # Read JSON layout once to determine environment size
        with open(self.txt_file_path, 'r') as f:
            config = json.load(f)
        height, width = config["height_width"]
        layout_size = max(height, width)

        if display_size is None:
            self.display_size = layout_size
        else:
            self.display_size = display_size

        assert display_mode in ["middle", "random"]
        assert self.display_size >= layout_size

        if self.agent_start_dir is None:
            self.agent_start_dir = random.randint(0, 3)

        # Initialize parent MiniGridEnv
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: custom_mission),
            grid_size=self.display_size,
            max_steps=max_steps,
            **kwargs,
        )

        self.actions = SimpleActions
        self.action_space = spaces.Discrete(len(self.actions))

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
        Renders the image of the environment with an extra row at the bottom displaying the item
        carried by the agent, if any. The agent can carry at most one item.

        :param full_image: The original image rendered by get_full_render.
        :return: Modified image with an additional row displaying the carried item, if any.
        """
        tile_size = self.tile_size

        carrying_objects = {
            "carrying": 1,
            "carrying_colour": 0,
        }

        # Check if the agent is carrying an object
        if self.carrying is not None and self.carrying != 0:
            carrying = OBJECT_TO_IDX[self.carrying.type]
            carrying_colour = COLOR_TO_IDX[self.carrying.color]

            carrying_objects = {
                "carrying": carrying,
                "carrying_colour": carrying_colour,
            }

        # Prepare to extract carried item and colour indices
        object_idx = carrying_objects.get('carrying', 1)
        colour_idx = carrying_objects.get('carrying_colour', 0)

        # Map indices to actual objects and colours
        object_name = IDX_TO_OBJECT.get(object_idx, "empty")
        colour_name = IDX_TO_COLOR.get(colour_idx, "black")

        # Create a grey background for the carried item row (matching the tile size)
        item_row = np.full((tile_size, tile_size, 3), fill_value=100, dtype=np.uint8)  # Default to grey

        if object_name != "empty":
            # Use the actual symbol for the object rather than the first letter
            symbol = self._get_object_symbol(object_name)
            # Generate a tile with the symbol and colour for the object carried by the agent
            item_row = self._draw_symbol_on_tile(item_row, symbol, colour_name)

        # Extend the original image with this new row at the bottom
        full_height, full_width, _ = full_image.shape

        # Ensure both the full_image and the item_row have the same width (adjust if necessary)
        # Put the item on the right side of the row (align to the bottom-right corner)
        full_image_width = full_image.shape[1]
        item_row_full = np.full((tile_size, full_image_width, 3), fill_value=100, dtype=np.uint8)  # Grey background
        item_row_full[:, -tile_size:, :] = item_row  # Add item to the right

        output_image = np.vstack([full_image, item_row_full])

        return output_image

    def _get_object_symbol(self, object_name):
        """
        Get the letter representing the object.
        This function returns a letter for the object.
        """
        if object_name == "ball":
            return "B"  # Use 'B' to represent the ball
        elif object_name == "box":
            return "X"  # Use 'X' to represent the box
        else:
            # Use the first letter of the object name as its symbol for other objects
            return object_name[0].upper() if object_name else "?"  # Return '?' if the object has no valid name

    def _draw_symbol_on_tile(self, tile, symbol, colour_name="black"):
        """
        Draw the given symbol (a letter) on a larger tile and resize it to the actual tile size.
        This helps improve the clarity and centring of the symbol.

        :param tile: The tile image (a NumPy array) where the symbol will be drawn.
        :param symbol: The symbol (a string, e.g., 'K' for key) to be drawn on the tile.
        :param colour_name: The colour of the object to draw on the tile.
        :return: The tile image with the symbol drawn on it, resized to the original tile size.
        """
        tile_size = tile.shape[0]  # Original tile size
        render_size = int(tile_size * 1.5)

        # Create a larger tile for rendering
        large_tile = np.full((render_size, render_size, 3), fill_value=100, dtype=np.uint8)

        # Convert NumPy array (large tile) to PIL Image
        large_tile_image = Image.fromarray(large_tile)

        # Create a drawing context for the larger tile
        draw = ImageDraw.Draw(large_tile_image)

        # Load a font, use default PIL font if no TTF file is available
        try:
            font = ImageFont.truetype("arial.ttf", size=render_size // 2)  # Larger font size for better clarity
        except IOError:
            font = ImageFont.load_default()

        # Get the size of the large tile
        tile_width, tile_height = large_tile_image.size

        # Get the bounding box of the symbol to centre it on the tile
        bbox = draw.textbbox((0, 0), symbol, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Calculate the position to centre the text
        position = ((tile_width - text_width) // 2, (tile_height - text_height) // 2)

        # Get the colour for the symbol from the colour name
        colour_rgb = COLORS.get(colour_name, [0, 0, 0])  # Default to black if colour_name is invalid

        # Draw a filled rectangle with the colour in the large tile
        draw.rectangle([0, 0, tile_width, tile_height], fill=tuple(colour_rgb))

        # Draw the symbol in the centre of the large tile
        draw.text(position, symbol, font=font, fill=(0, 0, 0))

        # Convert the large tile back to a NumPy array
        large_tile_np = np.array(large_tile_image)

        # Resize the large tile back to the original tile size
        tile_resized = Image.fromarray(large_tile_np).resize((tile_size, tile_size))

        return np.array(tile_resized)

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
        Layers are processed in order; later layers overwrite earlier ones.
        """
        self.grid = Grid(width, height)

        # Load JSON config and metadata
        config = self._read_file()
        H, W = config['height_width']
        layers = config['layers']

        # Calculate placement anchor
        free_width = self.display_size - W
        free_height = self.display_size - H
        if self.display_mode == "middle":
            anchor_x = free_width // 2
            anchor_y = free_height // 2
        elif self.display_mode == "random":
            anchor_x = random.choice(range(max(free_width, 1))) if free_width > 0 else 0
            anchor_y = random.choice(range(max(free_height, 1))) if free_height > 0 else 0
        else:
            raise ValueError("Invalid display mode.")

        image_direction = random.choice([0, 1, 2, 3]) if self.random_rotate else 0
        flip = random.choice([0, 1]) if self.random_flip else 0

        # Track filled positions to prevent overlap
        filled = set()
        key_positions = []
        goal_positions = []
        agent_positions = []
        orientation = "random"

        for layer in layers:
            obj_type = layer["obj"]
            colour = layer.get("colour", None)
            status = layer.get("status", None)
            orientation = layer.get("orientation", "random")
            dist = layer.get("distribution", "all")
            mat = layer["matrix"]

            # Collect all candidate positions
            candidates = [
                (x, y) for y in range(H) for x in range(W) if mat[y][x] == 1
            ]
            candidates = [(anchor_x + x, anchor_y + y) for x, y in candidates]
            candidates = [
                rotate_coordinate(x, y, image_direction, self.display_size) for x, y in candidates
            ]
            candidates = [
                flip_coordinate(x, y, flip, self.display_size) for x, y in candidates
            ]
            candidates = [p for p in candidates if p not in filled]

            if dist == "all":
                used = candidates
            elif dist == "one":
                if len(candidates) < 1:
                    raise ValueError(f"No available positions for object '{layer['name']}'")
                used = [random.choice(candidates)]
            elif isinstance(dist, float):
                count = max(1, int(len(candidates) * dist))
                used = random.sample(candidates, min(count, len(candidates)))
            else:
                raise ValueError(f"Unknown distribution: {dist}")

            for pos in used:
                x, y = pos
                obj = self.create_object(obj_type, colour, status)
                if obj_type == "agent":
                    agent_positions.append((x, y))
                elif obj_type == "key":
                    key_positions.append((x, y, colour))
                elif obj_type == "goal":
                    goal_positions.append((x, y))
                elif obj_type == "door":
                    pass  # already added
                self.grid.set(x, y, obj)
                filled.add(pos)

        # Place agent
        if self.agent_start_pos is not None:
            x, y = self.agent_start_pos
            x, y = anchor_x + x, anchor_y + y
            x, y = rotate_coordinate(x, y, image_direction, self.display_size)
            x, y = flip_coordinate(x, y, flip, self.display_size)
            self.agent_pos = (x, y)
        elif agent_positions:
            self.agent_pos = random.choice(agent_positions)
        else:
            raise ValueError("No available agent position")

        dir_map = {"up": 3, "right": 0, "down": 1, "left": 2, "random": random.randint(0, 3)}
        self.agent_dir = flip_direction(rotate_direction(dir_map.get(orientation, 0), image_direction), flip)

        # Handle carried object (none by default)
        self.carrying = None

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Reset the environment. Always regenerates the grid based on the JSON layout.
        """
        # Generate new grid from JSON layout
        self._gen_grid(self.width, self.height)

        if self.render_mode == "human":
            self.render()

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

        reward = -0.05  # give negative reward for normal steps

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
                reward = 1  # give settled 1 as reward,
                # print("Succeeded!")
                # instead of the original 1 - 0.9 * (self.step_count / self.max_steps)
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True
                reward = -1

            # if fwd_cell is not None and (fwd_cell.type == "wall" or fwd_cell.type == "door" and not fwd_cell.is_open):
            #     reward -= 0.05

        # # Pick up an object
        # elif action == self.actions.pickup:
        #     if fwd_cell and fwd_cell.can_pickup():
        #         if self.carrying is None or self.carrying == 0:
        #             self.carrying = fwd_cell
        #             self.carrying.cur_pos = np.array([-1, -1])
        #             self.grid.set(fwd_pos[0], fwd_pos[1], None)
        #             reward += 0.1
        #             # print("Key picked up!")
        #
        # # Drop an object
        # elif action == self.actions.drop:
        #     if not fwd_cell and self.carrying:
        #         self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
        #         self.carrying.cur_pos = fwd_pos
        #         self.carrying = None
        #         reward -= 0.1
        #
        # # Toggle/activate an object
        # elif action == self.actions.toggle:
        #     if fwd_cell:
        #         was_open = False
        #         if fwd_cell.type == "door" and fwd_cell.is_open:
        #             was_open = True
        #         if fwd_cell.type == "door" and self.any_key_opens_the_door:
        #             _door_toggle_any_colour(fwd_cell, self, fwd_pos)
        #         else:
        #             fwd_cell.toggle(self, fwd_pos)
        #         if fwd_cell.type == "door":
        #             if fwd_cell.is_open:
        #                 if not was_open:
        #                     reward += 0.1
        #                     # print("Door is open!")
        #             else:
        #                 if was_open:
        #                     reward -= 0.1

        # Unified toggle action (uni_toggle)
        elif action == self.actions.uni_toggle:
            # Check if there's a cell in the forward direction
            if fwd_cell:
                # If carrying nothing and forward cell can be picked up, perform pickup
                if self.carrying is None and fwd_cell.can_pickup():
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)
                    reward += 0.1
                    # print("Item picked up!")

                # If carrying an object and forward cell is empty, perform drop
                elif self.carrying and not fwd_cell:
                    self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                    self.carrying.cur_pos = fwd_pos
                    self.carrying = None
                    reward -= 0.1
                    # print("Item dropped!")

                # If forward cell is a door, perform toggle
                elif fwd_cell.type == "door":
                    was_open = fwd_cell.is_open
                    if self.any_key_opens_the_door:
                        _door_toggle_any_colour(fwd_cell, self, fwd_pos)
                    else:
                        fwd_cell.toggle(self, fwd_pos)
                    # Update rewards based on door status
                    if fwd_cell.is_open and not was_open:
                        reward += 0.1
                        # print("Door is open!")
                    elif not fwd_cell.is_open and was_open:
                        reward -= 0.1
                        # print("Door is closed!")

        # Done action (not used by default)
        # elif action == self.actions.done:
        #     pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        obs["carrying"] = {
            "carrying": 1,
            "carrying_colour": 0,
            # "carrying_contains": carrying_contains,
            # "carrying_contains_colour": carrying_contains_colour,
        }

        if self.carrying is not None and self.carrying != 0:
            carrying = OBJECT_TO_IDX[self.carrying.type]
            carrying_colour = COLOR_TO_IDX[self.carrying.color]
            # carrying_contains = 0 if self.carrying.contains is None else OBJECT_TO_IDX[self.carrying.contains.type]
            # carrying_contains_colour = 0 if self.carrying.contains is None else COLOR_TO_IDX[self.carrying.contains.color]

            obs["carrying"] = {
                "carrying": carrying,
                "carrying_colour": carrying_colour,
                # "carrying_contains": carrying_contains,
                # "carrying_contains_colour": carrying_contains_colour,
            }

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

        return obs, reward, terminated, truncated, {}

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
        txt_file_path='./maps/rand_door_key.txt',
        rand_gen_shape=None,
        display_size=None,
        display_mode="random",
        random_rotate=True,
        random_flip=True,
        custom_mission="Find the key and open the door.",
        render_mode="human",
        add_random_door_key=False,
    )
    manual_control = ManualControl(env)  # Allows manual control for testing and visualization
    manual_control.start()  # Start the manual control interface
