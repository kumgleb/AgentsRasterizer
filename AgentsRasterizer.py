import numpy as np
import cv2
import colorsys
from typing import Dict, Tuple


# TBD: define for corresponding classes
default_colors = {
    'vehicle': (255, 255, 0),
    'pedestrian': (255, 153, 51),
    'target': (255, 0, 0)
}


class AgentsRasterizer:
    """
    Class for nuScenes like scene representation.
    Target agent at current moment is always represented at the same position in the image.
    History represented as faded bounding boxes.
    """

    def __init__(self,
                 seconds_of_history: float = 2,  # [s] max value of available history
                 resolution: float = 0.1,        # [meters / pixel]
                 meters_ahead: float = 40,       # rasterized distance ahead of the target agent
                 meters_behind: float = 10,      # rasterized distance behind of the target agent
                 meters_left: float = 25,        # rasterized distance left of the target agent
                 meters_right: float = 25,       # rasterized distance right of the target agent
                 color_mapping: Dict = None,     # color mapping of considered classes
                 fade_lowest_value: float = 0.4  # lowest value in HSV
                 ):

        assert seconds_of_history >= 0, f'`seconds_of_history` should be non negative, provided: {seconds_of_history}.'
        self.seconds_of_history = seconds_of_history

        assert resolution > 0, f'`resolution` should be positive, : {resolution}.'
        self.resolution = resolution

        assert meters_ahead > 0, f'`meters_ahead` should be positive, provided: {meters_ahead}.'
        assert meters_behind > 0, f'`meters_behind` should be positive, provided: {meters_behind}.'
        assert meters_left > 0, f'`meters_left` should be positive, provided: {meters_left}.'
        assert meters_right > 0, f'`meters_right` should be positive, provided: {meters_right}.'
        self.meters_ahead = meters_ahead
        self.meters_behind = meters_behind
        self.meters_left = meters_left
        self.meters_right = meters_right

        self.color_mapping = color_mapping if color_mapping is not None else default_colors

        assert fade_lowest_value > 0, f'`fade_lowest_value` should be positive, provided: {fade_lowest_value}.'
        self.fade_lowest_value = fade_lowest_value

        self.img_center_pix = None

    def get_agents_history(self, tokens: list) -> Dict:
        """
        Collect history of agents by their tokens.
        :param tokens: list of agents tokens.
        :return: Dict with agent histories like:
            {'agent_1_token':
                {'coordinates': np.ndarray, shape: [2, n_history_points],
                 'yaw_angle': np.ndarray, shape: [n_history_points],
                 'bbox': np.ndarray, shape: [2],
                 'class': str},
             'agent_2_token': ...}
        """
        raise NotImplementedError

    def get_surrounding_agents(self, target_agent_token: str, radius_of_interest: float) -> list:
        """
        Collect tokens of agents that surround target object.
        :param target_agent_token: token of target agent.
        :param radius_of_interest: radius for surrounding agents to be considered [m].
        :return: list of tokens.
        """
        raise NotImplementedError

    def reverse_history(self, agents_history: Dict) -> Dict:
        """
        Helper function for convenience.
        Helps to draw more recent bboxes on top of older ones.
        Item with idx 0 - current moment, last item - latest moment.
        :param agents_history: Dict with object history.
        :return: Dict with reversed object history.
        """
        for agent_token in agents_history.keys():
            agents_history[agent_token]['coordinates'] = agents_history[agent_token]['coordinates'][::-1]
            agents_history[agent_token]['yaw_angle'] = agents_history[agent_token]['yaw_angle'][::-1]
        return agents_history

    def coords_to_row_col_pixels(self, coords_xy: np.ndarray, target_coords_xy: np.ndarray) -> Tuple[int, int]:
        """
        Transform global agent coordinates to pixels in target agent coordinates system.
        :param coords_xy: agent coordinates in global coordinate system in [m].
        :param target_coords_xy: target agent coordinates in global coordinate system in [m].
        :return: [row_pix, col_pix] - pixel values of agent center.
        """
        # Offset with respect to target agent.
        offset_x = coords_xy[0] - target_coords_xy[0]
        offset_y = coords_xy[1] - target_coords_xy[1]

        # To pixels. Note that now (0, 0) is the top left corner.
        x_pix = offset_x / self.resolution
        y_pix = -offset_y / self.resolution

        row_pix = int(self.img_center_pix[0] + y_pix)
        col_pix = int(self.img_center_pix[1] + x_pix)

        return row_pix, col_pix

    def get_bbox(self,
                 coords_xy: np.ndarray, yaw_rad: np.ndarray,
                 bbox_size: np.ndarray, target_coords_xy: np.ndarray) -> np.ndarray:
        """
        Create bbox as four corners in pixels.
        :param coords_xy: agent coordinates in global coordinate system in [m].
        :param yaw_rad: agent yaw in global coordinate system in [rad].
        :param bbox_size: agent bbox width and length in [m].
        :param target_coords_xy: target agent coordinates in global coordinate system in[m].
        :return: coordinates of four bbox corners in [pix].
        """
        row_pix_bbox_center, col_pix_bbox_center = self.coords_to_row_col_pixels(coords_xy, target_coords_xy)

        width = bbox_size[0] / self.resolution  # [pix]
        length = bbox_size[1] / self.resolution  # [pix]

        # Note that cv2 flip rows and cols to (x, y) plane, and treat clockwise rotation as positive.
        bbox_data = ((col_pix_bbox_center, row_pix_bbox_center), (length, width), -yaw_rad * 180 / np.pi)
        bbox = cv2.boxPoints(bbox_data)
        return bbox

    def fade_color(self,
                   base_color: Tuple[int, int, int],
                   step: int,
                   n_steps: int) -> Tuple[int, int, int]:
        """
        Fades color: older -> darker.
        :param base_color: base RGB color.
        :param step: number of current time step.
        :param n_steps: total number of time steps.
        :return: Tuple with faded RGB color.
        """

        if step == n_steps:
            return base_color

        hsv_color = colorsys.rgb_to_hsv(*base_color)
        increment = float(hsv_color[2]/255. - self.fade_lowest_value) / n_steps
        new_color = self.fade_lowest_value + step * increment
        new_rgb = colorsys.hsv_to_rgb(float(hsv_color[0]),
                                      float(hsv_color[1]),
                                      new_color * 255.)
        return new_rgb

    def draw_agents(self,
                    base_image: np.ndarray,
                    all_agents_history: Dict,
                    target_agent_token: str) -> np.ndarray:
        """
        Draw scene in target agent coordinate system.
        :param base_image: np.zeros, image background.
        :param all_agents_history: Dict with all agents of interest histories.
        :param target_agent_token: token of target agent.
        :return: np.ndarray with rendered agents.
        """

        # Coordinates of the target agent at the current moment.
        target_coords_xy = all_agents_history[target_agent_token]['coordinates'][:, 0]

        for agent_token, agent_data in all_agents_history.items():
            # Parse agent data
            coordinates = agent_data['coordinates']
            yaw_angles = agent_data['yaw_angle']
            bbox_size = agent_data['bbox']  # agent bbox width and length
            agent_class = 'target' if agent_token == target_agent_token else agent_data['class']

            base_color = self.color_mapping[agent_class]
            n_points = coordinates.shape[1]  # number of history points

            # Draw history
            for i in range(n_points):
                point_coords_xy = coordinates[i]
                yaw_rad = yaw_angles[i]  # [rad]
                bbox = self.get_bbox(point_coords_xy, yaw_rad, bbox_size, target_coords_xy)

                # Don't fade if no history available
                if n_points > 1:
                    color = self.fade_color(base_color, i, n_points - 1)

                base_image = cv2.fillPoly(base_image, pts=[np.int0(bbox)], color=color)

        return base_image

    def rotate_and_crop(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image to face current position of target agent vertically, than crop the image.
        :param image: np.ndattay with rasterized agents.
        :param angle: yaw angle of target agent at the current moment [rad].
        :return: rotated and cropped image in uint8.
        """

        angle_deg = angle * 180 / np.pi  # [deg]
        rotation_matrix = cv2.getRotationMatrix2D(self.img_center_pix, angle_deg, 1)

        rotated_image = cv2.wrapAffine(image, rotation_matrix, self.img_center_pix)

        row_crop = slice(0, int((self.meters_ahead + self.meters_behind) / self.resolution))
        col_crop = slice(int(self.img_center_pix[0] - self.meters_left / self.resolution),
                         int(self.img_center_pix[1] - self.meters_right / self.resolution))

        cropped_image = rotated_image[row_crop, col_crop].astype('uint8')

        return cropped_image

    def draw_representation(self, target_agent_token: str) -> np.ndarray:
        """
        Draw scene representation.
        :param target_agent_token:
        :return:
        """

        # Image preparation
        radius_of_interest = max([self.meters_ahead, self.meters_behind,
                                  self.meters_left, self.meters_right]) * 2  # [m]
        image_side_length = radius_of_interest // self.resolution  # [pix]
        base_image = np.zeros((image_side_length, image_side_length, 3))
        self.img_center_pix = np.array(image_side_length // 2, image_side_length // 2)

        # Collect agents of interest
        surrounding_agents_tokens = self.get_surrounding_agents(target_agent_token, radius_of_interest)

        # Collect history of objects to be drawn at the final raster
        all_tokens = [target_agent_token] + surrounding_agents_tokens
        agents_history = self.get_agents_history(all_tokens)

        # Reverse history for convenience
        agents_history = self.reverse_history(agents_history)

        # Build final raster
        final_raster = self.draw_agents(base_image, agents_history, target_agent_token)

        current_moment_target_yaw_rad = agents_history[target_agent_token]['yaw'][0]
        final_raster = self.rotate_and_crop(final_raster, current_moment_target_yaw_rad)

        return final_raster






