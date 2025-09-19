"""
Classic cart-pole system implemented by Rich Sutton et al. generalized to C carts connected with springs and dampers.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
from typing import Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


class CARTerpillarEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ## Description

    This environment corresponds to a generalization to C carts of the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.
    The C carts are connected within each other by means of springs and dampers, so that each movement applied to a cart has an impact on all
    other carts. In this way, the parameter C scales with the difficulty of the environment in terms of state and action space, and of dynamics.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1, ..., 2C-1}` indicating the direction
     of the fixed force one of the cart is pushed with.

    - 0: Push first cart to the left
    - 1: Push first cart to the right
    - 2: Push second cart to the left
    - 3: Push second cart to the right
    ...
    - 2C-2: Push C-th cart to the left
    - 2C-1: Push C-th cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4C,)` with the values corresponding to the following positions and velocities:

    | Num    | Observation               | Min                 | Max               |
    |--------|---------------------------|---------------------|-------------------|
    | 0      | 1st Cart Position         | -4.8                | 4.8               |
    | 1      | 1st Cart Velocity         | -Inf                | Inf               |
    | 2      | 1st Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3      | 1st Pole Angular Velocity | -Inf                | Inf               |
    ...
    | 4C-4   | Cth Cart Position         | -4.8                | 4.8               |
    | 4C-3   | Cth Cart Velocity         | -Inf                | Inf               |
    | 4C-2   | Cth Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 4C-1   | Cth Pole Angular Velocity | -Inf                | Inf               |
    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards
    Since the goal is to keep the poles upright for as long as possible, by default, a reward of `+1` is given for every step taken, including the termination step. The default reward threshold is 500 for v1 and 200 for v0 due to the time limit on the environment.

    If `sutton_barto_reward=True`, then a reward of `0` is awarded for every non-terminating step and `-1` for the terminating step. As a result, the reward threshold is 0 for v0 and v1.

    ## Starting State
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End
    The episode ends if any one of the following occurs:

    1. Termination: One of the Pole Angles is greater than ±12°
    2. Termination: One of the Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ## Arguments

    Cartpole only has `render_mode` as a keyword for `gymnasium.make`.
    On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.


    | Parameter               | Type       | Default                 | Description                                                                                   |
    |-------------------------|------------|-------------------------|-----------------------------------------------------------------------------------------------|
    | `sutton_barto_reward`   | **bool**   | `False`                 | If `True` the reward function matches the original sutton barto implementation                |
    | `n_poles`               | **int**    | 1                       | Number of carts composing the CARTerpillar                                                    | 
    | `gravity`               | **float**  | 9.8                     | Gravitational constant used upon poles                                                        |      
    | `spring_constant`       | **float**  | 1.0                     | Spring constant impacting all pairs of carts                                                  |      
    | `damper_constant`       | **float**  | 0.1                     | Damper constant impacting all pairs of carts                                                  |      
    """


    def __init__(
        self, sutton_barto_reward: bool = False, render_mode: str | None = None, n_poles: int = 1, gravity: float = 9.8,
        spring_constant: float = 1.0, damper_constant: float = 0.1
    ):
        self._sutton_barto_reward = sutton_barto_reward
        self.n_poles = n_poles

        self.gravity = gravity
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        
        # Spring-damper coupling parameters
        self.spring_constant = spring_constant
        self.damper_constant = damper_constant

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high_single = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )
        high = np.tile(high_single, self.n_poles)

        self.action_space = spaces.Discrete(2 * self.n_poles)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state: np.ndarray | None = None

        self.steps_beyond_terminated = None

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        
        # Determine target cartpole and force direction
        if action < self.n_poles:
            target_cartpole = action
            force_direction = 1
        else:
            target_cartpole = action - self.n_poles
            force_direction = -1
        
        force = self.force_mag * force_direction
        
        # Calculate spring-damper forces between all cart pairs
        spring_damper_forces = np.zeros(self.n_poles)
        
        for i in range(self.n_poles):
            for j in range(self.n_poles):
                if i != j:
                    # Extract positions and velocities for carts i and j
                    xi = self.state[i * 4]
                    xi_dot = self.state[i * 4 + 1]
                    xj = self.state[j * 4]
                    xj_dot = self.state[j * 4 + 1]
                    
                    # Spring force: F = -k * (displacement)
                    spring_force = -self.spring_constant * (xi - xj)
                    
                    # Damper force: F = -c * (relative velocity)
                    damper_force = -self.damper_constant * (xi_dot - xj_dot)
                    
                    spring_damper_forces[i] += spring_force + damper_force
        
        # Update each cartpole
        new_state = []
        terminated = False
        
        for i in range(self.n_poles):
            # Extract state for cartpole i
            start_idx = i * 4
            x, x_dot, theta, theta_dot = self.state[start_idx:start_idx + 4]
            
            # Apply control force only to target cartpole
            control_force = force if i == target_cartpole else 0.0
            
            # Total force includes control force and spring-damper forces
            total_force = control_force + spring_damper_forces[i]
            
            costheta = np.cos(theta)
            sintheta = np.sin(theta)

            # For the interested reader:
            # https://coneural.org/florian/papers/05_cart_pole.pdf
            temp = (
                total_force + self.polemass_length * np.square(theta_dot) * sintheta
            ) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length
                * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
            )
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

            if self.kinematics_integrator == "euler":
                x = x + self.tau * x_dot
                x_dot = x_dot + self.tau * xacc
                theta = theta + self.tau * theta_dot
                theta_dot = theta_dot + self.tau * thetaacc
            else:  # semi-implicit euler
                x_dot = x_dot + self.tau * xacc
                x = x + self.tau * x_dot
                theta_dot = theta_dot + self.tau * thetaacc
                theta = theta + self.tau * theta_dot

            new_state.extend([x, x_dot, theta, theta_dot])
            
            # Check termination for this cartpole
            if (x < -self.x_threshold
                or x > self.x_threshold
                or theta < -self.theta_threshold_radians
                or theta > self.theta_threshold_radians):
                terminated = True

        self.state = np.array(new_state, dtype=np.float64)

        if not terminated:
            reward = 0.0 if self._sutton_barto_reward else 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = -1.0 if self._sutton_barto_reward else 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = -1.0 if self._sutton_barto_reward else 0.0

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4 * self.n_poles,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        raise Exception("Render method not implemented yet!")


    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False