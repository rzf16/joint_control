'''
Cost function generation functions
Author: rzfeng
'''
from typing import Callable, Dict

import torch

from src.vehicles import Vehicle
from src.system import VehicleSystem

# Generates an MPPI cost function for a goal-reaching objective
# @input goal [torch.tensor (state_dim)]: goal state
# @input Q [torch.tensor (state_dim x state_dim)]: state weight matrix
# @input R [torch.tensor (control_dim x control_dim)]: control weight matrix
# output [function(torch.tensor (B), torch.tensor (B x T x state_dim), torch.tensor (B x T x control_dim)) -> torch.tensor (B)]:
#       MPPI cost function
def generate_goal_cost(goal: torch.tensor, Q: torch.tensor, R: torch.tensor) -> Callable[[torch.tensor, torch.tensor, torch.tensor], torch.tensor]:
    # Goal-reaching cost function for MPPI, computing the cost for a batch of state and control trajectories
    # @input t [torch.tensor (B)]: batch of initial timesteps
    # @input s [torch.tensor (B x T x state_dim)]: batch of state trajectories
    # @input u [torch.tensor (B x T x control_dim)]: batch of control trajectories
    # @output [torch.tensor (B)]: batch of costs
    def quad_goal_cost(t: torch.tensor, s: torch.tensor, u: torch.tensor) -> torch.tensor:
        # Not using any control cost for now, since MPPI has its own control cost formula!
        B = s.size(0)
        T = s.size(1)
        state_dim = s.size(2)
        # control_dim = u.size(2)
        batch_Q = Q.repeat(B*T,1,1)
        # batch_R = R.repeat(B*T,1,1)
        diffs = goal - s

        cost = torch.bmm(diffs.reshape(B*T,1,state_dim), torch.bmm(batch_Q, diffs.reshape(B*T, state_dim, 1))).reshape(B,T).sum(dim=1)
        # cost += torch.bmm(u.reshape(B*T,1,control_dim), torch.bmm(batch_R, u.reshape(B*T, control_dim, 1))).reshape(B,T).sum(dim=1)
        return cost
    return quad_goal_cost


# Generates an MPPI cost function for a trajectory-tracking objective
# @input ref [torch.tensor (T_ref x state_dim)]: reference trajectory WITH THE SAME TIME DISCRETIZATION AS THE CONTROLLER
# @input Q [torch.tensor (state_dim x state_dim)]: state weight matrix
# @input R [torch.tensor (control_dim x control_dim)]: control weight matrix
# output [function(torch.tensor(B), torch.tensor (B x T x state_dim), torch.tensor (B x T x control_dim)) -> torch.tensor (B)]:
#       MPPI cost function
def generate_traj_cost(ref: torch.tensor, Q: torch.tensor, R: torch.tensor) -> Callable[[torch.tensor, torch.tensor, torch.tensor], torch.tensor]:
    # Trajectory-tracking cost function for MPPI, computing the cost for a batch of state and control trajectories
    # @input t [torch.tensor (B)]: batch of initial timesteps
    # @input s [torch.tensor (B x T x state_dim)]: batch of state trajectories
    # @input u [torch.tensor (B x T x control_dim)]: batch of control trajectories
    # @output [torch.tensor (B)]: batch of costs
    def quad_traj_cost(t: torch.tensor, s: torch.tensor, u: torch.tensor) -> torch.tensor:
        # Not using any control cost for now, since MPPI has its own control cost formula!
        B = s.size(0)
        T = s.size(1)
        T_ref = ref.size(0)
        state_dim = s.size(2)
        # control_dim = u.size(2)

        # This assumes t is filled with the same element, but it's REALLY hard to get around this assumption without complex
        # logic to deal with the different length of each trajectory difference!
        ref_end = int(min(t[0] + T, T_ref))
        T_diff = ref_end - int(t[0])
        # Turns into a goal cost if we have passed the time horizon of the trajectory!
        diffs = ref[int(t[0]):ref_end,:] - s[:,:T_diff,:] if T_diff > 0 else ref[-1,:] - s
        T_diff = T if T_diff <= 0 else T_diff

        batch_Q = Q.repeat(B*T_diff,1,1)
        # batch_R = R.repeat(B*T_diff,1,1)

        cost = torch.bmm(diffs.reshape(B*T_diff,1,state_dim), torch.bmm(batch_Q, diffs.reshape(B*T_diff, state_dim, 1))).reshape(B,T_diff).sum(dim=1)
        # cost += torch.bmm(u.reshape(B*T_diff,1,control_dim), torch.bmm(batch_R, u.reshape(B*T_diff, control_dim, 1))).reshape(B,T_diff).sum(dim=1)
        return cost
    return quad_traj_cost


# TODO: put collision checking in vehicles?
# Generates an MPPI terminal cost function for avoidance of cylindrical obstacles and of the ground
# @input vehicle [Vehicle]: Vehicle object
# @input obstacles [torch.tensor (N x 4)]: obstacles specified in the form (x, y, radius, height)
# @input collision_cost [float]: cost of a collision on the trajectory
# @output [function(torch.tensor (B x T x state_dim)) -> torch.tensor (B)]: function computing collision costs for a batch of
#       state trajectories
def generate_obstacle_cost(vehicle: Vehicle, obstacles: torch.tensor, collision_cost: float):
    # MPPI collision terminal cost function
    # @input s [torch.tensor (B x T x state_dim)]: batch of state trajectories
    # @output [torch.tensor (B)]: batch of collision costs
    def obstacle_cost(s: torch.tensor) -> torch.tensor:
        B = s.size(0)
        T = s.size(1)
        state_dim = s.size(2)
        N = obstacles.size(0)
        pos = vehicle.get_pos3d(s.reshape(B*T, state_dim)).reshape(B, T, 3)

        # Check for ground collision
        ground_collision = torch.any(pos[:,:,2] - 0.5*vehicle.height < 0.0, dim=1)

        # Check for obstacle collisions
        batch_pos = pos.repeat(N,1,1,1)
        obstacle_collision = torch.any(torch.logical_and(
            torch.any(batch_pos[:,:,:,2] - 0.5*vehicle.height < obstacles[:,3], dim=2),
            torch.any(torch.linalg.norm(batch_pos[:,:,:,:2] - obstacles[:,:2].reshape(N,1,1,2), dim=3) < \
                      (vehicle.radius + obstacles[:,2]), dim=2)
        ).transpose(0,1), dim=1)

        return collision_cost * torch.logical_or(ground_collision, obstacle_collision).float()
    return obstacle_cost


# Generates an MPPI terminal cost function for self-collision avoidance within the system
# @input system [VehicleSystem]: the system
# @input collision_cost [float]: cost of a collision on the trajectory
# @output [function(torch.tensor (B x T x state_dim)) -> torch.tensor (B)]: function computing collision costs for a batch of
#       state trajectories
def generate_collision_cost(system: VehicleSystem, collision_cost: float):
    # MPPI self-collision terminal cost function
    # @input s [torch.tensor (B x T x state_dim)]: batch of state trajectories
    # @output [torch.tensor (B)]: batch of collision costs
    def collision_cost_fn(s: torch.tensor) -> torch.tensor:
        B = s.size(0)
        T = s.size(1)
        collision = torch.zeros(B, dtype=torch.bool)
        # There's probably a way to vectorize this, but not sure if it's worth the effort/complexity
        for i, (vehicle_name1, vehicle1) in enumerate(system.vehicles.items()):
            for j, (vehicle_name2, vehicle2) in enumerate(system.vehicles.items()):
                if i <= j:
                    continue
            
                s1 = s[:,:,system.state_idxs[vehicle_name1][0]:system.state_idxs[vehicle_name1][1]]
                s2 = s[:,:,system.state_idxs[vehicle_name2][0]:system.state_idxs[vehicle_name2][1]]
                pos1 = vehicle1.get_pos3d(s1.reshape(B*T, vehicle1.state_dim())).reshape(B, T, 3)
                pos2 = vehicle2.get_pos3d(s2.reshape(B*T, vehicle2.state_dim())).reshape(B, T, 3)

                pair_collision = torch.logical_and(
                    torch.logical_not(torch.logical_or(
                        pos1[:,:,2] - 0.5*vehicle1.height > pos2[:,:,2] + 0.5*vehicle2.height,
                        pos2[:,:,2] - 0.5*vehicle2.height > pos1[:,:,2] + 0.5*vehicle1.height,
                    )),
                    torch.linalg.norm(pos1[:,:,:2] - pos2[:,:,:2], dim=2) < vehicle1.radius + vehicle2.radius
                )
                collision = torch.logical_or(collision, torch.any(pair_collision, dim=1))
        return collision_cost * collision.float()
    return collision_cost_fn