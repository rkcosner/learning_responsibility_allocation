from turtle import back
import torch 
from tbsim.utils.geometry_utils import (
    VEH_VEH_collision, 
    VEH_VEH_distance,
    VEH_PED_collision, 
    PED_VEH_collision, 
    PED_PED_collision
)

from tbsim.dynamics import ( 
    Unicycle
)


def unicycle_dynamics(stateA, stateB):
        # states are [x, y, v, yaw]
        N_datapoints = stateA.shape[0]
        fA = torch.zeros(N_datapoints, 4,1, device = stateA.device)
        fA[:,0] = (stateA[:,2] * torch.cos(stateA[:,3]))[:,None]
        fA[:,1] = (stateA[:,2] * torch.sin(stateA[:,3]))[:,None]
        fB = torch.zeros(N_datapoints, 4,1, device = stateA.device)
        fB[:,0] = (stateB[:,2] * torch.cos(stateB[:,3]))[:,None]
        fB[:,1] = (stateB[:,2] * torch.sin(stateB[:,3]))[:,None]
        
        gA = torch.zeros(N_datapoints, 4,2, device = stateA.device)
        gA[:,2,0] = torch.ones(N_datapoints)
        gA[:,3,1] = torch.ones(N_datapoints)

        gB = torch.zeros(N_datapoints, 4,2, device = stateA.device)
        gB[:,2,0] = torch.ones(N_datapoints)
        gB[:,3,1] = torch.ones(N_datapoints)
        return fA, fB, gA, gB

class CBF(torch.nn.Module):
    """
        Differentiable Safety Function
        
    """ 
    def __init__(self): 
        super(CBF, self).__init__()
    def forward(self, state): 
        """
            args: 
                - state [B, A, T, D_states]
            returns: 
                - h_vals [B, A, T-1, 1], the safety values for the trajectory but not including the current state (because we don't have input information for the current state)
        """
        pass 

class NormBallCBF(CBF): 
    """
        Torch auto-grad compatible implementation of a norm ball cbf
    """

    def __init__(self, safe_radius = 1): 
        super(NormBallCBF, self).__init__() 
        self.safe_radius = safe_radius

    def forward(self, states):
        """
            Calculate safety values (computes h for unavailable states as though they were available, mask later)
            Args: 
                - states [B, A+1, T, D_states]
            Return: 
                - h_vals [B, A, T-1]
        """
        
        A = states.shape[1] - 1  # removing ego availability

        # Get the relative positions of the other agents with respect to ego pose for all states that we have inputs for
        ego_pos = states[:,0,1:,0:2]
        ego_pos = ego_pos[:,:,:,None]
        ego_pos = ego_pos.repeat(1,1,1,A)
        ego_pos = ego_pos.permute(0,3,1,2)
        dx_agents = states[:,1:,1:,0:2] - ego_pos

        h_vals  = torch.linalg.norm(dx_agents, ord = 2, axis = -1) # Compute 2 norm for vehicle positions 
        h_vals  = torch.pow(h_vals,2)
        h_vals -= self.safe_radius**2
        return h_vals

class ExtendedNormBallCBF(CBF): 
    """
        Torch auto-grad compatible implementation of a norm ball cbf
    """

    def __init__(self, safe_radius = 0.1, alpha_e = 40): 
        super(ExtendedNormBallCBF, self).__init__() 
        self.safe_radius = safe_radius
        self.alpha_e = alpha_e

    def forward(self, states):
        """
            Calculate safety values (computes h for unavailable states as though they were available, mask later)
            Args: 
                - states [B, A+1, T, D_states] states are [x, y, v, yaw]
            Return: 
                - h_vals [B, A, T-1]
        """
        
        A = states.shape[1] - 1  # removing ego availability
        # Get the relative positions of the other agents with respect to ego pose for all states that we have inputs for
        ego_pos = states[:,0,1:,0:2]
        ego_pos = ego_pos[:,:,:,None]
        ego_pos = ego_pos.repeat(1,1,1,A)
        ego_pos = ego_pos.permute(0,3,1,2)
        Dpos = states[:,1:,1:,0:2] - ego_pos

        h_des  = torch.linalg.norm(Dpos, ord = 2, axis = -1) # Compute 2 norm for vehicle positions 
        h_des  = torch.pow(h_des,2)
        h_des -= self.safe_radius**2

        # get ego dpos/dt
        ego_vel = states[:,0,1:,2, None]
        ego_yaw = states[:,0,1:,3, None]
        ego_dxdt = ego_vel * torch.cos(ego_yaw)
        ego_dydt = ego_vel * torch.sin(ego_yaw)
        ego_dposdt = torch.cat([ego_dxdt, ego_dydt], axis = -1)[...,None]
        ego_dposdt = ego_dposdt.repeat(1,1,1,A)
        ego_dposdt = ego_dposdt.permute(0,3,1,2)

        # get agent dpos/dt
        agent_vel = states[:,1:,1:,2,None]
        agent_yaw  = states[:,1:,1:,2,None]
        agent_dxdt = agent_vel * torch.cos(agent_yaw)
        agent_dydt = agent_vel * torch.sin(agent_yaw)
        agent_dposdt = torch.cat([agent_dxdt, agent_dydt], axis=-1)

        Ddposdt = agent_dposdt - ego_dposdt

        # Calculate extended barrier
        dot_pos_vel = torch.matmul(Dpos[:,:,:,None,:], Ddposdt[...,None])
        dot_pos_vel = dot_pos_vel.squeeze()     

        h_extended = 2 * dot_pos_vel + self.alpha_e * h_des 

        return h_extended

class RssCBF(CBF): 
    """
        Torch auto-grad compatible implementation of the RSS longitudinal CBF 
    """

    def __init__(self): 
        super(RssCBF, self).__init__() 
        self.reaction_time  = general_configs["REACTION_TIME"]
        self.max_accel      = general_configs["MAX_ACCEL"]
        self.max_brake      = general_configs["MAX_BRAKE"]
        self.min_brake      = general_configs["MIN_BRAKE"]

    def forward(self, data):
        """
            Args: 
                data    :   [x_front, x_rear, v_front, v_rear]
        """
        xf = data[:,0]
        xr = data[:,1]
        vf = data[:,2]
        vr = data[:,3]
        tau = self.reaction_time

        d_min  = general_configs["CAR_LENGTH"]
        d_min += vr*tau
        d_min += 1 / 2 * self.max_accel * tau**2
        d_min += 1 / 2 / self.min_brake * (vr + tau * self.max_accel)**2
        d_min += 1 / 2 / self.max_brake * vf**2
        d_min  = torch.nn.functional.relu(d_min)

        h = (xf - xr) - d_min

        return h 



def forward_project_states(ego_pos, agent_pos, backup_controller, N_tForward, dt):    
    ego_traj = []
    agent_traj = []
    next_ego_state = ego_pos
    next_agent_state = agent_pos
    for i in range(N_tForward): 
        # get controller input
        ego_input = backup_controller(next_ego_state)
        next_ego_state   = Unicycle.step(0, next_ego_state,   ego_input, dt, bound = False)
        agent_input = backup_controller(next_agent_state)
        next_agent_state = Unicycle.step(0, next_agent_state, agent_input, dt, bound = False) 
        ego_traj.append(next_ego_state[...,None])
        agent_traj.append(next_agent_state[...,None])
    ego_traj = torch.cat(ego_traj, axis=-1)
    agent_traj = torch.cat(agent_traj, axis=-1)
    ego_traj = ego_traj.permute(0,1,3,2)        # [B, A, T, D]
    agent_traj = agent_traj.permute(0,1,3,2)    # [B, A, T, D]

    return ego_traj, agent_traj

# Define backup controllers
def idle_controller(state): 
    B = state.shape[0]
    A = state.shape[1]
    return torch.zeros(B,A,2).to(state.device)

def braking_controller(state): 
    B = state.shape[0]
    A = state.shape[1]
    brake = -((torch.sigmoid(state[...,2]*4)-0.5)*2*9.0)[...,None] # 9.0 m/sec2, the sigmoid is chosen and shrunk so that braking is continuous
    turn = torch.zeros(B,A,1).to(state.device)
    input = torch.cat([brake, turn], axis = -1)    
    return input

class BackupBarrierCBF(CBF): 
    """
        Torch auto-grad compatible implementation of a norm ball cbf
    """

    def __init__(self, safe_radius = 0, T_horizon = 1, alpha=1, veh_veh = True, saturate_cbf=True, backup_controller_type="idle"): 
        super(BackupBarrierCBF, self).__init__() 
        self.safe_radius = safe_radius
        self.T_horizon = T_horizon
        self.alpha = alpha
        self.veh_veh = veh_veh
        self.saturate_cbf = saturate_cbf
        self.backup_controller_type = backup_controller_type

    def process_batch(self, batch): 
        T_idx = -1 # Most recent time step 
        A = batch["states"].shape[1] - 1 # Number of non-ego vehicles
        ego_pos = batch["states"][:,0,T_idx,:]
        ego_extent = batch["extent"][:,0,:]
        ego_pos = ego_pos.unsqueeze(1).repeat(1, A, 1)
        ego_extent = ego_extent.unsqueeze(1).repeat(1,A,1)
        agent_pos = batch["states"][:,1:,T_idx,:]
        agent_extent = batch["extent"][:,1:,:]
        dt = batch["dt"][0]
        data = torch.cat([ego_pos, agent_pos, ego_extent, agent_extent, dt * torch.ones_like(agent_extent[:,:,:1])], axis=-1)

        return data

    def forward(self, data):
        """
            Calculates Safety Values for the batch between each agent and the ego (agent0)
                The safety measure is h = min_{tau in [0, T_horizon]} distance(x_0(t+tau), x_j(t+tau))

            The assumed backup controller is constant velocity and driving forward, so the input is [accel=0,angle rate=0]
        """

        ego_state = data[...,0:4]
        agent_state = data[...,4:8]
        ego_extent = data[...,8:11]
        agent_extent = data[...,11:14]
        dt = data[...,14,None]


        N_tForward = int(self.T_horizon/dt[0,0])


        if self.backup_controller_type == "idle": 
            backup_controller = idle_controller
        elif self.backup_controller_type == "brake":
            backup_controller = braking_controller
        else: 
            print("please specify one of the possible backup controllers [idle, brake]")
            breakpoint()

        ego_traj, agent_traj = forward_project_states(ego_state, agent_state,backup_controller, N_tForward, dt)
        ego_extent   = ego_extent[..., None,:].repeat_interleave(N_tForward, axis=-2)
        agent_extent = agent_extent[..., None,:].repeat_interleave(N_tForward, axis=-2)
        if self.veh_veh: 
            dist = VEH_VEH_distance(ego_traj[...,0:3],agent_traj[...,0:3],ego_extent, agent_extent )
            dist = dist.amin(-1)
        else: # ball norm 
            veh_radius = 2
            dist = torch.linalg.norm(ego_traj[...,0:2] - agent_traj[...,0:2], axis=-1) - veh_radius #VEH_VEH_collision(ego_traj, agent_traj, ego_extent, agent_extent, return_dis = False)
            dist = dist.amin(-1) 
        h = dist 
        if self.saturate_cbf: 
            # Adding sigmoid to h to focus on locality
            h = (torch.sigmoid(h/5)-0.5)*2*5 # safety value can range (-5, 5) with 0 at 0, with real meaning within ~20 meters

        return h 


    def get_barrier_bits(self, h_vals, data, dynamics=unicycle_dynamics): 

        dhdx, = torch.autograd.grad(h_vals, inputs = data, grad_outputs = torch.ones_like(h_vals), create_graph=True)
        stateA_masked = data[0,:,0:4]
        stateB_masked = data[0,:,4:8]
        dhdxA_masked =  dhdx[0,:,0:4]          # dhdx wrt ego agent
        dhdxB_masked =  dhdx[0,:,4:8]          # dhdx wrt other agent
        h_masked = h_vals                # minus 1 time step because hs and inputs are only computed for everything up to current, excluding current

        (fA,fB,gA,gB) = dynamics(stateA_masked, stateB_masked)
        LfhA = torch.bmm(dhdxA_masked[:,None,:], fA).squeeze() 
        LfhB = torch.bmm(dhdxB_masked[:,None,:], fB).squeeze()
        LghA = torch.bmm(dhdxA_masked[:,None,:], gA)
        LghB = torch.bmm(dhdxB_masked[:,None,:], gB) 
        
        return [LfhA, LfhB, LghA, LghB] 
