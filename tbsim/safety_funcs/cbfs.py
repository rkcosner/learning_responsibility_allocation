import torch 

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

    def __init__(self, safe_radius = 5): 
        super(NormBallCBF, self).__init__() 
        self.safe_radius = safe_radius

    def forward(self, states):
        """
            Calculate safety values (computes h for unavailable states as though they were available, mask later)
            Args: 
                - states [B, A+1, T, D_states]
            Return: 
                - h_vals [B, A, T-1, 1]
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
