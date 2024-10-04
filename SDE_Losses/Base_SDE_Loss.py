from .SDE_Types import get_SDE_Type_Class

class Base_SDE_Loss_Class:
    def __init__(self, SDE_config, Energy_Class, model):
        SDE_Type_Config = SDE_config["SDE_Type_Config"]
        self.SDE_type = get_SDE_Type_Class(SDE_Type_Config)
        self.EnergyClass = Energy_Class
        self.model = model

    def simulate_reverse_sde_scan(self, params, key, n_states = 100, x_dim = 2, n_integration_steps = 1000):
        return self.SDE_type.simulate_reverse_sde_scan(self.model, params, key, n_states = n_states, x_dim = x_dim, n_integration_steps = n_integration_steps)

    def get_loss(self):
        """
        Calculate the loss between predictions and targets.

        :param predictions: The predicted values.
        :param targets: The ground truth values.
        :return: The calculated loss.
        """
        raise NotImplementedError("get_loss method not implemented")