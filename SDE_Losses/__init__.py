from .Base_SDE_Loss import Base_SDE_Loss_Class
from .Reverse_KL_Loss import Reverse_KL_Loss_Class
from .Reverse_KL_Loss_stop_grad import Reverse_KL_Loss_SG_Class
from .LogVariance_Loss import LogVariance_Loss_Class
from .LogVariance_Loss_with_grad import LogVariance_Loss_with_grad_Class
from .LogVarianceLoss_MC import LogVarianceLoss_MC_Class
from .Discrete_Time_rKL_Loss_reparam import Discrete_Time_rKL_Loss_Class_reparam
from .Discrete_Time_rKL_Loss_log_deriv import Discrete_Time_rKL_Loss_Class_log_deriv
from .LogVariance_Loss_weighted import LogVariance_Loss_weighted_Class
from .Bridge_LogVarLoss import Bridge_LogVarLoss_Class
from .Bridge_rKL import Bridge_rKL_Loss_Class
from .Bridge_rKL_logderivative import Bridge_rKL_logderiv_Loss_Class
from .Bridge_rKL_logderivative_DiffUCO import Bridge_rKL_logderiv_DiffUCO_Loss_Class
from .Reverse_KL_Loss_log_deriv import Reverse_KL_Loss_log_deriv_Class
from .Bridge_fKL_subtraj import Bridge_fKL_subtraj_Loss_Class
from .Bridge_fKL_logderivative import Bridge_fKL_logderiv_Loss_Class
from .Bridge_rKL_subtraj import Bridge_rKL_subtraj_Loss_Class
from .Bridge_rKL_fKL_logderivative import Bridge_rKL_fKL_logderiv_Loss_Class


SDE_Loss_registry = {"Reverse_KL_Loss": Reverse_KL_Loss_Class, "Reverse_KL_Loss_logderiv": Reverse_KL_Loss_log_deriv_Class, 
                     "Reverse_KL_Loss_stop_grad": Reverse_KL_Loss_SG_Class, "LogVariance_Loss": LogVariance_Loss_Class, "LogVariance_Loss_with_grad": LogVariance_Loss_with_grad_Class,
                      "LogVariance_Loss_MC": LogVarianceLoss_MC_Class, "Discrete_Time_rKL_Loss_reparam": Discrete_Time_rKL_Loss_Class_reparam,
                        "Bridge_rKL": Bridge_rKL_Loss_Class, "Bridge_LogVarLoss": Bridge_LogVarLoss_Class, "Bridge_rKL_logderiv": Bridge_rKL_logderiv_Loss_Class,
                       "Discrete_Time_rKL_Loss_log_deriv": Discrete_Time_rKL_Loss_Class_log_deriv, "LogVariance_Loss_weighted": LogVariance_Loss_weighted_Class,
                       "Bridge_rKL_logderiv_DiffUCO": Bridge_rKL_logderiv_DiffUCO_Loss_Class, "Bridge_fKL_subtraj": Bridge_fKL_subtraj_Loss_Class, 
                       "Bridge_rKL_subtraj": Bridge_rKL_subtraj_Loss_Class, "Bridge_fKL_logderiv": Bridge_fKL_logderiv_Loss_Class,
                       "Bridge_rKL_fKL_logderivative": Bridge_rKL_fKL_logderiv_Loss_Class}

def get_SDE_Loss_class(SDE_Loss_Config, Optimizer_Config, Energy_Class, Network_Config, model):
    return SDE_Loss_registry[SDE_Loss_Config["name"]](SDE_Loss_Config, Optimizer_Config, Energy_Class, Network_Config, model)