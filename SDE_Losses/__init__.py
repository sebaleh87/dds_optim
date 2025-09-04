from .Base_SDE_Loss import Base_SDE_Loss_Class
from .Reverse_KL_Loss import Reverse_KL_Loss_Class
from .Reverse_KL_Loss_stop_grad import Reverse_KL_Loss_SG_Class
from .LogVariance_Loss import LogVariance_Loss_Class
from .LogVarianceLoss_MC import LogVarianceLoss_MC_Class
from .Bridge_LogVarLoss import Bridge_LogVarLoss_Class
from .Bridge_rKL import Bridge_rKL_Loss_Class
from .Bridge_rKL_logderivative import Bridge_rKL_logderiv_Loss_Class
from .Reverse_KL_Loss_log_deriv import Reverse_KL_Loss_log_deriv_Class
from .PPO_loss import PPO_Loss_Class


SDE_Loss_registry = {"Reverse_KL_Loss": Reverse_KL_Loss_Class, "Reverse_KL_Loss_logderiv": Reverse_KL_Loss_log_deriv_Class, 
                     "Reverse_KL_Loss_stop_grad": Reverse_KL_Loss_SG_Class, "LogVariance_Loss": LogVariance_Loss_Class,
                      "LogVariance_Loss_MC": LogVarianceLoss_MC_Class,  "Bridge_rKL": Bridge_rKL_Loss_Class, "Bridge_LogVarLoss": Bridge_LogVarLoss_Class,
                        "Bridge_rKL_logderiv": Bridge_rKL_logderiv_Loss_Class, "PPO_Loss": PPO_Loss_Class
                       }

def get_SDE_Loss_class(SDE_Loss_Config, Optimizer_Config, Energy_Class, Network_Config, model):
    return SDE_Loss_registry[SDE_Loss_Config["name"]](SDE_Loss_Config, Optimizer_Config, Energy_Class, Network_Config, model)