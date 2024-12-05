
from flax import linen as nn
from .base_model_registry import select_base_network


class BaseModel(nn.Module):
    network_config: dict
    SDE_Loss_Config: dict

    def setup(self):
        self.model = select_base_network(self.network_config, self.SDE_Loss_Config)
        
    @nn.compact
    def __call__(self, in_dict, train = False):
        print(in_dict)
        return self.model.__call__(in_dict, train = train)
        