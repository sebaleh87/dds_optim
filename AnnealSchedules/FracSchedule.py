import jax.numpy as jnp
from .BaseSchedule import BaseScheduleClass

class FracScheduleClass(BaseScheduleClass):
    def __init__(self, AnnealConfig):
        self.start_temperature = AnnealConfig["T_start"]
        self.final_temperature = AnnealConfig["T_end"]
        self.steps = AnnealConfig["N_anneal"]
        self.N_warmup_steps = AnnealConfig["N_warmup"]
        self.lam = AnnealConfig["lam"]
        self.current_step = 0
        super().__init__(AnnealConfig)

    def update_temp(self):
        self.current_step += 1
        if(self.current_step < self.N_warmup_steps):
            return self.start_temperature
        elif self.current_step >= self.steps:
            return self.final_temperature
        else:
            lam = self.lam*300
            step_diff = self.current_step - self.N_warmup_steps
            value = self.final_temperature *1/(1- 0.998**(lam*(step_diff )/ self.steps + 1 ))
            return value

    def reset(self):
        self.current_step = 0