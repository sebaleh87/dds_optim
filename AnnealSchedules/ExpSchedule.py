import jax.numpy as jnp
from .BaseSchedule import BaseScheduleClass

class ExpScheduleClass(BaseScheduleClass):
    def __init__(self, AnnealConfig):
        self.start_temperature = AnnealConfig["T_start"]
        self.final_temperature = AnnealConfig["T_end"]
        self.steps = AnnealConfig["N_anneal"]
        self.N_warmup_steps = AnnealConfig["N_warmup"]
        self.lam = AnnealConfig["lam"]
        self.current_step = -1
        self.eps = 0.1
        super().__init__(AnnealConfig)

    def update_temp(self):
        self.current_step += 1
        if self.current_step <= self.N_warmup_steps:
            return self.start_temperature
        lam = self.compute_k(self.eps, self.start_temperature)
        value = jnp.exp(-lam * (self.current_step - self.N_warmup_steps) / self.steps) * (self.start_temperature-self.final_temperature) + self.final_temperature
        self.current_step += 1
        return value

    def compute_k(self, eps, T_start):
        k = -jnp.log((eps) / T_start)
        return k

    def reset(self):
        self.current_step = 0