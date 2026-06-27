# modules/models/loss.py
import torch
import torch.nn as nn

class CoxProportionalHazardsLoss(nn.Module):
    """
    Implements negative log partial likelihood for survival data.
    Handles right-censored patients correctly.
    """
    def __init__(self):
        super().__init__()

    def forward(self, log_hazards, durations, events):
        """
        log_hazards: Model output predictions [Batch, 1]
        durations: Time-to-event or time-to-censoring values [Batch]
        events: Binary tracking indicators (1 = observed death, 0 = censored) [Batch]
        """
        # Sort data by survival durations in descending order
        # This simplifies computing the 'Risk Set' for each patient
        R_sort_idx = torch.argsort(durations, descending=True)
        log_hazards = log_hazards[R_sort_idx].squeeze()
        events = events[R_sort_idx].squeeze()

        # Compute log of the sum of exponentiated hazards for patients still at risk
        # Using cumulative sum tracking down the sorted survival times
        log_sum_exp_hazards = torch.logcumreplace(torch.exp(log_hazards))
        
        # Calculate the partial likelihood strictly over observed events
        observed_losses = events * (log_hazards - log_sum_exp_hazards)
        
        # Compute final mean negative log likelihood
        loss = -torch.sum(observed_losses) / (torch.sum(events) + 1e-8)
        return loss

# Helper extension to safely accumulate logs for stable gradients
def torch_logcumreplace(x):
    """Maintains running log-sum-exp values along vector indices safely."""
    return torch.log(torch.cumsum(x, dim=0))