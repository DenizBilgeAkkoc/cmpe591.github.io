import torch
import torch.nn as nn

class ColorWeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0, red_weight=2.0, green_weight=1.0, blue_weight=1.0):
        super(ColorWeightedHuberLoss, self).__init__()
        self.delta = delta
        self.weights = torch.tensor([red_weight, green_weight, blue_weight])
        
    def forward(self, pred, target):
        # Ensure weights are on the same device as the input
        self.weights = self.weights.to(pred.device)
        
        # Calculate absolute difference
        abs_diff = torch.abs(pred - target)
        
        # Apply channel-specific weights
        if len(pred.shape) == 4:  # batch x channel x height x width
            weighted_diff = abs_diff * self.weights.view(1, -1, 1, 1)
        else:  # just channel x height x width
            weighted_diff = abs_diff * self.weights.view(-1, 1, 1)
            
        # Huber loss calculation with weighted differences
        quadratic_mask = (weighted_diff <= self.delta)
        linear_mask = (weighted_diff > self.delta)
        
        loss = torch.zeros_like(weighted_diff)
        loss[quadratic_mask] = 0.5 * weighted_diff[quadratic_mask]**2
        loss[linear_mask] = self.delta * weighted_diff[linear_mask] - 0.5 * self.delta**2
        
        return loss.mean()

# Example usage:
"""
# Initialize loss function with custom weights
criterion = ColorWeightedHuberLoss(
    delta=1.0,
    red_weight=2.0,    # Red errors count twice as much
    green_weight=1.0,  # Normal weight for green
    blue_weight=1.0    # Normal weight for blue
)

# Use in training loop
loss = criterion(predictions, targets)
loss.backward()
"""