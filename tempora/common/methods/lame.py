import torch
import torch.nn.functional as F

from .base import Method


# Source: https://github.com/fiveai/LAME
# Paper : https://arxiv.org/abs/2201.05718
# Note  : This version differs from the original in three ways: (1) it implicitly uses the kNN affinity matrix, (2) it
#         drops the 'bound_lambda' smoothness factor, and (3) improves code quality (in my opinion). LAME keeps the 
#         model frozen and treats adaptation as a manifold smoothing problem on the output probabilities. The core idea
#         is that samples close in the feature space should predict similarly.
class LAME(Method):
    def __init__(self, model, neighbors=5, max_steps=100):
        super().__init__()
        self.frozen = False

        self.model = self._configure_model(model)
        self.neighbors = neighbors  # the 'k' in kNN
        self.max_steps = max_steps

    def forward(self, x):
        with torch.no_grad():
            outputs, features = self.model(x, return_feature_and_logits=True)

            if self.frozen:
                return outputs.softmax(1)

            unary = -outputs.log_softmax(1)
            affinity_matrix = self._compute_affinity_matrix(F.normalize(features, p=2, dim=-1))

            # The optimisation balances a model's belief with the spatial consistency of features in a batch
            outputs = self._laplacian_optimization(unary, affinity_matrix)

        return outputs

    def reset(self):
        pass

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def _configure_model(self, model):
        model.eval()
        model.requires_grad_(False)

        return model

    # Iteratively refine class probabilities to minimise deviation from the model's initial probabilities (unary term) 
    # and maximise agreement with neighbours (pairwise term) until the energy stabilises or max steps is reached.
    # Note: outputs.log_softmax(1).softmax(-1) = outputs.softmax(1)
    def _laplacian_optimization(self, unary, affinity_matrix):
        class_probs = (-unary).softmax(-1)
        prev_energy = float("inf")
        
        step = 0
        converged = False
        while step < self.max_steps and not converged: 
            pairwise = affinity_matrix @ class_probs
            class_probs = (-unary + pairwise).softmax(-1)  # Combine model's original belief with neighbour consensus

            unary_term = (unary * class_probs).sum()  # Deviation from model belief
            entropy_term = torch.xlogy(class_probs, class_probs).sum()  # Penalty for overconfidence
            pairwise_term = (pairwise * class_probs).sum()  # Reward for consistency among neighbors
            curr_energy = (unary_term + entropy_term - pairwise_term).item()

            converged = step > 1 and abs(curr_energy - prev_energy) <= 1e-8 * abs(prev_energy)
            
            prev_energy = curr_energy
            step += 1

        return class_probs
    
    def _compute_affinity_matrix(self, X):
        N = X.size(0)

        # Calculate pairwise distance between N samples and use it to identify k closest neighbours for each sample
        distances = torch.cdist(X, X, p=2)
        knn_index = distances.topk(min(self.neighbors + 1, N), -1, largest=False).indices[:, 1:]

        # W[i, j] = 1 means sample j is one of sample i's k-nearest neighbours
        W = torch.zeros(N, N, device=X.device)
        W.scatter_(dim=-1, index=knn_index, value=1.0)

        return W