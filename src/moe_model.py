import torch
import torch.nn as nn
import torch.nn.functional as F


class Classic_MOE(nn.Module):
    def __init__(self, experts, input_dim=48000):
        super(Classic_MOE, self).__init__()

        self.experts = nn.ModuleList(experts)
        self.gating_network = nn.Linear(input_dim, 4)

    def forward(self, x):
        expert_outputs = [expert(x)[0] for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=-1)

        gating_weights = self.gating_network(x)
        gating_weights = F.softmax(gating_weights, dim=-1)
        weighted_output = torch.einsum('bn,bcn->bc', gating_weights, expert_outputs)

        return weighted_output, gating_weights


class Enhanced_MOE(nn.Module):
    def __init__(self, experts, freezing=False):
        super(Enhanced_MOE, self).__init__()

        self.experts = nn.ModuleList(experts)
        num_experts = len(experts)

        self.lrelu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(32)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(64, 32)

        self.pooling = nn.Parameter(torch.ones(32))

        self.gating_network = nn.Sequential(
            nn.Linear(32 * (num_experts + 1), 64),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, num_experts),
        )

        if freezing:
            self.freezing()

    def freezing(self):
        for expert in self.experts:
            expert.freeze_residual_part()
            expert.freeze_processing_part()
            # expert.unfreeze_processing_part()

    def forward(self, x):

        outputs = [expert(x)[0] for expert in self.experts]
        embeddings = [expert(x)[1] for expert in self.experts]

        emb_1 = self.lrelu(self.bn(self.fc1(embeddings[0])))
        emb_2 = self.lrelu(self.bn(self.fc2(embeddings[1])))
        emb_3 = self.lrelu(self.bn(self.fc3(embeddings[2])))
        emb_4 = self.lrelu(self.bn(self.fc4(embeddings[3])))

        combined = emb_1 * emb_2 * emb_3 * emb_4
        weighted_combined = combined * self.pooling.unsqueeze(0)

        concatenated_embeddings = torch.cat((emb_1, emb_2, emb_3, emb_4, weighted_combined), dim=1)
        gating_weights = self.gating_network(concatenated_embeddings)
        gating_weights = F.softmax(gating_weights, dim=-1)

        weighted_logits = torch.stack(outputs, dim=-1)
        weighted_logits = torch.einsum('bn,bcn->bc', gating_weights, weighted_logits)

        return weighted_logits, gating_weights
