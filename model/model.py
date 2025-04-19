import numpy as np
from torch.nn import Module

from . import *


def get_dct_matrix(N):
    """
    Generate the DCT and inverse DCT matrices of size NxN.
    """
    dct_matrix = np.eye(N)
    for k in range(N):
        for i in range(N):
            weight = np.sqrt(2 / N)
            if k == 0:
                weight = np.sqrt(1 / N)
            dct_matrix[k, i] = weight * np.cos(np.pi * (i + 0.5) * k / N)
    idct_matrix = np.linalg.inv(dct_matrix)
    return dct_matrix, idct_matrix


class DAFCN(Module):
    """
    Diffusion-Aware Frequency-based Convolutional Network for human motion prediction.
    """

    def __init__(self, in_features=48, kernel_size=10, d_model=512, num_stage=2, dct_n=10):
        super(DAFCN, self).__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        self.dct_n = dct_n
        assert kernel_size == 10  # current implementation only supports kernel_size = 10

        # Query branch
        self.query_net = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, bias=False),
            nn.ReLU()
        )

        # Key branch
        self.key_net = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, bias=False),
            nn.ReLU()
        )

        # Graph Convolutional Network
        self.gcn = GCN(input_feature=dct_n, hidden_feature=d_model, p_dropout=0.3,
                       num_stage=num_stage, node_n=in_features)

        # MLP for final output fusion
        self.mlp = nn.Sequential(
            nn.Linear(in_features=40, out_features=256, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=40, bias=False)
        )

        # Frequency-based Convolution
        self.ffc = RealFFC(dim=3)

    def forward(self, input_seq, output_n=25, input_n=50, itera=1):
        """
        Forward pass of the DAFCN model.

        Args:
            input_seq (Tensor): [batch_size, seq_len, feature_dim]
            output_n (int): number of output frames
            input_n (int): number of input frames
            itera (int): number of refinement iterations

        Returns:
            Tensor: predicted output sequences
        """
        dct_n = self.dct_n
        input_seq = input_seq[:, :input_n]  # [B, input_n, F]
        seq_clone = input_seq.clone()
        batch_size = seq_clone.size(0)

        # Prepare key and query sequences
        key_seq = seq_clone.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        query_seq = seq_clone.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        # DCT transform matrix
        dct_mat, idct_mat = get_dct_matrix(self.kernel_size + output_n)
        dct_mat = torch.from_numpy(dct_mat).float().to(input_seq.device)
        idct_mat = torch.from_numpy(idct_mat).float().to(input_seq.device)

        # Prepare DCT value blocks
        num_val_blocks = input_n - self.kernel_size - output_n + 1
        block_len = self.kernel_size + output_n
        idx = np.expand_dims(np.arange(block_len), axis=0) + np.expand_dims(np.arange(num_val_blocks), axis=1)
        dct_value_blocks = seq_clone[:, idx].clone().reshape(batch_size * num_val_blocks, block_len, -1)
        dct_value_blocks = torch.matmul(dct_mat[:dct_n].unsqueeze(0), dct_value_blocks)
        dct_value_blocks = dct_value_blocks.reshape(batch_size, num_val_blocks, dct_n, -1).transpose(2, 3).reshape(
            batch_size, num_val_blocks, -1)

        # Positional indices for input GCN
        gcn_input_idx = list(range(-self.kernel_size, 0)) + [-1] * output_n
        outputs = []

        # Precompute keys
        key_feat = self.key_net(key_seq / 1000.0)

        for _ in range(itera):
            # Prepare FFC input: concatenate with padding
            gcn_input = seq_clone[:, -input_n:].clone()
            ffc_padding = gcn_input[:, [-1] * 10]
            ffc_input = torch.cat([gcn_input, ffc_padding], dim=1).transpose(1, 2)
            ffc_input = ffc_input.view(batch_size, 3, -1, input_n + 10)
            ffc_out = self.ffc(ffc_input).reshape(batch_size, -1, input_n + 10)[:, :, :dct_n].transpose(1, 2)

            # Query feature
            query_feat = self.query_net(query_seq / 1000.0)
            attention_score = torch.matmul(query_feat.transpose(1, 2), key_feat) + 1e-15
            attention_weights = attention_score / attention_score.sum(dim=2, keepdim=True)
            attended_dct = torch.matmul(attention_weights, dct_value_blocks)[:, 0].reshape(batch_size, -1, dct_n)

            # DCT transform for GCN
            gcn_in = seq_clone[:, gcn_input_idx]
            dct_input = torch.matmul(dct_mat[:dct_n].unsqueeze(0), gcn_in).transpose(1, 2)
            gcn_out = self.gcn(dct_input)

            # Fuse attended and GCN features
            combined_dct = torch.cat([gcn_out, attended_dct], dim=-1)
            gcn_recon = torch.matmul(idct_mat[:, :dct_n].unsqueeze(0), combined_dct[:, :, :dct_n].transpose(1, 2))

            # Final output prediction
            fused_features = torch.cat([gcn_recon, ffc_out], dim=1).transpose(1, 2)
            output_frame = self.mlp(fused_features).transpose(1, 2)[:, :dct_n]
            outputs.append(output_frame.unsqueeze(2))

            # Iterative update
            if itera > 1:
                seq_clone = torch.cat([seq_clone, output_frame[:, -output_n:]], dim=1)

                new_block_start = 1 - 2 * self.kernel_size - output_n
                block_idx = np.expand_dims(np.arange(block_len), axis=0) + np.expand_dims(
                    np.arange(new_block_start, -self.kernel_size - output_n + 1), axis=1)

                new_keys = seq_clone[:, block_idx[0, :-1]].transpose(1, 2)
                key_feat = torch.cat([key_feat, self.key_net(new_keys / 1000.0)], dim=2)

                new_vals = seq_clone[:, block_idx].clone().reshape(batch_size * self.kernel_size, block_len, -1)
                new_vals = torch.matmul(dct_mat[:dct_n].unsqueeze(0), new_vals).reshape(batch_size, self.kernel_size,
                                                                                        dct_n, -1).transpose(2,
                                                                                                             3).reshape(
                    batch_size, self.kernel_size, -1)
                dct_value_blocks = torch.cat([dct_value_blocks, new_vals], dim=1)

                query_seq = seq_clone[:, -self.kernel_size:].transpose(1, 2)

        return torch.cat(outputs, dim=2)
