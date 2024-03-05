import torch
import torch.nn as nn

from muvo.models.common import PositionEmbeddingSine, PositionEmbeddingSine3D


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc_z = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.fc_r = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.fc_q = nn.Linear(hidden_dim + input_dim, hidden_dim)
        # self.conv_z = nn.Conv1d(hidden_dim + input_dim, hidden_dim, 1)
        # self.conv_r = nn.Conv1d(hidden_dim + input_dim, hidden_dim, 1)
        # self.conv_q = nn.Conv1d(hidden_dim + input_dim, hidden_dim, 1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=-1)

        z = torch.sigmoid(self.fc_z(hx))
        r = torch.sigmoid(self.fc_r(hx))
        q = torch.tanh(self.fc_q(torch.cat([r * h, x], dim=-1)))

        h = (1 - z) * h + z * q
        return h


class ConvGRUCellGlo(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # self.fc_z = nn.Linear(hidden_dim + input_dim, hidden_dim)
        # self.fc_r = nn.Linear(hidden_dim + input_dim, hidden_dim)
        # self.fc_q = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.conv_z = nn.Conv1d(hidden_dim + input_dim, hidden_dim, 1)
        self.conv_r = nn.Conv1d(hidden_dim + input_dim, hidden_dim, 1)
        self.conv_q = nn.Conv1d(hidden_dim + input_dim, hidden_dim, 1)

        self.w = nn.Conv1d(hidden_dim + input_dim, hidden_dim + input_dim, 1)

        self.conv_z_glo = nn.Conv1d(hidden_dim + input_dim, hidden_dim, 1)
        self.conv_r_glo = nn.Conv1d(hidden_dim + input_dim, hidden_dim, 1)
        self.conv_q_glo = nn.Conv1d(hidden_dim + input_dim, hidden_dim, 1)

    def forward(self, h, x):
        h = h.permute(1, 2, 0)  # N, B, C -> B, C, N
        x = x.permute(1, 2, 0)

        hx = torch.cat([h, x], dim=1)

        glo = torch.sigmoid(self.w(hx)) * hx
        glo = glo.mean(dim=-1, keepdim=True)

        z = torch.sigmoid(self.conv_z(hx) + self.conv_z_glo(glo))
        r = torch.sigmoid(self.conv_r(hx) + self.conv_r_glo(glo))
        q = torch.tanh(self.conv_q(torch.cat([r * h, x], dim=1)) + self.conv_q_glo(glo))

        h = (1 - z) * h + z * q
        return h.permute(2, 0, 1)  # B, C, N -> N, B, C


class RepresentationModelTD(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.min_std = 0.1

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=in_channels, nhead=8
        )
        self.module = nn.TransformerDecoder(
            decoder_layer=self.transformer_decoder_layer, num_layers=6
        )
        self.fc = nn.Linear(in_channels, 2 * latent_dim)
        # self.conv = nn.Conv1d(in_channels, 2 * latent_dim, 1)

        self.type_embeddings = nn.Parameter(torch.zeros(1, 1, in_channels, 4))  # N, B, C, n_types
        self.query_embed_image = nn.Parameter(torch.zeros(1, in_channels, 10, 26))
        self.query_embed_lidar = nn.Parameter(torch.zeros(1, in_channels, 4, 64))
        self.query_embed_voxel = nn.Parameter(torch.zeros(1, in_channels, 12, 12, 4))  # B, C, X, Y, Z
        self.query_embed_policy = nn.Parameter(torch.zeros(1, 1, in_channels))

        self.reset_parameters()

        self.pos_embedding = PositionEmbeddingSine(
            num_pos_feats=in_channels // 2,
            normalize=True,
        )
        self.pos_embedding_3d = PositionEmbeddingSine3D(
            num_pos_feats=in_channels // 3,
            normalize=True,
        )

    def reset_parameters(self):
        nn.init.uniform_(self.type_embeddings)
        nn.init.uniform_(self.query_embed_image)
        nn.init.uniform_(self.query_embed_lidar)
        nn.init.uniform_(self.query_embed_voxel)
        nn.init.uniform_(self.query_embed_policy)

    def forward(self, x):
        def sigmoid2(tensor: torch.Tensor, min_value: float) -> torch.Tensor:
            return 2 * torch.sigmoid(tensor / 2) + min_value

        bs = x.shape[1]

        # B, C, H, W -> N, B, C
        query_embed_image = self.query_embed_image + self.pos_embedding(self.query_embed_image)
        query_embed_image = query_embed_image.flatten(2).permute(2, 0, 1) + self.type_embeddings[:, :, :, 0]

        query_embed_lidar = self.query_embed_lidar + self.pos_embedding(self.query_embed_lidar)
        query_embed_lidar = query_embed_lidar.flatten(2).permute(2, 0, 1) + self.type_embeddings[:, :, :, 1]

        query_embed_voxel = self.query_embed_voxel + self.pos_embedding_3d(self.query_embed_voxel)
        query_embed_voxel = query_embed_voxel.flatten(2).permute(2, 0, 1) + self.type_embeddings[:, :, :, 2]

        query_embed_policy = self.query_embed_policy + self.type_embeddings[:, :, :, 3]

        query_embed = torch.cat([query_embed_image, query_embed_lidar, query_embed_voxel, query_embed_policy], dim=0)

        mu_log_sigma = self.fc(self.module(query_embed.repeat(1, bs, 1), x))
        mu, log_sigma = torch.split(mu_log_sigma, self.latent_dim, dim=-1)

        sigma = sigmoid2(log_sigma, self.min_std)
        return mu, sigma


class RSSMTD(nn.Module):
    def __init__(self, embedding_dim, action_dim, hidden_state_dim, state_dim, receptive_field,
                 use_dropout=False,
                 dropout_probability=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_state_dim = hidden_state_dim
        self.receptive_field = receptive_field
        # Sometimes unroll the prior instead of always updating with the posterior
        # so that the model learns to unroll for more than 1 step in the future when imagining
        self.use_dropout = use_dropout
        self.dropout_probability = dropout_probability

        self.n_out_tokens = 26 * 10 + 64 * 4 + 12 * 12 * 4 + 1

        self.type_embeddings = nn.Parameter(torch.zeros(1, 1, hidden_state_dim, 3))  # N, B, C, n_types
        nn.init.uniform_(self.type_embeddings)

        # Map input of the gru to a space with easier temporal dynamics
        self.pre_gru_net = nn.Sequential(
            nn.Linear(state_dim, hidden_state_dim),
            nn.LeakyReLU(True),
        )

        self.recurrent_model = ConvGRUCellGlo(
            input_dim=hidden_state_dim,
            hidden_dim=hidden_state_dim,
        )

        # Map action to a higher dimensional input
        self.posterior_action_module = nn.Sequential(
            nn.Linear(action_dim, hidden_state_dim),
            nn.LeakyReLU(True),
        )

        self.posterior = RepresentationModelTD(
            in_channels=hidden_state_dim,
            latent_dim=state_dim,
        )

        # Map action to a higher dimensional input
        self.prior_action_module = nn.Sequential(
            nn.Linear(action_dim, hidden_state_dim),
            nn.LeakyReLU(True),
        )
        self.prior = RepresentationModelTD(in_channels=hidden_state_dim, latent_dim=state_dim)
        self.active_inference = False
        if self.active_inference:
            print('ACTIVE INFERENCE!!')

    def forward(self, input_embedding, action, use_sample=True, policy=None):
        """
        Inputs
        ------
            input_embedding: torch.Tensor size (B, S, C)
            action: torch.Tensor size (B, S, 2)
            use_sample: bool
                whether to use sample from the distributions, or taking the mean

        Returns
        -------
            output: dict
                prior: dict
                    hidden_state: torch.Tensor (B, S, C_h)
                    sample: torch.Tensor (B, S, C_s)
                    mu: torch.Tensor (B, S, C_s)
                    sigma: torch.Tensor (B, S, C_s)
                posterior: dict
                    hidden_state: torch.Tensor (B, S, C_h)
                    sample: torch.Tensor (B, S, C_s)
                    mu: torch.Tensor (B, S, C_s)
                    sigma: torch.Tensor (B, S, C_s)
        """
        output = {
            'prior': [],
            'posterior': [],
        }

        #  Initialisation
        batch_size, sequence_length, _, _ = input_embedding.shape
        h_t = input_embedding.new_zeros((self.n_out_tokens, batch_size, self.hidden_state_dim))
        sample_t = input_embedding.new_zeros((self.n_out_tokens, batch_size, self.state_dim))
        for t in range(sequence_length):
            if t == 0:
                action_t = torch.zeros_like(action[:, 0])
            else:
                action_t = action[:, t - 1]
            output_t = self.observe_step(
                h_t, sample_t, action_t, input_embedding[:, t], use_sample=use_sample, policy=policy,
            )
            # During training sample from the posterior, except when using dropout
            # always use posterior for the first frame
            use_prior = self.training and self.use_dropout and torch.rand(1).item() < self.dropout_probability and t > 0

            if use_prior:
                sample_t = output_t['prior']['sample']
            else:
                sample_t = output_t['posterior']['sample']
            h_t = output_t['prior']['hidden_state']

            for key, value in output_t.items():
                output[key].append(value)

        output = self.stack_list_of_dict_tensor(output, dim=1)  # N, B, C -> B, C, N -> B, S, C, N
        return output

    def observe_step(self, h_t, sample_t, action_t, embedding_t, use_sample=True, policy=None):
        embedding_t = embedding_t.permute(2, 0, 1)
        imagine_output = self.imagine_step(h_t, sample_t, action_t, use_sample, policy=policy)

        latent_action_t = self.posterior_action_module(action_t)[None]
        embedding_t_tokens = embedding_t + self.type_embeddings[:, :, :, 0]
        latent_action_t_tokens = latent_action_t + self.type_embeddings[:, :, :, 2]
        posterior_mu_t, posterior_sigma_t = self.posterior(
            torch.cat([imagine_output['hidden_state'], embedding_t_tokens, latent_action_t_tokens], dim=0)
        )

        sample_t = self.sample_from_distribution(posterior_mu_t, posterior_sigma_t, use_sample=use_sample)

        posterior_output = {
            'hidden_state': imagine_output['hidden_state'],
            'sample': sample_t,
            'mu': posterior_mu_t,
            'sigma': posterior_sigma_t,
        }

        output = {
            'prior': imagine_output,
            'posterior': posterior_output,
        }

        return output

    def imagine_step(self, h_t, sample_t, action_t, use_sample=True, policy=None):
        if self.active_inference:
            # Predict action with policy
            action_t = policy(sample_t[-1])

        latent_action_t = self.prior_action_module(action_t)[None]

        input_t = self.pre_gru_net(sample_t)
        h_t = self.recurrent_model(h_t, input_t)
        h_t_tokens = h_t + self.type_embeddings[:, :, :, 1]
        latent_action_t_tokens = latent_action_t + self.type_embeddings[:, :, :, 2]
        prior_mu_t, prior_sigma_t = self.prior(torch.cat([h_t_tokens, latent_action_t_tokens], dim=0))
        sample_t = self.sample_from_distribution(prior_mu_t, prior_sigma_t, use_sample=use_sample)
        imagine_output = {
            'hidden_state': h_t,
            'sample': sample_t,
            'mu': prior_mu_t,
            'sigma': prior_sigma_t,
        }
        return imagine_output

    @staticmethod
    def sample_from_distribution(mu, sigma, use_sample):
        sample = mu
        if use_sample:
            noise = torch.randn_like(sample)
            sample = sample + sigma * noise
        return sample

    @staticmethod
    def stack_list_of_dict_tensor(output, dim=1):
        new_output = {}
        for outter_key, outter_value in output.items():
            if len(outter_value) > 0:
                new_output[outter_key] = dict()
                for inner_key in outter_value[0].keys():
                    new_output[outter_key][inner_key] = \
                        torch.stack([x[inner_key].permute(1, 2, 0) for x in outter_value], dim=dim)
        return new_output
