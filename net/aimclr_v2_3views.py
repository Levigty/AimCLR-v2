import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class


class AimCLR_v2_3views(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """
    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

        if not self.pretrain:
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_q_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=num_class,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_q_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=num_class,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature

            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_q_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=feature_dim,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_k_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=feature_dim,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_q_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_k_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_k.fc)
                self.encoder_q_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                         nn.ReLU(),
                                                         self.encoder_q.fc)
                self.encoder_k_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                         nn.ReLU(),
                                                         self.encoder_k.fc)
                self.encoder_q_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_q.fc)
                self.encoder_k_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_k.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
            for param_q, param_k in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.encoder_q_bone.parameters(), self.encoder_k_bone.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            # create the queue
            self.register_buffer("queue", torch.randn(feature_dim, self.K))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_motion", torch.randn(feature_dim, self.K))
            self.queue_motion = F.normalize(self.queue_motion, dim=0)
            self.register_buffer("queue_ptr_motion", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_bone", torch.randn(feature_dim, self.K))
            self.queue_bone = F.normalize(self.queue_bone, dim=0)
            self.register_buffer("queue_ptr_bone", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_motion(self):
        for param_q, param_k in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_bone(self):
        for param_q, param_k in zip(self.encoder_q_bone.parameters(), self.encoder_k_bone.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def _dequeue_and_enqueue_motion(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_motion)
        gpu_index = keys.device.index
        self.queue_motion[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def _dequeue_and_enqueue_bone(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_bone)
        gpu_index = keys.device.index
        self.queue_bone[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0  # for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K
        self.queue_ptr_motion[0] = (self.queue_ptr_motion[0] + batch_size) % self.K
        self.queue_ptr_bone[0] = (self.queue_ptr_bone[0] + batch_size) % self.K

    def forward(self, im_q_extreme, im_q=None, im_k=None, stream='all', cross=False, mine=False, topk=1, vote=2):
        '''
        First-Stage
        :param im_q_extreme: a batch of extreme query images
        :param im_q: a batch of query images
        :param im_k: a batch of key images
        :param stream: 'joint' or 'bone' or 'motion' or 'all'-(3-stream)
        :param cross: multi-stream aggregation and interaction (True or False)
        :param mine: single-stream nearest neighbors mining (True or False)
        :param topk: top-k similar in memory bank
        :param vote: vote value of three streams
        :return:
        '''
        # single-stream nearest neighbors mining
        if mine:
            return self.mining(im_q_extreme, im_q, im_k, topk)

        # multi-stream aggregation and interaction
        if cross:
            return self.cross_training(im_q_extreme, im_q, im_k, topk, vote)

        # im_q_motion
        im_q_motion = torch.zeros_like(im_q)
        im_q_motion[:, :, :-1, :, :] = im_q[:, :, 1:, :, :] - im_q[:, :, :-1, :, :]

        # im_q_bone
        im_q_bone = torch.zeros_like(im_q)
        for v1, v2 in self.Bone:
            im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]

        if not self.pretrain:
            if stream == 'joint':
                return self.encoder_q(im_q)
            elif stream == 'motion':
                return self.encoder_q_motion(im_q_motion)
            elif stream == 'bone':
                return self.encoder_q_bone(im_q_bone)
            elif stream == 'all':
                return (self.encoder_q(im_q) + self.encoder_q_motion(im_q_motion) + self.encoder_q_bone(im_q_bone)) / 3.
            else:
                raise ValueError

        # im_k
        im_k_motion = torch.zeros_like(im_k)
        im_k_motion[:, :, :-1, :, :] = im_k[:, :, 1:, :, :] - im_k[:, :, :-1, :, :]

        im_k_bone = torch.zeros_like(im_k)
        for v1, v2 in self.Bone:
            im_k_bone[:, :, :, v1 - 1, :] = im_k[:, :, :, v1 - 1, :] - im_k[:, :, :, v2 - 1, :]

        # im_q_extreme
        im_q_extreme_motion = torch.zeros_like(im_q_extreme)
        im_q_extreme_motion[:, :, :-1, :, :] = im_q_extreme[:, :, 1:, :, :] - im_q_extreme[:, :, :-1, :, :]

        im_q_extreme_bone = torch.zeros_like(im_q_extreme)
        for v1, v2 in self.Bone:
            im_q_extreme_bone[:, :, :, v1 - 1, :] = im_q_extreme[:, :, :, v1 - 1, :] - im_q_extreme[:, :, :, v2 - 1, :]

        # compute joint query features and the extremely augmented joint query features
        q = self.encoder_q(im_q)  # NxC
        q_extreme = self.encoder_q(im_q_extreme)  # NxC

        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)

        # compute motion query features and the extremely augmented motion query features
        q_motion = self.encoder_q_motion(im_q_motion)  # NxC
        q_extreme_motion = self.encoder_q_motion(im_q_extreme_motion)  # NxC

        q_motion = F.normalize(q_motion, dim=1)
        q_extreme_motion = F.normalize(q_extreme_motion, dim=1)

        # compute bone query features and the extremely augmented bone query features
        q_bone = self.encoder_q_bone(im_q_bone)  # NxC
        q_extreme_bone = self.encoder_q_bone(im_q_extreme_bone)  # NxC

        q_bone = F.normalize(q_bone, dim=1)
        q_extreme_bone = F.normalize(q_extreme_bone, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            self._momentum_update_key_encoder_motion()
            self._momentum_update_key_encoder_bone()

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

            k_motion = self.encoder_k_motion(im_k_motion)
            k_motion = F.normalize(k_motion, dim=1)

            k_bone = self.encoder_k_bone(im_k_bone)
            k_bone = F.normalize(k_bone, dim=1)

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_motion = torch.einsum('nc,nc->n', [q_motion, k_motion]).unsqueeze(-1)
        l_neg_motion = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion.clone().detach()])

        l_pos_bone = torch.einsum('nc,nc->n', [q_bone, k_bone]).unsqueeze(-1)
        l_neg_bone = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_motion = torch.cat([l_pos_motion, l_neg_motion], dim=1)
        logits_bone = torch.cat([l_pos_bone, l_neg_bone], dim=1)

        # apply temperature
        logits /= self.T
        logits_motion /= self.T
        logits_bone /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # compute extreme logits
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_motion_e = torch.einsum('nc,nc->n', [q_extreme_motion, k_motion]).unsqueeze(-1)
        l_neg_motion_e = torch.einsum('nc,ck->nk', [q_extreme_motion, self.queue_motion.clone().detach()])

        l_pos_bone_e = torch.einsum('nc,nc->n', [q_extreme_bone, k_bone]).unsqueeze(-1)
        l_neg_bone_e = torch.einsum('nc,ck->nk', [q_extreme_bone, self.queue_bone.clone().detach()])

        # extreme logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_motion_e = torch.cat([l_pos_motion_e, l_neg_motion_e], dim=1)
        logits_bone_e = torch.cat([l_pos_bone_e, l_neg_bone_e], dim=1)

        # apply temperature
        logits_e /= self.T
        logits_motion_e /= self.T
        logits_bone_e /= self.T

        logits_e = torch.softmax(logits_e, dim=1)
        labels_e = logits.clone().detach()  # use normal logits as supervision
        labels_e = torch.softmax(labels_e, dim=1)
        labels_e = labels_e.detach()

        logits_motion_e = torch.softmax(logits_motion_e, dim=1)
        labels_motion_e = logits_motion.clone().detach()  # use normal logits as supervision
        labels_motion_e = torch.softmax(labels_motion_e, dim=1)
        labels_motion_e = labels_motion_e.detach()

        logits_bone_e = torch.softmax(logits_bone_e, dim=1)
        labels_bone_e = logits_bone.clone().detach()  # use normal logits as supervision
        labels_bone_e = torch.softmax(labels_bone_e, dim=1)
        labels_bone_e = labels_bone_e.detach()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_motion(k_motion)
        self._dequeue_and_enqueue_bone(k_bone)

        return logits, logits_motion, logits_bone, labels, logits_e, labels_e, logits_motion_e, \
               labels_motion_e, logits_bone_e, labels_bone_e

    def mining(self, im_q_extreme, im_q, im_k, topk):
        '''
        Second-Stage: single-stream nearest neighbors mining
        :param im_q_extreme: a batch of extreme query images
        :param im_q: a batch of query images
        :param im_k: a batch of key images
        :param topk: top-k similar in memory bank in single-stream nearest neighbors mining
        :return:
        '''
        # im_q_motion
        im_q_motion = torch.zeros_like(im_q)
        im_q_motion[:, :, :-1, :, :] = im_q[:, :, 1:, :, :] - im_q[:, :, :-1, :, :]

        # im_q_bone
        im_q_bone = torch.zeros_like(im_q)
        for v1, v2 in self.Bone:
            im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]

        # im_k
        im_k_motion = torch.zeros_like(im_k)
        im_k_motion[:, :, :-1, :, :] = im_k[:, :, 1:, :, :] - im_k[:, :, :-1, :, :]

        im_k_bone = torch.zeros_like(im_k)
        for v1, v2 in self.Bone:
            im_k_bone[:, :, :, v1 - 1, :] = im_k[:, :, :, v1 - 1, :] - im_k[:, :, :, v2 - 1, :]

        # im_q_extreme
        im_q_extreme_motion = torch.zeros_like(im_q_extreme)
        im_q_extreme_motion[:, :, :-1, :, :] = im_q_extreme[:, :, 1:, :, :] - im_q_extreme[:, :, :-1, :, :]

        im_q_extreme_bone = torch.zeros_like(im_q_extreme)
        for v1, v2 in self.Bone:
            im_q_extreme_bone[:, :, :, v1 - 1, :] = im_q_extreme[:, :, :, v1 - 1, :] - im_q_extreme[:, :, :, v2 - 1, :]

        # compute joint query features and the extremely augmented joint query features
        q = self.encoder_q(im_q)  # NxC
        q_extreme = self.encoder_q(im_q_extreme)  # NxC

        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)

        # compute motion query features and the extremely augmented motion query features
        q_motion = self.encoder_q_motion(im_q_motion)  # NxC
        q_extreme_motion = self.encoder_q_motion(im_q_extreme_motion)  # NxC

        q_motion = F.normalize(q_motion, dim=1)
        q_extreme_motion = F.normalize(q_extreme_motion, dim=1)

        # compute bone query features and the extremely augmented bone query features
        q_bone = self.encoder_q_bone(im_q_bone)  # NxC
        q_extreme_bone = self.encoder_q_bone(im_q_extreme_bone)  # NxC

        q_bone = F.normalize(q_bone, dim=1)
        q_extreme_bone = F.normalize(q_extreme_bone, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            self._momentum_update_key_encoder_motion()
            self._momentum_update_key_encoder_bone()

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

            k_motion = self.encoder_k_motion(im_k_motion)
            k_motion = F.normalize(k_motion, dim=1)

            k_bone = self.encoder_k_bone(im_k_bone)
            k_bone = F.normalize(k_bone, dim=1)

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_motion = torch.einsum('nc,nc->n', [q_motion, k_motion]).unsqueeze(-1)
        l_neg_motion = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion.clone().detach()])

        l_pos_bone = torch.einsum('nc,nc->n', [q_bone, k_bone]).unsqueeze(-1)
        l_neg_bone = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_motion = torch.cat([l_pos_motion, l_neg_motion], dim=1)
        logits_bone = torch.cat([l_pos_bone, l_neg_bone], dim=1)

        # apply temperature
        logits /= self.T
        logits_motion /= self.T
        logits_bone /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # compute extreme logits
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_motion_e = torch.einsum('nc,nc->n', [q_extreme_motion, k_motion]).unsqueeze(-1)
        l_neg_motion_e = torch.einsum('nc,ck->nk', [q_extreme_motion, self.queue_motion.clone().detach()])

        l_pos_bone_e = torch.einsum('nc,nc->n', [q_extreme_bone, k_bone]).unsqueeze(-1)
        l_neg_bone_e = torch.einsum('nc,ck->nk', [q_extreme_bone, self.queue_bone.clone().detach()])

        # extreme logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_motion_e = torch.cat([l_pos_motion_e, l_neg_motion_e], dim=1)
        logits_bone_e = torch.cat([l_pos_bone_e, l_neg_bone_e], dim=1)

        # apply temperature
        logits_e /= self.T
        logits_motion_e /= self.T
        logits_bone_e /= self.T

        logits_e = torch.softmax(logits_e, dim=1)
        labels_e = logits.clone().detach()  # use normal logits as supervision
        labels_e = torch.softmax(labels_e, dim=1)
        labels_e = labels_e.detach()

        logits_motion_e = torch.softmax(logits_motion_e, dim=1)
        labels_motion_e = logits_motion.clone().detach()  # use normal logits as supervision
        labels_motion_e = torch.softmax(labels_motion_e, dim=1)
        labels_motion_e = labels_motion_e.detach()

        logits_bone_e = torch.softmax(logits_bone_e, dim=1)
        labels_bone_e = logits_bone.clone().detach()  # use normal logits as supervision
        labels_bone_e = torch.softmax(labels_bone_e, dim=1)
        labels_bone_e = labels_bone_e.detach()

        # Topk
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)

        _, topkdix_motion = torch.topk(l_neg_motion, topk, dim=1)
        _, topkdix_motion_e = torch.topk(l_neg_motion_e, topk, dim=1)

        _, topkdix_bone = torch.topk(l_neg_bone, topk, dim=1)
        _, topkdix_bone_e = torch.topk(l_neg_bone_e, topk, dim=1)

        topk_onehot_j = torch.zeros_like(l_neg)
        topk_onehot_m = torch.zeros_like(l_neg_motion)
        topk_onehot_b = torch.zeros_like(l_neg_bone)

        topk_onehot_j.scatter_(1, topkdix, 1)
        topk_onehot_j.scatter_(1, topkdix_e, 1)

        topk_onehot_m.scatter_(1, topkdix_motion, 1)
        topk_onehot_m.scatter_(1, topkdix_motion_e, 1)

        topk_onehot_b.scatter_(1, topkdix_bone, 1)
        topk_onehot_b.scatter_(1, topkdix_bone_e, 1)

        pos_mask_j = torch.cat([torch.ones(topk_onehot_j.size(0), 1).cuda(), topk_onehot_j], dim=1)
        pos_mask_m = torch.cat([torch.ones(topk_onehot_m.size(0), 1).cuda(), topk_onehot_m], dim=1)
        pos_mask_b = torch.cat([torch.ones(topk_onehot_b.size(0), 1).cuda(), topk_onehot_b], dim=1)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_motion(k_motion)
        self._dequeue_and_enqueue_bone(k_bone)

        return logits, pos_mask_j, logits_motion, pos_mask_m, logits_bone, pos_mask_b, logits_e, labels_e, logits_motion_e, \
               labels_motion_e, logits_bone_e, labels_bone_e

    def cross_training(self, im_q_extreme, im_q, im_k, topk=4, vote=2):
        '''
        Third-Stage: multi-stream aggregation and interaction
        :param im_q_extreme: a batch of extreme query images
        :param im_q: a batch of query images
        :param im_k: a batch of key images
        :param topk: top-k similar in memory bank
        :param vote: vote value of three streams
        :return:
        '''
        # im_q_motion
        im_q_motion = torch.zeros_like(im_q)
        im_q_motion[:, :, :-1, :, :] = im_q[:, :, 1:, :, :] - im_q[:, :, :-1, :, :]

        # im_q_bone
        im_q_bone = torch.zeros_like(im_q)
        for v1, v2 in self.Bone:
            im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]

        # im_k
        im_k_motion = torch.zeros_like(im_k)
        im_k_motion[:, :, :-1, :, :] = im_k[:, :, 1:, :, :] - im_k[:, :, :-1, :, :]

        im_k_bone = torch.zeros_like(im_k)
        for v1, v2 in self.Bone:
            im_k_bone[:, :, :, v1 - 1, :] = im_k[:, :, :, v1 - 1, :] - im_k[:, :, :, v2 - 1, :]

        # im_q_extreme
        im_q_extreme_motion = torch.zeros_like(im_q_extreme)
        im_q_extreme_motion[:, :, :-1, :, :] = im_q_extreme[:, :, 1:, :, :] - im_q_extreme[:, :, :-1, :, :]

        im_q_extreme_bone = torch.zeros_like(im_q_extreme)
        for v1, v2 in self.Bone:
            im_q_extreme_bone[:, :, :, v1 - 1, :] = im_q_extreme[:, :, :, v1 - 1, :] - im_q_extreme[:, :, :, v2 - 1, :]

        # compute joint query features and the extremely augmented joint query features
        q = self.encoder_q(im_q)  # NxC
        q_extreme = self.encoder_q(im_q_extreme)  # NxC

        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)

        # compute motion query features and the extremely augmented motion query features
        q_motion = self.encoder_q_motion(im_q_motion)  # NxC
        q_extreme_motion = self.encoder_q_motion(im_q_extreme_motion)  # NxC

        q_motion = F.normalize(q_motion, dim=1)
        q_extreme_motion = F.normalize(q_extreme_motion, dim=1)

        # compute bone query features and the extremely augmented bone query features
        q_bone = self.encoder_q_bone(im_q_bone)  # NxC
        q_extreme_bone = self.encoder_q_bone(im_q_extreme_bone)  # NxC

        q_bone = F.normalize(q_bone, dim=1)
        q_extreme_bone = F.normalize(q_extreme_bone, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            self._momentum_update_key_encoder_motion()
            self._momentum_update_key_encoder_bone()

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

            k_motion = self.encoder_k_motion(im_k_motion)
            k_motion = F.normalize(k_motion, dim=1)

            k_bone = self.encoder_k_bone(im_k_bone)
            k_bone = F.normalize(k_bone, dim=1)

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_motion = torch.einsum('nc,nc->n', [q_motion, k_motion]).unsqueeze(-1)
        l_neg_motion = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion.clone().detach()])

        l_pos_bone = torch.einsum('nc,nc->n', [q_bone, k_bone]).unsqueeze(-1)
        l_neg_bone = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_motion = torch.cat([l_pos_motion, l_neg_motion], dim=1)
        logits_bone = torch.cat([l_pos_bone, l_neg_bone], dim=1)

        # apply temperature
        logits /= self.T
        logits_motion /= self.T
        logits_bone /= self.T

        # compute extreme logits
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_motion_e = torch.einsum('nc,nc->n', [q_extreme_motion, k_motion]).unsqueeze(-1)
        l_neg_motion_e = torch.einsum('nc,ck->nk', [q_extreme_motion, self.queue_motion.clone().detach()])

        l_pos_bone_e = torch.einsum('nc,nc->n', [q_extreme_bone, k_bone]).unsqueeze(-1)
        l_neg_bone_e = torch.einsum('nc,ck->nk', [q_extreme_bone, self.queue_bone.clone().detach()])

        # extreme logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_motion_e = torch.cat([l_pos_motion_e, l_neg_motion_e], dim=1)
        logits_bone_e = torch.cat([l_pos_bone_e, l_neg_bone_e], dim=1)

        # apply temperature
        logits_e /= self.T
        logits_motion_e /= self.T
        logits_bone_e /= self.T

        logits_e = torch.softmax(logits_e, dim=1)
        labels_e = logits.clone().detach()  # use normal logits as supervision
        labels_e = torch.softmax(labels_e, dim=1)
        labels_e = labels_e.detach()

        logits_motion_e = torch.softmax(logits_motion_e, dim=1)
        labels_motion_e = logits_motion.clone().detach()  # use normal logits as supervision
        labels_motion_e = torch.softmax(labels_motion_e, dim=1)
        labels_motion_e = labels_motion_e.detach()

        logits_bone_e = torch.softmax(logits_bone_e, dim=1)
        labels_bone_e = logits_bone.clone().detach()  # use normal logits as supervision
        labels_bone_e = torch.softmax(labels_bone_e, dim=1)
        labels_bone_e = labels_bone_e.detach()

        # Topk
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)

        _, topkdix_motion = torch.topk(l_neg_motion, topk, dim=1)
        _, topkdix_motion_e = torch.topk(l_neg_motion_e, topk, dim=1)

        _, topkdix_bone = torch.topk(l_neg_bone, topk, dim=1)
        _, topkdix_bone_e = torch.topk(l_neg_bone_e, topk, dim=1)

        topk_onehot_j1 = torch.zeros_like(l_neg)
        topk_onehot_j2 = torch.zeros_like(l_neg_e)
        topk_onehot_m1 = torch.zeros_like(l_neg_motion)
        topk_onehot_m2 = torch.zeros_like(l_neg_motion_e)
        topk_onehot_b1 = torch.zeros_like(l_neg_bone)
        topk_onehot_b2 = torch.zeros_like(l_neg_bone_e)

        topk_onehot_j1.scatter_(1, topkdix, 1)
        topk_onehot_j2.scatter_(1, topkdix_e, 1)

        topk_onehot_m1.scatter_(1, topkdix_motion, 1)
        topk_onehot_m2.scatter_(1, topkdix_motion_e, 1)

        topk_onehot_b1.scatter_(1, topkdix_bone, 1)
        topk_onehot_b2.scatter_(1, topkdix_bone_e, 1)

        # Interact information
        label_3views = topk_onehot_j1 * topk_onehot_j2 + \
                       topk_onehot_m1 * topk_onehot_m2 + \
                       topk_onehot_b1 * topk_onehot_b2

        label_3views = torch.where(label_3views >= vote, torch.ones(1).to(label_3views.device),
                                   torch.zeros(1).to(label_3views.device))
        label_3views.detach_()

        pos_mask = torch.cat([torch.ones(label_3views.size(0), 1).cuda(), label_3views], dim=1)

        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_motion(k_motion)
        self._dequeue_and_enqueue_bone(k_bone)

        return logits, logits_motion, logits_bone, pos_mask, logits_e, labels_e, logits_motion_e, \
               labels_motion_e, logits_bone_e, labels_bone_e