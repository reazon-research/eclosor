import numpy as np
import os
import pickle
try:
    import torch
    import torch.nn as nn
    class GMFTModel(nn.Module):
        def __init__(self, user_common_features, item_common_features, item_coordinates, item_temporal_features,
                     embedding_dim, hidden_layer_dim, time_emb_dim):
            super(GMFTModel, self).__init__()
            self.item_coordinates = item_coordinates
            self.coordinates = item_coordinates.unsqueeze(0)
            self.item_emb = nn.Embedding(len(item_common_features), embedding_dim, max_norm=1.)
            nn.init.uniform_(self.item_emb.weight, a=-1.0, b=1.0)
            self.user_emb = nn.Embedding(len(user_common_features), embedding_dim, max_norm=1.)
            nn.init.uniform_(self.user_emb.weight, a=-1.0, b=1.0)
            self.embedding_dim = embedding_dim
            self.item_features = item_common_features
            self.user_features = user_common_features
            self.distance_filter = nn.Sequential(
                nn.Linear(1, 1),
                nn.Sigmoid(),
            )
            self.mlp = nn.Sequential(
                nn.Linear(item_common_features.shape[1], hidden_layer_dim),
                nn.BatchNorm1d(hidden_layer_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_layer_dim, hidden_layer_dim>>1),
                nn.BatchNorm1d(hidden_layer_dim>>1),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_layer_dim>>1, hidden_layer_dim>>2),
            )
            self.time_emb = nn.Embedding(item_temporal_features.shape[1], time_emb_dim, max_norm=1.)
            nn.init.uniform_(self.time_emb.weight, a=-1.0, b=1.0)
            self.item_temporal_features = nn.Embedding.from_pretrained(torch.FloatTensor(item_temporal_features))

        def merged_item_features(self):
            return torch.cat([self.item_emb.weight,
                              self.mlp(self.item_features),
                              self.item_temporal_features.weight @ self.time_emb.weight], 1)

        def merged_user_features(self, user, time):
            return torch.cat([self.user_emb.weight[user],
                              self.mlp(self.user_features[user]),
                              self.time_emb.weight[time]], 1)

        def forward(self, x):
            user, coordinate, time = x
            distance = torch.linalg.norm(self.coordinates - coordinate.view(-1, 1, 2), dim=2)
            distance = distance.unsqueeze(2)
            distance = self.distance_filter(distance)
            distance = distance.squeeze(2)
            score = self.merged_user_features(user, time) @ self.merged_item_features().T
            return score * distance

        def save(self, model_file):
            self.eval()
            with torch.no_grad():
                torch.save(self.state_dict(), model_file)

        def load(self, model_file):
            self.load_state_dict(torch.load(model_file))
except ImportError:
    pass

class GMFRecommender:
    def _update_dicts(self):
        self._sid2sno = { id : no for no, id in enumerate(self.items) }
        self._uid2uno = { id : no for no, id in enumerate(self.users) }
        self.nusers = len(self.users)

    def init_model(self, users, items, user_common_features, item_common_features, item_coordinates, item_temporal_features,
                   embedding_dim=64, hidden_layer_dim=32, time_emb_dim=4):
        self.items = items
        self.users = users
        self._update_dicts()
        self.model = GMFTModel(user_common_features, item_common_features, item_coordinates, item_temporal_features,
                               embedding_dim, hidden_layer_dim, time_emb_dim)
    def save(self, model_file):
        self.model.eval()
        with torch.no_grad():
            self.params = dict(
                items = self.items,
                users = self.users,
                item_coordinates = self.model.item_coordinates,
                item_feature = self.model.merged_item_features().cpu().detach().numpy(),
                user_feature = torch.cat([self.model.user_emb.weight, self.model.mlp(self.model.user_features)], 1).cpu().detach().numpy(),
                time_emb = self.model.time_emb.weight.cpu().detach().numpy(),
                distance_weight = self.model.distance_filter[0].weight.cpu().detach().numpy()[0],
                distance_bias = self.model.distance_filter[0].bias.cpu().detach().numpy()[0]
            )
            pickle.dump(self.params, open(model_file + '.pkl', 'wb'))

    def load(self, model_file):
        self.params = pickle.load(open(model_file + '.pkl', 'rb'))
        self.items = self.params['items']
        self.users = self.params['users']
        self._update_dicts()

    def _user_number(self, user_id):
        return self._uid2uno.get(user_id, self.nusers)

    def _item_numbers(self, item_ids):
        return [self._sid2sno.get(id, 0) for id in item_ids]

    def _feature_score(self, user, time, items=None):
        if items:
            item_feature = self.params['item_feature'][items]
        else:
            item_feature = self.params['item_feature']
        return np.concatenate((self.params['user_feature'][user], self.params['time_emb'][time])) @ item_feature.T

    def _distance_score(self, coordinate, items=None):
        if items:
            item_coordinates = self.params['item_coordinates'][items]
        else:
            item_coordinates = self.params['item_coordinates']
        distance = np.linalg.norm(item_coordinates - coordinate, axis=1)
        distance = distance * self.params['distance_weight'] + self.params['distance_bias']
        return 1/(1 + np.exp(-distance))

    def predict(self, user_id, coordinate, time, limit=10):
        user = self._user_number(user_id)
        score = self._feature_score(user, time) * self._distance_score(coordinate)
        return [self.items[no] for no in np.argsort(score)[:-(limit+1):-1]]

    def rerank(self, user_id, coordinate, time, item_ids, limit=10):
        user = self._user_number(user_id)
        items = self._item_numbers(item_ids)
        score = self._feature_score(user, time, items) * self._distance_score(coordinate, items)
        return [item_ids[i] for i in np.argsort(score)[:-(limit+1):-1]]

    def get_scores(self, user_id, coordinate, time, item_ids):
        user = self._user_number(user_id)
        items = self._item_numbers(item_ids)
        return self._feature_score(user, time, items) * self._distance_score(coordinate, items)

    def rerank_with_score(self, user_id, coordinate, time, item_ids, limit=10, score_threshold=0):
        user = self._user_number(user_id)
        items = self._item_numbers(item_ids)
        score = self._feature_score(user, time, items) * self._distance_score(coordinate, items)
        order = np.argsort(score)[:-(limit+1):-1]
        scores = [score[i] for i in order]
        ids = [item_ids[i] for i in order]
        return ids, scores

    def is_known_user(self, user_id):
        return user_id in self._uid2uno
