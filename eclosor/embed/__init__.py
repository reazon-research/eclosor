import os
import numpy as np
try:
    from keras.models import Model
    from keras.layers import Input, Flatten, Embedding, Dot
    from keras.optimizers import Adam
    from keras.initializers import RandomUniform
except ImportError:
    pass

class EmbedRecommender:
    def __init__(self):
        super().__init__()
        self._sid2sno = None
        self._sno2sid = None
        self._uid2uno = None
        self._uno2uid = None
        self._uno2snos = None

    def _create_interaction_matrix(self, df, item_id_column='shop_id', user_id_column='user_id'):
        sno2sid = []
        sid2sno = {}
        for no, group in enumerate(df.groupby(by=item_id_column)):
            id, records = group
            sno2sid.append(id)
            sid2sno[id] = no

        uno2uid = []
        uid2uno = {}
        uno2snos = {}
        for no, group in enumerate(df.groupby(by=user_id_column)):
            id, records = group
            uno2uid.append(id)
            uid2uno[id] = no
            uno2snos[no] = set([sid2sno[i] for i in records[item_id_column]])

        self._sid2sno = sid2sno
        self._sno2sid = np.array(sno2sid)
        self._uid2uno = uid2uno
        self._uno2uid = np.array(uno2uid)
        self._uno2snos = uno2snos

    def _create_model(self, dim=32, lr=0.05, beta_1=0.9, beta_2=0.999, decay=0.01):
        n_users = len(self._uno2uid)
        n_items = len(self._sno2sid)
        input_u = Input(shape=(1,))
        input_s = Input(shape=(1,))
        initializer = RandomUniform(minval=0., maxval=1.)
        emb_u = Embedding(n_users, dim, embeddings_initializer=initializer)(input_u)
        emb_u = Flatten()(emb_u)
        emb_s = Embedding(n_items, dim, embeddings_initializer=initializer)(input_s)
        emb_s = Flatten()(emb_s)
        distance = Dot(axes = 1, normalize=True)([emb_u, emb_s])
        model = Model([input_u, input_s], distance)
        opt = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
        self._model = model

    def _get_weights(self):
        e_u, e_s = self._model.get_weights()
        self._e_u = e_u / np.linalg.norm(e_u, axis=1, keepdims=True)
        self._e_s = e_s / np.linalg.norm(e_s, axis=1, keepdims=True)

    def _resample_cases(self):
        self._get_weights()
        res = []
        n_hit = 0
        n_try = 0
        for uno, snos in self._uno2snos.items():
            n_pos = len(snos)
            n_neg = 0
            res += [(uno, sno, 1) for sno in snos]
            cos = self._e_s @ self._e_u[uno].T
            for sno in np.argsort(cos)[::-1]:
                n_try += 1
                if sno in snos:
                    n_hit += 1
                else:
                    res.append((uno, sno, 0))
                    n_neg += 1
                    if n_neg >= n_pos:
                        break
        return np.array(res)

    def train_by_positive_interactions(self, df, dim=32, iterations=30, batch_size=30000, epochs=100, verbose=1):
        self._create_interaction_matrix(df)
        self._create_model(dim)
        for n_iter in range(iterations):
            if verbose:
                print(f'Iteration {n_iter+1}/{iterations}')
            train_data = self._resample_cases()
            self._model.fit([train_data[:,0], train_data[:,1]], train_data[:,2],
                            batch_size=batch_size, epochs=epochs, verbose=verbose)
        self._get_weights()

    def save_model(self, data_directory):
        np.savez_compressed(os.path.join(data_directory, 'embed.npz'),
                            sno2sid=self._sno2sid,
                            uno2uid=self._uno2uid,
                            e_u=self._e_u,
                            e_s=self._e_s)

    def load_model(self, data_directory):
        with np.load(os.path.join(data_directory, 'embed.npz')) as npz:
            self._e_u = npz['e_u']
            self._e_s = npz['e_s']
            self._sno2sid = npz['sno2sid']
            self._sid2sno = { id : no for no, id in enumerate(self._sno2sid) }
            self._uno2uid = npz['uno2uid']
            self._uid2uno = { id : no for no, id in enumerate(self._uno2uid) }

    def predict(self, user_id, limit=10):
        if user_id not in self._uid2uno:
            return []
        cos = self._e_s @ self._e_u[self._uid2uno[user_id]].T
        return [self._sno2sid[no] for no in np.argsort(cos)[:-(limit+1):-1]]

    def rerank(self, user_id, item_ids, limit=10):
        if user_id not in self._uid2uno:
            return item_ids
        e_u = self._e_u[self._uid2uno[user_id]].T
        cos = [self._e_s[self._sid2sno.get(id, 0)] @ e_u for id in item_ids]
        return [item_ids[i] for i in np.argsort(cos)[:-(limit+1):-1]]

    def rerank_with_score(self, user_id, item_ids, limit=10, score_threshold=0):
        if user_id not in self._uid2uno:
            return item_ids
        e_u = self._e_u[self._uid2uno[user_id]].T
        cos = [(self._e_s[self._sid2sno.get(id, 0)] @ e_u, id) for id in item_ids]
        res = [(score, id) for score, id in cos if score >= score_threshold]
        res.sort(reverse=True)
        scores, ids = zip(*res[:limit])
        return ids, scores

    def is_known_user(self, user_id):
        return user_id in self._uid2uno

if __name__ == '__main__':
    import click
    import pyarrow as pa
    @click.command()
    @click.option('--orders-data',
                  default='data',
                  help='The orders arrowfile path. (ex. ../data/orders.arrow)',
                  show_default=True)
    @click.option('--output-directory',
                  default='data',
                  help='The output data directory. (ex. ../data)',
                  show_default=True)
    def main(orders_data, output_directory):
        emb = EmbedRecommender()
        emb.train_by_positive_interactions(pa.ipc.open_file(orders_data).read_pandas())
        emb.save_model(output_directory)
        # load
        emb.load_model(output_directory)
        # sample user
        u = emb._uno2uid[0]
        # recommended items
        l = emb.predict(u)
        # rerank
        l.reverse()
        assert(emb.rerank(u, l)==emb.predict(u))
    main()
