import os
import fasttext
import random
import nltk
import numpy as np

from helpers import normalize_name, StatCalc
from PokemonML.value_network.data.categories import (
    SPECIES,
    ITEMS,
    ABILITIES,
    MOVES,
)


def main():
    embedder = Embedder()
    move_embedder = MoveEmbedder()

    # move_embedder.create_embeddings()

    # embedder.create_model('sents_per_poke_1200+.txt', dims=4)
    # embedder.create_model('sents_per_poke_1200+.txt', dims=8)
    # embedder.create_model('sents_per_poke_1200+.txt', dims=12)
    # embedder.create_model('sents_per_poke_1200+.txt', dims=16)
    # embedder.create_model('sents_per_poke_1200+.txt', dims=24)
    # embedder.create_model('sents_per_poke_1200+.txt', dims=32)
    # embedder.create_model('sents_per_poke_1200+.txt', dims=48)
    # embedder.create_model('sents_per_poke_1200+.txt', dims=64)
    # embedder.create_model('sents_per_poke_1200+.txt', dims=96)
    # embedder.create_model('sents_per_poke_1200+.txt', dims=128)

    # embedder.test_random_embeddings(SPECIES, 10, dims=16)
    # embedder.test_random_embeddings(ITEMS, 10, dims=16)
    # embedder.test_random_embeddings(MOVES, 10, dims=16)
    # embedder.test_random_embeddings(ABILITIES, 10, dims=16)

    # embedder.test_embedding_dims('durant', SPECIES)
    # embedder.test_embedding_dims('aguavberry', ITEMS)
    # embedder.test_embedding_dims('hex', MOVES)
    # embedder.test_embedding_dims('drought', ABILITIES)


class Embedder:
    def __init__(self):
        self.sents_folder = "sent_files/"
        self.poke_file = 'sents_per_poke.txt'

        self.model_folder = "model_files/"

    def create_model(self, dims=32):
        sents_file = os.path.join(self.sents_folder, self.poke_file)
        model = fasttext.train_unsupervised(sents_file, model="cbow", dim=dims, ws=50, minn=0, maxn=0)

        out_file = os.path.join(self.model_folder, 'poke_embeddings_' + str(dims) + '_dim.bin')
        model.save_model(out_file)

        print(f'model saved at:\n'
              f'{out_file}')

    def test_embedding_dims(self, entity, category, dims=(4, 8, 12, 16, 24, 32, 48, 64, 96, 128)):
        for dim in dims:
            file_name = 'poke_embeddings_' + str(dim) + '_dim.bin'

            model_file = os.path.join(self.model_folder, file_name)
            model = fasttext.load_model(model_file)

            print('dims:', dim)
            self.show_most_similar(entity, model, category.keys())

    def test_random_embeddings(self, category, n, dims=32):
        file_name = 'poke_embeddings_' + str(dims) + '_dim.bin'

        model_file = os.path.join(self.model_folder, file_name)
        model = fasttext.load_model(model_file)

        for i in range(n):
            entity, _ = random.choice(list(category.items()))
            self.show_most_similar(entity, model, category.keys())

    def show_most_similar(self, term, model, vocabulary, num_terms=10):
        vocabulary = set(vocabulary)
        neighbors = model.get_nearest_neighbors(term, k=100)
        keep = [(sim, word) for (sim, word) in neighbors if word in vocabulary]

        print(term)
        for result in keep[:num_terms]:
            print(result)
        print()


class MoveEmbedder:
    def __init__(self):
        self.sents_folder = "sent_files/"
        self.move_file = 'sents_per_move.txt'

    def create_embeddings(self, file_out='bag_of_moves.txt'):
        file = os.path.join(self.sents_folder, self.move_file)

        all_attributes = []
        move_attributes = {}

        with open(file, 'r') as f_out:
            for line in f_out:
                sent = nltk.word_tokenize(line)

                move = sent.pop(0)
                move_attributes[move] = sent

                for word in sent:
                    all_attributes.append(word)

        all_attributes = sorted(set(all_attributes))
        embeddings = np.zeros((len(MOVES) + 1, len(all_attributes)), dtype=int)

        for move, i in MOVES.items():
            attributes = move_attributes[move]

            for j, att in enumerate(all_attributes):
                if att in attributes:
                    embeddings[i][j] = 1

        # file = os.path.join(self.sents_folder, file_out)
        # np.savetxt(file, embeddings.astype(int), delimiter=',')












if __name__ == "__main__":
    main()

