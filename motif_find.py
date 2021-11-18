import argparse
import pandas as pd
import numpy as np

EXTERNAL_STATES = 4


class Forward:
    def __init__(self, seq, initial_emission, p, q):
        self.seq = seq
        self.initial_emission = initial_emission
        self.p = p
        self.q = q
        self.motif_states_num = len(self.initial_emission)
        self.positions = self.motif_states_num + EXTERNAL_STATES
        self.seq_len = len(self.seq)
        self.mat = np.zeros(shape=(self.positions, self.seq_len))
        self.state_to_index = {
            'Bstart': 0,
            'B1': 1,
        }
        for i in range(self.motif_states_num):
            self.state_to_index[f'M{i}'] = i + 2
        self.state_to_index['B2'] = self.motif_states_num + 2
        self.state_to_index['Bend'] = self.motif_states_num + 3
        print(self.state_to_index)

    # def tau(self, state1, state2):
    #     transitions = np.zeros(shape=(self.positions, self.positions))
    #     transitions[self.state_to_index['Bstart']][[self.state_to_index['B1'] = self.q
    #     transitions[self.state_to_index['Bstart']][[self.state_to_index['B2'] = 1- self.q
    #     transitions[self.state_to_index['B1']][[self.state_to_index['B1'] = 1+ self.p ################
    #
        def fill_mat(self):
            for position in range(self.seq_len):
                # in first row char is '^' and we are at start state
                if position == 0:
                    self.mat[self.state_to_index['Bstart']][position] = 1
                    # others rows be zero by default
                else:
                    for state in self.positions:
                        sum = 0
                        for next_state in self.state_to_index.keys():
                            sum += self.mat[self.state_to_index[state]][position-1]\
                                   * self.tau(state, next_state)
                        sum *= self.emission[state][position]


def print_result(hmm, seq):
    # prints result strings in chunks of 50 chars
    for i in range(0, len(seq), 50):
        print(hmm[i:i + 50])
        print(seq[i:i + 50])
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', help='Algorithm (e.g. viterbi)',
                        required=True)
    parser.add_argument('seq',
                        help='A sequence over the alphabet [A,C,G,T] (e.g. ACTGGACTACGTCATGCA)')
    parser.add_argument('initial_emission',
                        help='Path to emission table (e.g. initial_emission.tsv)')
    parser.add_argument('p', help='transition probability p (e.g. 0.01)',
                        type=float)
    parser.add_argument('q', help='transition probability q (e.g. 0.5)',
                        type=float)
    args = parser.parse_args()
    seq = '^' + args.seq
    seq = seq + '$'
    initial_emission = pd.read_csv(args.initial_emission, sep='\t',
                                   index_col=0)
    if args.alg == 'viterbi':
        raise NotImplementedError

    elif args.alg == 'forward':
        forward = Forward(seq, initial_emission, args.p, args.q)


    elif args.alg == 'backward':
        raise NotImplementedError

    elif args.alg == 'posterior':
        raise NotImplementedError


if __name__ == '__main__':
    main()
