import argparse

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
