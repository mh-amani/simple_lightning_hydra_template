import math
import numpy as np
import argparse
import os
import datasets

def create_data(n, d, L, T, l, eps, rng):

    def sample_rotation_near_identity(d, eps, rng):
        M = rng.standard_normal((d, d)) / np.sqrt(d)
        S = (M - M.T).astype(np.float64)
        I = np.eye(d, dtype=np.float64)
        O = np.linalg.solve(I - eps * S, I + eps * S)  # (I - eps * S)^{-1}(I + eps * S) is orthogonal
        return O

    def normalize_to_sqrt_d(x):
        nrm = np.linalg.norm(x) + 1e-12
        return x * (math.sqrt(d) / nrm)
    
    def generate_sequence(d, L, eps, rng):
        A = sample_rotation_near_identity(d, eps, rng)
        x0 = normalize_to_sqrt_d(rng.standard_normal(d).astype(np.float32))
        xs = [x0]
        for _ in range(L):
            xs.append(A @ xs[-1])
        y = xs[-1]
        for _ in range(T - L):
            y = A @ y
        return np.stack(xs, axis=0), A, y

    X_list, y_list, A_list, x0_list = [], [], [], []
    for _ in range(n):
        X, A, y = generate_sequence(d, L, eps, rng)
        X_list.append(X.astype(np.float32))
        y_list.append(y.astype(np.float32))
        A_list.append(A.astype(np.float32))
        x0_list.append(X[0].astype(np.float32))
    return np.stack(X_list), np.stack(y_list), A_list, x0_list


def return_dataset_dict(kwargs):
    rng = np.random.default_rng(kwargs['seed'])
    Xtr, ytr, Atr, x0tr = create_data(kwargs['train_size'], kwargs['d'], kwargs['L'], kwargs['T'], kwargs['l'], kwargs['eps'], rng)
    Xte, yte, Ate, x0te = create_data(kwargs['test_size'], kwargs['d'], kwargs['L'], kwargs['T'], kwargs['l'],  kwargs['eps'], rng)
    dataset_dict = datasets.DatasetDict({
        'train': datasets.Dataset.from_dict({
            'x': Xtr, 
            'y': ytr.tolist(),
            'A': Atr,
            'x0': x0tr
        }),
        'validation': datasets.Dataset.from_dict({
            'x': Xte,
            'y': yte.tolist(),
            'A': Ate,
            'x0': x0te
        }),
        'test': datasets.Dataset.from_dict({
            'x': Xte,
            'y': yte.tolist(),
            'A': Ate,
            'x0': x0te
        })
    })
    return dataset_dict



if __name__ == '__main__':

    # print current working directory
    print(os.getcwd())
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name_or_path', default='scan')
    # parser.add_argument('--name', default='simple')
    parser.add_argument('--local_dir', default=None)
    parser.add_argument('--train_size', type=int, default=512)
    parser.add_argument('--validation_size', type=int, default=2048)
    parser.add_argument('--test_size', type=int, default=2048)

    # kwargs required to create the dataset
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--d', type=int, default=30)
    parser.add_argument('--L', type=int, default=50)
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--l', type=int, default=30)
    parser.add_argument('--T', type=int, default=100)

    args = parser.parse_args()
    kwargs = vars(args)

    if args.local_dir is None:
        local_dir = f'./data/{args.dataset_name_or_path}/train_{args.train_size}_validation_{args.validation_size}_test_{args.test_size}_'
    else:
        local_dir = args.local_dir

    # dataset = datasets.load_dataset(args.dataset_name_or_path, args.name)

    dataset = return_dataset_dict(kwargs)

    train_data = dataset['train']

    if dataset.get('test', None) is not None:
        test_data = dataset['test']
    else:
        train_data, test_data = train_data.train_test_split(test_size=args.test_size).values()
    
    if dataset.get('validation', None) is not None:
        validation_data = dataset['validation']
    else:
        train_data, validation_data = train_data.train_test_split(test_size=args.validation_size).values()
     
    print(f"Saving to {local_dir}")

    # Save each dataset into its respective split folder
    train_data.to_parquet(os.path.join(local_dir, 'train.parquet'))
    validation_data.to_parquet(os.path.join(local_dir, 'validation.parquet'))
    test_data.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # saving sizes, samples, and informations about the datasets
    with open(os.path.join(local_dir, 'dataset_info.txt'), 'w') as f:
        f.write(f"Train size: {len(train_data)}\n")
        f.write(f"Validation size: {len(validation_data)}\n")
        f.write(f"Test size: {len(test_data)}\n")
        f.write(f"Sample of train data: {train_data[0]}\n")
        f.write(f"Sample of validation data: {validation_data[0]}\n")
        f.write(f"Sample of test data: {test_data[0]}\n")


    

# example usage
# python scripts/create_data/sphere.py --dataset_name_or_path sphere --train_size 512 --validation_size 2048 --test_size 2048 --d 30 --L 50 --eps 0.01 --l 30 --T 100 --seed 42 
