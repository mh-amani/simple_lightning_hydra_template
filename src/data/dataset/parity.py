from typing import Dict, Union, List
import random
from datasets import Dataset
import numpy as np

class ParityDataset(Dataset):
    def __init__(self, sequence_length: int, dataset_size: int = 2 ** 10):
        self.sequence_length = sequence_length
        self.data_size = dataset_size  

    def generate_parity_sequence(self) -> Dict[str, Union[List[int], int]]:
        sequence = [random.randint(0, 1) for _ in range(self.sequence_length)]
        cot = []
        current_parity = sequence[0]
        for i in range(1, len(sequence)):
            current_parity = current_parity ^ sequence[i]
            cot.append(current_parity)

        return {
            "sequence": sequence,
            "cot": cot,
            "answer": current_parity
        }

    def __getitem__(self, index: Union[int, List[int]]) -> Dict[str, Union[List[List[int]], List[int]]]:
        if isinstance(index, int):
            return self.generate_parity_sequence()
        elif isinstance(index, list):
            # Create a dictionary with batched values
            batch = [self.generate_parity_sequence() for _ in index]
            return {
                "sequence": np.array([item["sequence"] for item in batch]),
                "cot": np.array([item["cot"] for item in batch]),
                "answer": np.array([item["answer"] for item in batch])
            }
        else:
            raise TypeError("Index must be an int or a list of ints")


    def __len__(self) -> int:
        return self.data_size

# Example usage:
# dataset = ParityDataset(sequence_length=5)
# sample = dataset[0]  # Get a fresh sample
# print(sample)  # Output: {'sequence': [1, 0, 1, 1, 0], 'cot': [1, 0, 1, 0], 'answer': 0}
