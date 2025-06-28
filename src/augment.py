import random

import nlpaug.augmenter.word as naw


def augment_abstract(text: str, n: int = 2) -> list:
    """
    Generate n augmented versions of an abstract using synonym replacement.
    """
    aug = naw.SynonymAug(aug_src="wordnet")
    augmented_texts = []

    for _ in range(n):
        augmented = aug.augment(text)
        if isinstance(augmented, list):
            augmented = random.choice(augmented)
        augmented_texts.append(augmented)

    return augmented_texts
