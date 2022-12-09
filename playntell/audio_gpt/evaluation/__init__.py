from .bleu import Bleu
from .cider import Cider
from .meteor import Meteor
from .rouge import Rouge
from .tokenizer import PTBTokenizer


def compute_scores(gts, gen, what):
    if what == "BLEU":
        metric = Bleu()
    elif what == "METEOR":
        metric = Meteor()
    elif what == "ROUGE":
        metric = Rouge()
    elif what == "CIDEr":
        metric = Cider()
    else:
        raise ValueError(f"Unknown metric {what}")

    score, scores = metric.compute_score(gts, gen)

    return score, scores
