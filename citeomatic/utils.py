import functools
import importlib
from typing import TypeVar, Iterator, Callable, List

T = TypeVar('T')
U = TypeVar('U')

def import_from(module, name):
    '''
    usage example:
    grid = import_from('sklearn.model_selection', 'GridSearchCV')
    is equivalent to:
    from sklearn.model_selection import GridSearchV as grid
    '''
    module = importlib.import_module(module)
    # module = reload(module) # maybe needed for py2?
    return getattr(module, name)

def flatten(lst):
    """Flatten `lst` (return the depth first traversal of `lst`)"""
    out = []
    for v in lst:
        if v is None:
            continue
        if isinstance(v, list):
            out.extend(flatten(v))
        else:
            out.append(v)
    return out


def once(fn):
    cache = {}

    @functools.wraps(fn)
    def _fn():
        if 'result' not in cache:
            cache['result'] = fn()
        return cache['result']

    return _fn


def batch_apply(
    generator: Iterator[T],
    evaluator: Callable[[List[T]], List[U]],
    batch_size=256
):
    """
    Invoke `evaluator` using batches consumed from `generator`.

    Some functions (e.g. Keras models) are much more efficient when evaluted with large batches of
    inputs at a time. This function simplifies streaming data through these models.
    """
    for batch in batchify(generator, batch_size):
        yield from evaluator(batch)


def batchify(it: Iterator[T], batch_size=128) -> Iterator[List[T]]:
    batch = []
    for item in it:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch
