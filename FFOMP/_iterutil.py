"""
Some utilities for iterating over nested sequences
==================================================

The properties of data points are supported to be tensors of any order. For
properties that are tensors of order greater than two, they are going to be
stored as nested lists for both the numeric reference values and the symbolic
model values. This model contains some utility functions for iterating over
such kind of nested sequences.

.. autosummary::
    :toctree:

    flatten_zip
    map_nested

"""


from collections import abc
import itertools


def flatten_zip(tensor1, tensor2):
    """Flattens and zips two nested sequences correspondingly

    The return value will always be a list of corresponding pairs of the scalar
    values in the two given tensors. For scalars (0-order tensor), the list
    will be of length one.

    :param tensor1: The first tensor, given as possibly nested sequences of
        values.
    :param tensor2: The second tensor.
    :returns: The list of corresponding pairs of values in the two given
        tensors.
    :rtype: list
    :raises ValueError: if the two given tensors are not of exactly the same
        shape.
    """

    # First check for None, which will be the case for lists of unequal
    # lengths.
    if tensor1 is None or tensor2 is None:
        raise ValueError()

    if isinstance(tensor1, abc.Iterable):
        # If we are not at the lowest scalar level.
        if isinstance(tensor2, abc.Iterable):
            # If the corresponding value is correspondingly iterable, as
            # expected.
            return list(itertools.chain.from_iterable(
                flatten_zip(i, j)
                for i, j in itertools.zip_longest(tensor1, tensor2)
                ))
        else:
            # If the corresponding reference value is not
            # correspondingly iterable.
            raise ValueError()
    else:
        # If we are at a scalar value.
        if isinstance(tensor2, abc.Iterable):
            # When the reference value is uncorrespondingly iterable.
            raise ValueError()
        else:
            # When the reference value is correspondingly scalar.
            return [(tensor1, tensor2), ]


def map_nested(func, nested_iterable):
    """Map a function to possibly nested sequences

    In the result, the structure of the original possibly-nested sequences are
    kept the same of the given value.

    :param function func: The function on scalar values that is to be maped.
    :param nested_iterable: The possibly-nested sequence of non-iterable scalar
        values.
    :returns: The same structure, the scalar values replaced by the result of
        the function application.
    """

    if isinstance(nested_iterable, abc.Iterable):
        return [
            map_nested(func, i) for i in nested_iterable
            ]
    else:
        return func(nested_iterable)


def map_binary_nested(func, tensor1, tensor2):
    """Maps a binary function to two nested sequences

    Corresponding elements in the two possibly-nested structures will be zipped
    together to be fed to the given function, with the result being put into
    the corresponding position in the result.

    :param function func: The function to be mapped over.
    :param tensor1: The first possibly-nested structure.
    :param tensor2: The second possibly-nested structure, with the same shape
        as the first structure.
    :returns: A structure with the same structure as the two arguments, with
        results being the result of applying the binary function onto the
        corresponding elements of the two arguments at the position.
    :raises ValueError: if the two structures are not of the same shape.
    """

    if isinstance(tensor1, abc.Iterable):

        if isinstance(tensor2, abc.Iterable):
            return type(tensor1)(
                map_binary_nested(func, i, j)
                for i, j in zip(tensor1, tensor2)
                )
        else:
            raise ValueError()

    else:

        if isinstance(tensor2, abc.Iterable):
            raise ValueError()
        else:
            return func(tensor1, tensor2)
