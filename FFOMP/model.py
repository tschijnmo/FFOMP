"""
The abstract model base class
=============================

This module contains the definition for the abstract base class for models.
Although not required, in favor of duck-typing, actual models are adviced to be
derived from this base class to make sure that the subclass implements all the
required methods.

.. autosummary::
    :toctree:

    Model

"""


import abc


class Model(abc.ABC):

    """
    The abstract base class for models

    This abstract base class defines all the behaviour required for models,

    1. The capability to be directly called by the raw dictionary of the data
       points to return the dictionary for the symbolic results for the
       modelled property,
    2. Having the ``model_params`` property to retrieve a definite sequence of
       the model parameters, as ``ModelParam`` instances,
    3. The capability to present the result nicely when called on the
       ``present`` method with the fitted values of the symbolic parameters.

    """

    @abc.abstractmethod
    def __call__(self, data_pnt):
        """Computes the symbolic value of the properties at the data point

        :param data_pnt: The dictionary for the all the properties, input and
            output, for the data point.
        :returns: The dictionary for the properties able to be modelled by the
            model. With the keys being the property names and values being the
            symbolic expression for the modelled results based on the model
            parameters.
        :rtype: dict
        """

        pass

    @abc.abstractproperty
    def model_params(self):
        """The model parameters

        The value is a sequence of the sympy symbols for the model parameters,
        with each entry being a ``ModelParam`` instance.
        """

        pass

    @abc.abstractmethod
    def present(self, param_vals, output):
        """Presents the result of the fitting textually

        This function will be called by the fit job instance with the sequence
        of the numerical values for the optimized parameters in the same order
        as the symbols. And the model is responsible to formatting and writting
        the results according to their own semantics. The output stream should
        also be given as an output stream object.
        """

        pass
