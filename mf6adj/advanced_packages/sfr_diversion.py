"""Diversions taken from a streamflow-routing reach.

A diversion removes water from the flow a reach passes on, under one of four
rules. Writing $v$ for the rate the diversion is given and $q$ for the flow
available to it when its turn comes, the rules and the derivatives that matter
to the adjoint are

===========  =====================  ==============
rule         diverted               d(diverted)/dq
===========  =====================  ==============
FRACTION     ``q * v``              ``v``
EXCESS       ``max(q - v, 0)``      1 above ``v``
UPTO         ``min(v, q)``          1 below ``v``
THRESHOLD    ``v`` if ``q >= v``    0
===========  =====================  ==============

MODFLOW 6 keeps the rule in a character array that is not in its memory store,
so which one is in effect cannot be read. It does not have to be. The flow
available to a diversion, the rate it was given, and the flow it actually took
are all recoverable, and those three determine the derivative: where two rules
give the same diverted flow they also give the same derivative, so the rule
itself can stay unknown.
"""

import numpy as np

# the flows are formed in one pass rather than iterated on, so a rule that is
# in effect reproduces the diverted flow to rounding
MATCH_TOL = 1.0e-9


def _candidates(available, requested):
    """Return what each rule would divert, and how that follows the flow."""
    return (
        (available * requested, requested),
        (max(available - requested, 0.0), 1.0 if available > requested else 0.0),
        (min(requested, available), 1.0 if available < requested else 0.0),
        (requested if available >= requested else 0.0, 0.0),
    )


def diversion_derivative(available, requested, taken):
    """Return how a diverted flow follows the flow available to the diversion.

    Returns
    -------
    tuple
        The derivative, and whether the flows determine it. Where no rule
        reproduces the diverted flow, or the rules that do disagree, the
        derivative is reported as undetermined and returned as zero.
    """
    scale = max(abs(available), abs(requested), abs(taken), 1.0)
    slopes = [
        slope
        for diverted, slope in _candidates(available, requested)
        if abs(diverted - taken) <= MATCH_TOL * scale
    ]
    if not slopes or max(slopes) - min(slopes) > MATCH_TOL:
        return 0.0, False
    return slopes[0], True


def reach_coefficients(available, requested, taken):
    """Return how the diversions of one reach share out its outflow.

    The diversions are applied in turn, each taking from what the one before it
    left, so the flow passed on carries the product of what each leaves behind.

    Parameters
    ----------
    available : float
        Flow the reach has to give before any diversion is taken.
    requested : sequence of float
        Rate each diversion is given, in the order they are applied.
    taken : sequence of float
        Flow each diversion actually took.

    Returns
    -------
    tuple
        The derivative of each diverted flow with respect to the flow the reach
        had before any diversion, the derivative of the flow passed on, and
        whether every rule was determined.
    """
    coefficients = np.zeros(len(requested))
    remaining = 1.0
    determined = True
    for i, (rate, flow) in enumerate(zip(requested, taken)):
        slope, known = diversion_derivative(available, rate, flow)
        determined = determined and known
        coefficients[i] = remaining * slope
        remaining *= 1.0 - slope
        available -= flow
    return coefficients, remaining, determined
