Notation
========

Demesinfer (and the ``momi3`` module built on top of it) uses the following notation for demographic models.

Paths
-----

A **path** is a tuple of strings and integers that uniquely identifies a single parameter of the population history.  
Examples include population sizes, split times, and migration rates.

We follow the notation used in the ``demes`` package.

**Examples:**

- Starting and ending size of the ancestral population ``anc`` (the first deme, index 0, with one epoch):

  .. code-block:: python

      ("demes", 0, "epochs", 0, "start_size")
      ("demes", 0, "epochs", 0, "end_size")

- Starting and ending time of the first epoch of ``anc``:

  .. code-block:: python

      ("demes", 0, "epochs", 0, "start_time")
      ("demes", 0, "epochs", 0, "end_time")

- Migration rate between two populations in a simple IWM model:

  .. code-block:: python

      ("migrations", 0, "rate")


Parameters
----------

A **parameter** is a ``frozenset`` of multiple paths.  
Using a ``frozenset`` means these paths are **enforced to take the same value** and are optimized together during inference.

**Example:**

In a simple IWM model with three populations (each with one epoch), the starting time of the two descendant populations and the ending time of the ancestral population should be equal.  
This can be represented as a single parameter:

.. code-block:: python

    frozenset({
        ("demes", 0, "epochs", 0, "end_time"),
        ("demes", 1, "start_time"),
        ("demes", 2, "start_time"),
        ("migrations", 0, "start_time"),
        ("migrations", 1, "start_time"),
    })