Notation
========

Demesinfer (and the ``momi3`` module built on top of it) uses the following notation for demographic models.

Paths
-----

A **path** is a tuple of strings and integers that uniquely identifies a single parameter of the population history. These paths correspond to the nested dictionary structure used by msprime's Demography class and follow the same notation as the ''demes'' package.

To better understand this structure, consider the following example model:

.. code-block:: python

    import msprime as msp
    demo = msp.Demography()
    demo.add_population(initial_size=5000, name="anc")
    demo.add_population(initial_size=5000, name="P0")
    demo.add_population(initial_size=5000, name="P1")
    demo.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.0001)
    tmp = [f"P{i}" for i in range(2)]
    demo.add_population_split(time=1000, derived=tmp, ancestral="anc")

Note: This code can be cleaned up, there's some unnecessary lines like tmp. Shorten this code.

To inspect, debug, and understand the demographic model's data structure, one can view the exact model with the following:

.. code-block:: python

    g.as_dict()

Note: edit this to display the dictionary output

This dictionary contains all demographic parameters in a hierarchical format, and a path corresponds to the specific sequence of keys needed to access any particular parameter within this nested structure.

**Examples:**

Starting and ending size of the ancestral population ``anc`` (the first deme, index 0, with one epoch):

.. code-block:: python

    ("demes", 0, "epochs", 0, "start_size")
    ("demes", 0, "epochs", 0, "end_size")

Starting and ending time of the first epoch of ``anc``:

.. code-block:: python

    ("demes", 0, "epochs", 0, "start_time")
    ("demes", 0, "epochs", 0, "end_time")

Migration rate between two populations in a simple IWM model:

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
