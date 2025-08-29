========
Notation
========
Demesinfer (and subsequently, the momi3 module included in it) uses the following notation for demographic models.

- **Paths** A single parameter representing a feature of the population history. Examples include population sizes, split times, and migration rates.

In demesinfer, we follow the notation used in demes package, and represents it by a tuple of strings and integers.

For example, if the first deme ("demes", 0) is the ancestral population "anc", with only 1 epoch,

the starting and ending size of the ancestral population "anc" can be represented as:

("demes", 0, "epochs", 0, "start_size"), ("demes", 0, "epochs", 0, "end_size")

Similarly, the starting and ending time of the first epoch can be represented as:

("demes", 0, "epochs", 0, "start_time"), ("demes", 0, "epochs", 0, "end_time")

Additionally, consider a simple IWM model, the migration rate between two populations can be represented as:

("migrations", 0, "rate")

- **Parameter** A frozenset of a couple of paths that are enforced to be equal and optimized together during inference.

For example, in a simple IWM model with three populations and each having 1 epoch, the starting time of the

two descendant populations and the ending time of the ancestral population are equal, and can be represented as a 

single parameter:

frozenset({('demes', 0, 'epochs', 0, 'end_time'),
                ('demes', 1, 'start_time'),
                ('demes', 2, 'start_time'),
                ('migrations', 0, 'start_time'),
                ('migrations', 1, 'start_time')})
