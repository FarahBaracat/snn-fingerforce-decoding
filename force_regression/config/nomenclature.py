# from enum import Enum

# class Nomen(Enum):
#     """
#     TaskTaxonomy is an enumeration that characterizes the tasks.

#     Attributes:
#         FLEX (str): Represents the 'flex' task.
#         EXT (str): Represents the 'ext' task.
#         FORCE (str): Represents the 'force' measurements
#         INTRA (str): Represents the data from intramuscular electrodes.
#         SURFACE (str): Represents the data from surface electrodes.
#         TRAP (str): Represents the trapezoid task.
#         PSEUDO (str): Represents the pseudo task.
#     """
FLEX = 'flex'
EXT = 'ext'
FORCE = 'force'     # the force measurements
TARGET = 'target'   # the clean target to follow
EMG = 'EMG'         # the raw EMG data

INTRA = 'intra'
SURFACE = 'surf'
TRAP = 'Trap'
PSEUDO = 'Pseudo'
FIRST_REP = 'rep1'
SECOND_REP = 'rep2'
LANDMARK = 'landmark'  # landmark points on the trapexoid force profile
MU_TIMESTAMP = 'timestamps'
FR_FIRST_REP = 'fr_rep1'
FR_SECOND_REP = 'fr_rep2'
    