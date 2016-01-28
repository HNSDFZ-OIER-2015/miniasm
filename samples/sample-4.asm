MEM 1024
SET 1 1  # Count
SET 2 100  # Offest
SET 3 0  # Pointer
SET 4 101  # Bound

# Generate a sequence of 1 ... 100
ADD 1 2 3
DEC 3
CPY 1 3
PRT *3
INC 1
LESS 1 4 5
JIF 5 6
EXIT 0
