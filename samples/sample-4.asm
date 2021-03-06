MEM 1024
SET 1 0  # Count
SET 2 100  # Array
SET 3 0  # Pointer
READ 4  # n
SET 9 200
ADD 4 9 11
MEM *11

# Generate a sequence of 1 ... n
ADD 1 2 3
CPY 1 *3
INC *3
PRT **3
INC 1
LESS 1 4 6
JIF 6 11

# Binary search
READ 10  # Target
SET 11 0  # Left
SET 12 *4  # Right
DEC 12
ADD 11 12 13  # Sum/Middle
SHR 13 1  # Divide with two

# Compare mid to target
ADD 2 13 50
CPY *50 49
EQU 49 10 51  # 51 is result
JIF 51 75  # Exit

# Detect direction
LESS 10 49 51
JIF 51 37  # True
JMP 39  # False
CPY 13 12

GTER 10 49 51  # Else
JIF 51 42  # True
JMP 44  # False
CPY 13 11

NOP

SUB 12 11 52  # Distance of Right and Left

# Abs
SET 55 0
LESS 52 55 51
JIF 51 53
JMP 54
NEC 52
NOP

SET 60 2
LESS 52 60 51
JIF 51 60
JMP 24 # To Beginning
SET 23 0
EQU 52 23 55  # Is distance != 0?
NOT 55
JIF 55 67
JMP 75

# Check whether left is correct
ADD 2 11 50
CPY *50 49
EQU 49 10 55
JIF 55 73
CPY 12 13
JMP 75
CPY 11 13

NOP
CPY 13 11
EQU 11 10 50
NOT 50
JIF 50 82 
PRT *11
JMP 84
PRT -1

EXIT 0
