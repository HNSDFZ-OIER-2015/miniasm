MEM 1024
SET 1 0  # Count
SET 2 100  # Array
SET 3 0  # Pointer
READ 4  # n
SET 9 200
ADD 4 9 11
MEM *11

# Generate a sequence of 1 ... n
NOP 70
ADD 1 2 3
READ *3
# INC *3
# PRT **3
INC 1
LESS 1 4 6
JIF 6 *70

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
JIF 51 *80  # Exit

# Detect direction
LESS 10 49 51
JIF 51 2  # True
JMP 3  # False
CPY 13 12
JMP *72

GTER 10 49 51  # Else
JIF 51 2  # True
JMP *72  # False
CPY 13 11

NOP 72

SUB 12 11 52  # Distance of Right and Left

# Abs
SET 55 0
LESS 52 55 51
JIF 51 2
JMP 2
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

NOP *80
CPY 13 11
ADD 2 11 13
EQU *13 10 50
NOT 50
JIF 50 81
INC 11
PRT *11
JMP 86
PRT -1

EXIT 0
