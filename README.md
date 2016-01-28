# miniasm V0.0.3
* Add keyword `READ`:
```
READ 1
PRT *1  # Will output what you inputed
```

* Add keyword `LEQ` and `GEQ`, which means `less equal` and `greater equal`:
```
LEQ index index index
GEQ index index index
```

# miniasm V0.0.2
* `PRT` now prints value:
```
PRT value
```

* Use `*` operator to get value of a index.
```
SET 1 233
PRT *1  # Will print `233`
```

# miniasm V0.0.1
Basic syntax and instructions.
```
MEM value
SET index value
CPY index index
ADD index index index
SUB index index index
MUL index index index
DIV index index index
MOD index index index
INC index
DEC index
NEC index
AND index index index
OR index index index
XOR index index index
NOT index
SHL index value
SHR index value
*ROL index value
*ROR index value
EQU index index index
GTER index index index
LESS index index index
JMP value
JIF index value
PRT index
NOP
EXIT value
# comments
```
*: May not implement these.
