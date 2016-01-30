#!/usr/bin/env python3

#
# Copyright 2015 riteme
#

import sys
import copy
import readline


DEBUG_FLAG = False
NO_MEMORY_DUMP = False
NO_JUMP_DUMP = False
NO_EXECUTE_DUMP = False
NO_OUTPUT = False


def log_info(message):
    if not NO_OUTPUT:
        print("(info) {0}".format(message))


def log_warning(message):
    if not NO_OUTPUT:
        print("(warn) {0}".format(message))


def log_error(message):
    if not NO_OUTPUT:
        print("(error) {0}".format(message))


def log_debug(message):
    if DEBUG_FLAG and not NO_OUTPUT:
        print("(debug) {0}".format(message))


def force_info(message):
    print("(info) {0}".format(message))


def force_warning(message):
    print("(warn) {0}".format(message))


def force_error(message):
    print("(error) {0}".format(message))


def force_debug(message):
    print("(debug) {0}".format(message))


class BufferObject(object):

    """BufferObject provides interfaces to read data in files or string."""

    def __init__(self):
        super(BufferObject, self).__init__()

    def __iter__(self):
        return self

    def __next__(self):
        if self.eof():
            raise StopIteration
        else:
            result = self.get_next()

            if result is None:
                raise StopIteration
            else:
                return result

    def get_next(self):
        raise NotImplementedError(
            "This function is overloaded in derived classes")

    def eof(self):
        raise NotImplementedError(
            "This function is overloaded in derived classes")


class FileBuffer(BufferObject):

    """Buffer to read chars from a file."""

    def __init__(self, file_path):
        super(FileBuffer, self).__init__()
        self.file_path = file_path
        self._file = open(file_path)
        self._buffer = ""
        self._index = -1
        self._ended = False

    def __del__(self):
        if hasattr(self, "_file") and not self._file.closed:
            self.close()

    def get_next(self):
        """Get the next char in the file stream."""
        assert not self._ended, "File stream met EOF"

        if self._index == len(self._buffer) - 1:
            self._buffer = self._file.readline()
            self._index = -1

            if not self._buffer:
                self._ended = True
                return None

        self._index += 1
        return self._buffer[self._index]

    def eof(self):
        return self._ended

    def close(self):
        """Close current file object."""
        assert not self._file.closed, "File have been closed"
        self._file.close()


class Token(object):

    """Tokens means words."""

    UNKNOWN = 0
    KEYWORD = 1
    LITERAL = 2
    OPERATOR = 3
    NEWLINE = 4
    FINALLY = 5

    TOKEN_NAME = {
        KEYWORD: "Keyword",
        LITERAL: "Literal",
        NEWLINE: "Newline",
        OPERATOR: "Operator",
        FINALLY: "EOF"
    }

    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    NUMBERS = ".+-0123456789"
    OPERATORS = "*"
    SPACE_CHAR = " "
    COMMENT_CHAR = "#"
    NEWLINE_CHAR = "\n"

    def __init__(self, type, text):
        super(Token, self).__init__()
        self.type = type
        self.text = text

    def __str__(self):
        return self.text

    def __repr__(self):
        return "<Token: type = {0}, text = {1}>".format(
            self.TOKEN_NAME[self.type],
            self.text
        )


class TokenStream(object):

    """TokenStream contains a list of tokens."""

    def __init__(self):
        super(TokenStream, self).__init__()
        self.tokens = []

    def __iter__(self):
        return iter(self.tokens)

    def append(self, token_type, token_text):
        self.tokens.append(Token(token_type, token_text))

    def pop(self):
        self.tokens.pop()

    def last(self):
        return self.tokens[-1]


class Tokenizer(object):

    """Tokenizer is a DFA to parse a miniasm source."""

    def __init__(self, buffer):
        super(Tokenizer, self).__init__()
        self.buffer = buffer
        self.tokens = TokenStream()
        self._current = []
        self._mode = Token.KEYWORD
        self._ignore_mode = False

        assert not self.buffer.eof(), "Invalid file buffer"

    def tokenize(self):
        assert not self.buffer.eof(), "Buffer met EOF"

        for char in self.buffer:
            if char in Token.NEWLINE_CHAR:  # Newline
                if len(self._current) > 0:
                    self.tokens.append(self._mode, "".join(self._current))
                    self._current = []

                # Insert NOP for empty lines
                if self.tokens.last().type == Token.NEWLINE:
                    self.tokens.append(Token.KEYWORD, "NOP")

                # Newline token
                self.tokens.append(Token.NEWLINE, "Newline")

                self._mode = Token.KEYWORD
                self._ignore_mode = False

            elif char in Token.SPACE_CHAR:  # Space
                if len(self._current) > 0:
                    self.tokens.append(self._mode, "".join(self._current))
                    self._current = []

                self._mode = Token.LITERAL

            elif char in Token.COMMENT_CHAR:
                self._ignore_mode = True

            elif self._ignore_mode:
                continue

            elif char in Token.OPERATORS:
                self._mode = Token.OPERATOR
                self._current.append(char)

            elif self._mode == Token.OPERATOR and char not in Token.OPERATORS:
                if len(self._current) > 0:
                    self.tokens.append(Token.OPERATOR, "".join(self._current))
                    self._current = []
                    self._mode = Token.LITERAL

                self._current.append(char)

            elif char in Token.ALPHABET or char in Token.NUMBERS:
                self._current.append(char)

            else:
                raise RuntimeError(
                    "Can't parse char: {0} (\"\{1}\")".format(char, ord(char)))

        if self.buffer.eof():
            self.tokens.append(Token.FINALLY, "EOF")


class Statement(object):

    """Statement means a line of code."""

    def __init__(self, function, args):
        super(Statement, self).__init__()
        self.function = function
        self.args = args

    def __str__(self):
        return "<Statement: function: {0}, args: {1}>".format(
            self.function, self.args
        )


class Syntactic(object):

    """Syntax alalysis."""

    def __init__(self, tokens):
        super(Syntactic, self).__init__()
        self.tokens = tokens
        self.statements = []

    def analyze(self):
        command = None
        args = []
        operator = []

        for token in self.tokens:
            if token.type == Token.FINALLY:
                break

            elif token.type == Token.NEWLINE:
                if command is not None:
                    self.statements.append(Statement(command, args))
                    command = None
                    args = []
                    operator = []

            elif command is None:
                command = token

            else:
                if token.type == Token.OPERATOR:
                    operator.append(token)
                else:
                    operator.append(token)
                    args.append(operator)
                    operator = []


class MemoryPool(object):

    """MemoryPool manages memory."""

    def __init__(self):
        super(MemoryPool, self).__init__()
        self.memory = []

    def resize(self, size):
        if not NO_MEMORY_DUMP:
            log_debug("memory resize: {0}".format(size))

        distance = size - len(self.memory) + 1

        if distance < 0:
            while distance < 0:
                self.memory.pop()

                distance += 1
        elif distance > 0:
            while distance > 0:
                self.memory.append(int())

                distance -= 1

        self.memory[0] = size

    def memget(self, index):
        if index < 0 or index >= len(self.memory):
            raise IndexError("Memory out of range.")

        if not NO_MEMORY_DUMP:
            log_debug("memory get: {0} = {1}".format(
                index, self.memory[index]
            ))

        return self.memory[index]

    def memset(self, index, data):
        if index < 0 or index >= len(self.memory):
            raise IndexError("Memory out of range.")

        if not NO_MEMORY_DUMP:
            log_debug("memory set: {0} = {1}".format(
                index, data
            ))

        self.memory[index] = data


class Program(object):

    """Run it!"""

    NOT_STARTED = 0
    RUNNING = 1
    EXITED = 2

    PROMPT_STRING = "> "

    def dereference(self, args):
        """Dereference parameters (*index syntax)"""
        if len(args) == 1:
            return int(args[0].text)

        result = int(args[-1].text)
        for i in range(0, len(args[0].text)):
            result = self.memory.memget(
                result
            )

        return result

    def MEM(self, value):
        self.memory.resize(value)

    def SET(self, index, value):
        self.memory.memset(index, value)

    def CPY(self, index1, index2):
        self.memory.memset(index2, self.memory.memget(index1))

    def ADD(self, index1, index2, index3):
        self.memory.memset(
            index3,
            self.memory.memget(index1) + self.memory.memget(index2)
        )

    def SUB(self, index1, index2, index3):
        self.memory.memset(
            index3,
            self.memory.memget(index1) - self.memory.memget(index2)
        )

    def MUL(self, index1, index2, index3):
        self.memory.memset(
            index3,
            self.memory.memget(index1) * self.memory.memget(index2)
        )

    def DIV(self, index1, index2, index3):
        self.memory.memset(
            index3,
            self.memory.memget(index1) / self.memory.memget(index2)
        )

    def MOD(self, index1, index2, index3):
        self.memory.memset(
            index3,
            self.memory.memget(index1) % self.memory.memget(index2)
        )

    def INC(self, index):
        self.memory.memset(
            index,
            self.memory.memget(index) + 1
        )

    def DEC(self, index):
        self.memory.memset(
            index,
            self.memory.memget(index) - 1
        )

    def NEC(self, index):
        self.memory.memset(
            index,
            -self.memory.memget(index)
        )

    def AND(self, index1, index2, index3):
        self.memory.memset(
            index3,
            self.memory.memget(index1) & self.memory.memget(index2)
        )

    def OR(self, index1, index2, index3):
        self.memory.memset(
            index3,
            self.memory.memget(index1) | self.memory.memget(index2)
        )

    def XOR(self, index1, index2, index3):
        self.memory.memset(
            index3,
            self.memory.memget(index1) ^ self.memory.memget(index2)
        )

    def NOT(self, index):
        self.memory.memset(
            index,
            int(not self.memory.memget(index))
        )

    def SHL(self, index, value):
        self.memory.memset(
            index,
            self.memory.memget(index) << value
        )

    def SHR(self, index, value):
        self.memory.memset(
            index,
            self.memory.memget(index) >> value
        )

    def EQU(self, index1, index2, index3):
        self.memory.memset(
            index3,
            int(self.memory.memget(index1) == self.memory.memget(index2))
        )

    def GTER(self, index1, index2, index3):
        self.memory.memset(
            index3,
            int(self.memory.memget(index1) > self.memory.memget(index2))
        )

    def LESS(self, index1, index2, index3):
        self.memory.memset(
            index3,
            int(self.memory.memget(index1) < self.memory.memget(index2))
        )

    def JMP(self, value):
        self.position = value

        if not NO_JUMP_DUMP:
            log_debug("jump: {}".format(self.position + 1))

    def JIF(self, index, value):
        if self.memory.memget(index):
            self.JMP(value)

    def PRT(self, value):
        print(value)

    def NOP(self, index=None):
        if not index is None:
            self.memory.memset(index, self.position - 1)

    def EXIT(self, value):
        self.status = Program.EXITED
        self.exitcode = value

    def READ(self, index):
        value = int(input(Program.PROMPT_STRING))
        self.memory.memset(index, value)

    def LEQ(self, index1, index2, index3):
        self.memory.memset(
            index3,
            int(self.memory.memget(index1) <= self.memory.memget(index2))
        )

    def GEQ(self, index1, index2, index3):
        self.memory.memset(
            index3,
            int(self.memory.memget(index1) >= self.memory.memget(index2))
        )

    def JMOV(self, value):
        self.position += value - 1

        if not NO_JUMP_DUMP:
            log_debug("jump: {}".format(self.position + 1))

    def JIFM(self, index, value):
        if self.memory.memget(index):
            self.JMOV(value)

    FUNCTIONS = [
        MEM, SET, CPY, ADD, SUB, MUL,
        DIV, MOD, INC, DEC, NEC, AND,
        OR,  XOR, NOT, SHL, SHR, EQU,
        GTER, LESS, JMP, JIF, PRT, NOP,
        EXIT, READ, LEQ, GEQ, JMOV, JIFM
    ]

    FUNCTION_MAP = {}

    def __init__(self, statements):
        super(Program, self).__init__()
        self.statements = statements
        self.memory = MemoryPool()
        self.status = Program.NOT_STARTED
        self.exitcode = 0
        self.position = 0

        for function in Program.FUNCTIONS:
            Program.FUNCTION_MAP[function.__name__] = function

    def exit(self, exitcode=0):
        self.exitcode = exitcode
        self.status = Program.EXITED

    def execute(self):
        self.status = Program.RUNNING

        while (
            self.status == Program.RUNNING and
            self.position < len(self.statements)
        ):
            statement = copy.deepcopy(
                self.statements[self.position]
            )

            self.position += 1

            for i in range(0, len(statement.args)):
                try:
                    statement.args[i] = self.dereference(
                        statement.args[i]
                    )
                except IndexError:
                    raise RuntimeError(
                        "At line {0}: Can't dereference parameter.".format(
                            self.position
                        ))

            if not NO_EXECUTE_DUMP:
                log_debug("{0}: {1} {2}".format(
                    self.position,
                    statement.function.text,
                    statement.args
                ))

            assert self.status == Program.RUNNING
            Program.FUNCTION_MAP[statement.function.text.upper()](
                self, *statement.args
            )

        return self.exitcode


FLAG_NO_DEBUG = "--no-debug"
FLAG_DEBUG = "--debug"
FLAG_NO_MEMORY_DUMP = "--no-memory-dump"
FLAG_NO_JUMP_DUMP = "--no-jump-dump"
FLAG_NO_EXECUTE_DUMP = "--no-execute-dump"
FLAG_NO_OUTPUT = "--no-output"

if __name__ == "__main__":
    def parse_args():
        global DEBUG_FLAG
        global NO_MEMORY_DUMP
        global NO_JUMP_DUMP
        global NO_EXECUTE_DUMP
        global NO_OUTPUT

        filename = ""

        for arg in sys.argv[1:]:
            if arg.startswith("-"):
                if arg == FLAG_NO_OUTPUT:
                    NO_OUTPUT = True

                elif arg == FLAG_NO_DEBUG:
                    DEBUG_FLAG = False

                elif arg == FLAG_DEBUG:
                    DEBUG_FLAG = True

                elif arg == FLAG_NO_MEMORY_DUMP:
                    NO_MEMORY_DUMP = True

                elif arg == FLAG_NO_JUMP_DUMP:
                    NO_JUMP_DUMP = True

                elif arg == FLAG_NO_EXECUTE_DUMP:
                    NO_EXECUTE_DUMP = True

                else:
                    force_warning("Unknown parameter: {0}".format(arg))

            else:
                filename = arg

        return filename

    if len(sys.argv) < 2:
        log_error("No input file.")
        exit(-1)

    if DEBUG_FLAG:
        log_info("Debug mode is on.")

    readline.parse_and_bind("tab: complete")

    filename = parse_args()

    try:
        file_buffer = FileBuffer(filename)
    except:
        log_error("Can't open file.")
        exit(-1)

    tokenizer = Tokenizer(file_buffer)
    tokenizer.tokenize()

    syntactic = Syntactic(tokenizer.tokens)
    syntactic.analyze()

    program = Program(syntactic.statements)
    try:
        program.execute()
    except IndexError as exception:
        log_error("At line {0}: {1}.".format(
            program.position,
            exception.args[0]
        ))

    except KeyError as exception:
        log_error("At line {0}: Unknown keyword {1}.".format(
            program.position,
            exception.args[0]
        ))

    except AssertionError:
        log_warning("Program exited.")

    except Exception as exception:
        log_error("An exception occured while running.\n{}".format(
            exception
        ))

    status = program.exitcode

    if status != 0:
        log_warning("Program exited unnormally.")
