#!/usr/bin/env python3

#
# Copyright 2015 riteme
#

import sys


DEBUG_FLAG = False


def log_info(message):
    print("(info) {0}".format(message))


def log_warning(message):
    print("(warn) {0}".format(message))


def log_error(message):
    print("(error) {0}".format(message))


def log_debug(message):
    if DEBUG_FLAG:
        print("(debug) {0}".format(message))


class FileBuffer(object):

    """Buffer to read chars from a file."""

    def __init__(self, file_path):
        super(FileBuffer, self).__init__()
        self.file_path = file_path
        self._file = open(file_path)
        self._buffer = ""
        self._index = -1
        self._ended = False

    def __del__(self):
        if not self._file.closed:
            self.close()

    def get_next(self):
        """Get the next char in the file stream."""
        assert not self._ended, "File stream met EOF."

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
        assert not self._file.closed, "File have been closed."
        self._file.close()


class Token(object):

    """Tokens means words."""

    UNKNOWN = 0
    KEYWORD = 1
    LITERAL = 2
    NEWLINE = 3
    FINALLY = 4

    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    NUMBERS = "0123456789"

    def __init__(self, type, text):
        super(Token, self).__init__()
        self.type = type
        self.text = text

    def __str__(self):
        return "<Token: type = {0}, text = {1}>".format(
            {
                Token.KEYWORD: "Keyword",
                Token.LITERAL: "Literal",
                Token.NEWLINE: "Newline",
                Token.FINALLY: "EOF"
            }[self.type],
            self.text
        )


class Tokenizer(object):

    """Tokenizer is a DFA to parse a miniasm source."""

    def __init__(self, buffer):
        super(Tokenizer, self).__init__()
        self.buffer = buffer
        self.tokens = []
        self._current = []
        self._mode = Token.KEYWORD
        self._ignore_mode = False

        assert not self.buffer.eof(), "Invalid file buffer"

    def tokenize(self, one_line=False):
        while not self.buffer.eof():
            char = self.buffer.get_next()

            # Met eof
            if char is None:
                break

            if self._ignore_mode:
                continue

            if char == " ":  # Space
                if len(self._current) > 0:
                    self.tokens.append(
                        Token(self._mode, "".join(self._current)))
                    self._current = []

                self._mode = Token.LITERAL

            elif char == "\n":  # Newline
                if len(self._current) > 0:
                    self.tokens.append(
                        Token(self._mode, "".join(self._current)))
                    self._current = []

                # Newline token
                self.tokens.append(Token(Token.NEWLINE, "Newline"))

                self._mode = Token.KEYWORD
                self._ignore_mode = False

            elif char == "#":
                self._ignore_mode = True

            elif char in Token.ALPHABET or char in Token.NUMBERS:
                self._current.append(char)

            else:
                raise RuntimeError("Can't parse char: {}".format(char))

        if self.buffer.eof():
            self.tokens.append(Token(Token.FINALLY, "EOF"))


if __name__ != "__main__":
    raise ImportError("It's not designed as a module to be imported.")

if len(sys.argv) < 2:
    log_error("No input file.")
    exit(-1)

filename = sys.argv[1]
file_buffer = FileBuffer(filename)
tokenizer = Tokenizer(file_buffer)

tokenizer.tokenize()

for token in tokenizer.tokens:
    print(token)
