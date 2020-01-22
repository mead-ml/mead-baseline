import re
from typing import Pattern
from binascii import hexlify
from functools import partial
from eight_mile.utils import exporter


__all__ = []
export = exporter(__all__)


class MN:
    GZIP = b'1f8b'
    TAR = b'7573746172'
    TAR_START = 257
    ZIP = b'504b0304'


def check_mn(b: bytes, mn: bytes = None, start: int = 0) -> bool:
    if hexlify(b[start:start+20])[:len(mn)] == mn:
        return True
    return False


def int2byte(x: int) -> bytes:
    """Takes an int in the 8-bit range and returns a single-character byte"""
    return bytes((x,))


_text_characters: bytes = (b''.join(int2byte(i) for i in range(32, 127)) + b'\n\r\t\f\b')


# Borrowed from: https://eli.thegreenplace.net/2011/10/19/perls-guess-if-file-is-text-or-binary-implemented-in-python
def is_text_file(block: bytes):
    """ Uses heuristics to guess whether the given file is text or binary,
        by reading a single block of bytes from the file.
        If more than 30% of the chars in the block are non-text, or there
        are NUL ('\x00') bytes in the block, assume this is a binary file.
    """
    if b'\x00' in block:
        # Files with null bytes are binary
        return False
    elif not block:
        # An empty file is considered a valid text file
        return True

    # Use translate's 'deletechars' argument to efficiently remove all
    # occurrences of _text_characters from the block
    nontext = block.translate(None, _text_characters)
    return float(len(nontext)) / len(block) <= 0.30


check_gzip = partial(check_mn, mn=MN.GZIP)
check_tar = partial(check_mn, mn=MN.TAR, start=MN.TAR_START)
check_zip = partial(check_mn, mn=MN.ZIP)


class RE(object):
    HTML = re.compile(b"(<!doctype html>|<html.*?>)")
    BIN = re.compile(b"\d+? \d+?$", re.MULTILINE)


def check_re(b: bytes, regex: Pattern = None) -> bool:
    return True if regex.match(b) else False


check_html = partial(check_re, regex=RE.HTML)


@export
def mime_type(file_name: str) -> str:
    b = open(file_name, 'rb').read(1024)
    if check_gzip(b):
        return "application/gzip"
    if check_tar(b):
        return "application/x-tar"
    if check_zip(b):
        return "application/zip"
    if check_html(b):
        return "text/html"
    if is_text_file(b):
        return "text/plain"
    return "application/w2v"
