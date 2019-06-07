import re
from binascii import hexlify
from functools import partial
from baseline.utils import export
import sys
from six import PY3

__all__ = []
exporter = export(__all__)

class MN(object):
    GZIP = b'1f8b'
    TAR = b'7573746172'
    TAR_START = 257
    ZIP = b'504b0304'

def check_mn(b, mn=None, start=0):
    if hexlify(b[start:start+20])[:len(mn)] == mn:
        return True
    return False

# A function that takes an integer in the 8-bit range and returns
# a single-character byte object in py3 / a single-character string
# in py2.
#
int2byte = (lambda x: bytes((x,))) if PY3 else chr

_text_characters = (
        b''.join(int2byte(i) for i in range(32, 127)) +
        b'\n\r\t\f\b')

# Borrowed from: https://eli.thegreenplace.net/2011/10/19/perls-guess-if-file-is-text-or-binary-implemented-in-python
def is_text_file(block):
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

def check_re(b, regex=None):
    return True if regex.match(b) else False

check_html = partial(check_re, regex=RE.HTML)


@exporter
def mime_type(file_name):
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
