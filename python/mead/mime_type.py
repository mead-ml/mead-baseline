import re
from binascii import hexlify
from functools import partial
from baseline.utils import export

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

check_gzip = partial(check_mn, mn=MN.GZIP)
check_tar = partial(check_mn, mn=MN.TAR, start=MN.TAR_START)
check_zip = partial(check_mn, mn=MN.ZIP)

class RE(object):
    HTML = re.compile(b"(<!doctype html>|<html.*?>)")
    BIN = re.compile(b"\d+? \d+?$", re.MULTILINE)

def check_re(b, regex=None):
    return True if regex.match(b) else False

check_html = partial(check_re, regex=RE.HTML)
check_bin = partial(check_re, regex=RE.BIN)

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
    if check_bin(b):
        return "application/w2v"
    return "text/plain"
