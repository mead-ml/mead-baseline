import random
import string

chars = list(string.ascii_letters + string.digits)

def r_str(length=None, min_=5, max_=20):
    if length is None:
        length = random.randint(min_, max_)
    return ''.join([random.choice(chars) for _ in range(length)])
