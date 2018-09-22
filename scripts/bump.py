import os
import re
import argparse


VERSION_REGEX = re.compile(r'''^__version__ = ['"](?P<major>\d+)\.(?P<minor>\d+).(?P<patch>\d+)(?:dev(?P<dev>\d*))?["']$''')


def parse_version(regex, data):
    return regex.match(data)


def bump_version(data, version):
    match = parse_version(VERSION_REGEX, data)
    if match is None:
        raise ValueError("Version string could not be parsed.")
    current = match.group(version)
    if current is None:
        bumped = 0
    elif current == '':
        bumped = 1
    else:
        bumped = int(current) + 1
    return set_version(match, version, bumped)


def set_version(match, version, value):
    if version == 'major':
        return set_major(match, value)
    elif version == 'minor':
        return set_minor(match, value)
    elif version == 'patch':
        return set_patch(match, value)
    return set_dev(match, value)


def set_major(match, number):
    return "{}.{}.{}".format(number, 0, 0)

def set_minor(match, number):
    return "{}.{}.{}".format(
        match.group('major'),
        number, 0
    )

def set_patch(match, number):
    return "{}.{}.{}".format(
        match.group('major'),
        match.group('minor'),
        number
    )

def set_dev(match, number):
    if number == 0:
        number = ''
    return "{}.{}.{}dev{}".format(
        match.group('major'),
        match.group('minor'),
        match.group('patch'),
        number
    )


def projects_to_file(name):
    loc = os.path.realpath(os.path.dirname(__file__))
    if name == 'baseline':
        return os.path.realpath(os.path.join(loc, '..', 'python', 'baseline', 'version.py'))
    elif name == 'xpctl':
        return os.path.realpath(os.path.join(loc, '..', 'python', 'xpctl', 'version.py'))
    elif name == 'hpctl':
        return os.path.realpath(os.path.join(loc, '..', 'python', 'hpctl', 'hpctl', 'version.py'))
    return name


def main():
    parser = argparse.ArgumentParser(description="Bump versions.")
    parser.add_argument("file")
    parser.add_argument("type", choices={'major', 'minor', 'patch', 'dev'})
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    file_name = projects_to_file(args.file)
    with open(file_name) as f:
        data = f.read().strip("\n")
    new_version = bump_version(data, args.type)
    result = '__version__ = "{}"\n'.format(new_version)
    if args.test:
        print("The found version string:")
        print("\t{}".format(data))
        print("The version string after a {} bump:".format(args.type))
        print("\t{}".format(result))
        return
    with open(file_name, 'w') as f:
        f.write(result)


if __name__ == "__main__":
    main()
