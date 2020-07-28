import argparse
from baseline.utils import read_config_stream
from mead.utils import hash_config, convert_path


def main():
    parser = argparse.ArgumentParser(description="Get the mead hash of a config.")
    parser.add_argument('config', help='JSON/YML Configuration for an experiment: local file or remote URL', type=convert_path, default="$MEAD_CONFIG")
    args = parser.parse_args()

    config = read_config_stream(args.config)
    print(hash_config(config))


if __name__ == "__main__":
    main()
