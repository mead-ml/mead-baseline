# Simple example C++ programs to run exported pytorch models.

## Build

 * Run `./install.sh`
   * This will download dependencies and compile the two programs.
   * You will need a c++ compiler that supports c++11
 * You can also manually build with:
   * `mkdir -p build && cd build`
   * `cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..`
   * `make`

## Run

 * Each program takes as input the path to the model bundle produced by `mead-export`
   * **Note** These expect the model bundle structure created when exported with `--is_remote false`
 * Remaining command line arguments are interpreted as tokens in the input
 * `tag-text model_bundle token1 token2 ...`

### Featurization gotchas

These are very simple demo models and don't have the most robust featurization code. Namely they assume that all features not named `char` are `Token1D` features that have been lowercased. Any feature named `char` is assumed to be `Char2D` features that are not lowercased. This handles the most common configs we use for the tagger and classifier but it can't handle super complex things.
