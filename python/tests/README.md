# Writing tests

## Versions

Support python 2 and 3 in your tests. This means using the `mock` backport library to import `MagicMock` or `patch` rather than `unittest.mock`


## Frameworks

There are a lot of optional dependencies that baseline has (various deep learning frameworks, pyyaml, etc). Then writing tests that depend on these libraries use `import_name = pytest.importorskip('import_name')` so that these tests will be skipped if the dependency is not installed.


## Accessing test data

When accessing test data on disk make sure the paths are based on the location of the test file rather than relative paths based on the current working directory. This lets pytest be run from anywhere.

```python
import os

file_loc = os.path.realpath(os.path.dirname(__file__))
data_loc = os.path.join(file_loc, 'test_data')

file_path = os.path.join(data_loc, 'file_name')
```
