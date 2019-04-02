# Build the protobuff processing code. Requires grpc-tools to run. `pip install grpc-tools`

cd ../python/protos

python -m grpc.tools.protoc -I . --python_out=.. --grpc_python_out=.. tensorflow_serving/apis/*.proto
python -m grpc.tools.protoc -I . --python_out=.. --grpc_python_out=.. tensorflow/core/framework/*.proto

cd ../tensorflow
echo `pwd`
touch __init__.py
cd core
touch __init__.py
cd framework
touch __init__.py

cd ../../../tensorflow_serving

echo `pwd`
touch __init__.py
cd apis
touch __init__.py
