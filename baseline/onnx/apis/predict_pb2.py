# SPDX-License-Identifier: Apache-2.0

# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: predict.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import baseline.onnx.apis.onnx_ml_pb2 as onnx__ml__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='predict.proto',
  package='onnxruntime.server',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\rpredict.proto\x12\x12onnxruntime.server\x1a\ronnx-ml.proto\"\xaf\x01\n\x0ePredictRequest\x12>\n\x06inputs\x18\x02 \x03(\x0b\x32..onnxruntime.server.PredictRequest.InputsEntry\x12\x15\n\routput_filter\x18\x03 \x03(\t\x1a@\n\x0bInputsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12 \n\x05value\x18\x02 \x01(\x0b\x32\x11.onnx.TensorProto:\x02\x38\x01J\x04\x08\x01\x10\x02\"\x97\x01\n\x0fPredictResponse\x12\x41\n\x07outputs\x18\x01 \x03(\x0b\x32\x30.onnxruntime.server.PredictResponse.OutputsEntry\x1a\x41\n\x0cOutputsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12 \n\x05value\x18\x02 \x01(\x0b\x32\x11.onnx.TensorProto:\x02\x38\x01\x62\x06proto3')
  ,
  dependencies=[onnx__ml__pb2.DESCRIPTOR,])




_PREDICTREQUEST_INPUTSENTRY = _descriptor.Descriptor(
  name='InputsEntry',
  full_name='onnxruntime.server.PredictRequest.InputsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='onnxruntime.server.PredictRequest.InputsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='onnxruntime.server.PredictRequest.InputsEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=158,
  serialized_end=222,
)

_PREDICTREQUEST = _descriptor.Descriptor(
  name='PredictRequest',
  full_name='onnxruntime.server.PredictRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='inputs', full_name='onnxruntime.server.PredictRequest.inputs', index=0,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_filter', full_name='onnxruntime.server.PredictRequest.output_filter', index=1,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_PREDICTREQUEST_INPUTSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=53,
  serialized_end=228,
)


_PREDICTRESPONSE_OUTPUTSENTRY = _descriptor.Descriptor(
  name='OutputsEntry',
  full_name='onnxruntime.server.PredictResponse.OutputsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='onnxruntime.server.PredictResponse.OutputsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='onnxruntime.server.PredictResponse.OutputsEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=317,
  serialized_end=382,
)

_PREDICTRESPONSE = _descriptor.Descriptor(
  name='PredictResponse',
  full_name='onnxruntime.server.PredictResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='outputs', full_name='onnxruntime.server.PredictResponse.outputs', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_PREDICTRESPONSE_OUTPUTSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=231,
  serialized_end=382,
)

_PREDICTREQUEST_INPUTSENTRY.fields_by_name['value'].message_type = onnx__ml__pb2._TENSORPROTO
_PREDICTREQUEST_INPUTSENTRY.containing_type = _PREDICTREQUEST
_PREDICTREQUEST.fields_by_name['inputs'].message_type = _PREDICTREQUEST_INPUTSENTRY
_PREDICTRESPONSE_OUTPUTSENTRY.fields_by_name['value'].message_type = onnx__ml__pb2._TENSORPROTO
_PREDICTRESPONSE_OUTPUTSENTRY.containing_type = _PREDICTRESPONSE
_PREDICTRESPONSE.fields_by_name['outputs'].message_type = _PREDICTRESPONSE_OUTPUTSENTRY
DESCRIPTOR.message_types_by_name['PredictRequest'] = _PREDICTREQUEST
DESCRIPTOR.message_types_by_name['PredictResponse'] = _PREDICTRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PredictRequest = _reflection.GeneratedProtocolMessageType('PredictRequest', (_message.Message,), dict(

  InputsEntry = _reflection.GeneratedProtocolMessageType('InputsEntry', (_message.Message,), dict(
    DESCRIPTOR = _PREDICTREQUEST_INPUTSENTRY,
    __module__ = 'predict_pb2'
    # @@protoc_insertion_point(class_scope:onnxruntime.server.PredictRequest.InputsEntry)
    ))
  ,
  DESCRIPTOR = _PREDICTREQUEST,
  __module__ = 'predict_pb2'
  # @@protoc_insertion_point(class_scope:onnxruntime.server.PredictRequest)
  ))
_sym_db.RegisterMessage(PredictRequest)
_sym_db.RegisterMessage(PredictRequest.InputsEntry)

PredictResponse = _reflection.GeneratedProtocolMessageType('PredictResponse', (_message.Message,), dict(

  OutputsEntry = _reflection.GeneratedProtocolMessageType('OutputsEntry', (_message.Message,), dict(
    DESCRIPTOR = _PREDICTRESPONSE_OUTPUTSENTRY,
    __module__ = 'predict_pb2'
    # @@protoc_insertion_point(class_scope:onnxruntime.server.PredictResponse.OutputsEntry)
    ))
  ,
  DESCRIPTOR = _PREDICTRESPONSE,
  __module__ = 'predict_pb2'
  # @@protoc_insertion_point(class_scope:onnxruntime.server.PredictResponse)
  ))
_sym_db.RegisterMessage(PredictResponse)
_sym_db.RegisterMessage(PredictResponse.OutputsEntry)


_PREDICTREQUEST_INPUTSENTRY._options = None
_PREDICTRESPONSE_OUTPUTSENTRY._options = None
# @@protoc_insertion_point(module_scope)
