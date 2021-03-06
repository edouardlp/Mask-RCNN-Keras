# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: maskrcnn/results.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='maskrcnn/results.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x16maskrcnn/results.proto\"\x95\x03\n\x06Result\x12$\n\timageInfo\x18\x01 \x01(\x0b\x32\x11.Result.ImageInfo\x12%\n\ndetections\x18\x02 \x03(\x0b\x32\x11.Result.Detection\x1a\x1e\n\x06Origin\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x1a%\n\x04Size\x12\r\n\x05width\x18\x01 \x01(\x01\x12\x0e\n\x06height\x18\x02 \x01(\x01\x1a\x42\n\x04Rect\x12\x1e\n\x06origin\x18\x01 \x01(\x0b\x32\x0e.Result.Origin\x12\x1a\n\x04size\x18\x02 \x01(\x0b\x32\x0c.Result.Size\x1aI\n\tImageInfo\x12\x11\n\tdatasetId\x18\x01 \x01(\t\x12\n\n\x02id\x18\x02 \x01(\t\x12\r\n\x05width\x18\x03 \x01(\x05\x12\x0e\n\x06height\x18\x04 \x01(\x05\x1ah\n\tDetection\x12\x13\n\x0bprobability\x18\x01 \x01(\x01\x12\x0f\n\x07\x63lassId\x18\x02 \x01(\x05\x12\x12\n\nclassLabel\x18\x03 \x01(\t\x12!\n\x0b\x62oundingBox\x18\x04 \x01(\x0b\x32\x0c.Result.Rect\"#\n\x07Results\x12\x18\n\x07results\x18\x01 \x03(\x0b\x32\x07.Resultb\x06proto3')
)




_RESULT_ORIGIN = _descriptor.Descriptor(
  name='Origin',
  full_name='Result.Origin',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='Result.Origin.x', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='Result.Origin.y', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=114,
  serialized_end=144,
)

_RESULT_SIZE = _descriptor.Descriptor(
  name='Size',
  full_name='Result.Size',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='width', full_name='Result.Size.width', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='Result.Size.height', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=146,
  serialized_end=183,
)

_RESULT_RECT = _descriptor.Descriptor(
  name='Rect',
  full_name='Result.Rect',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='origin', full_name='Result.Rect.origin', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='size', full_name='Result.Rect.size', index=1,
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
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=185,
  serialized_end=251,
)

_RESULT_IMAGEINFO = _descriptor.Descriptor(
  name='ImageInfo',
  full_name='Result.ImageInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='datasetId', full_name='Result.ImageInfo.datasetId', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='id', full_name='Result.ImageInfo.id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='Result.ImageInfo.width', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='Result.ImageInfo.height', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=253,
  serialized_end=326,
)

_RESULT_DETECTION = _descriptor.Descriptor(
  name='Detection',
  full_name='Result.Detection',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='probability', full_name='Result.Detection.probability', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='classId', full_name='Result.Detection.classId', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='classLabel', full_name='Result.Detection.classLabel', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='boundingBox', full_name='Result.Detection.boundingBox', index=3,
      number=4, type=11, cpp_type=10, label=1,
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
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=328,
  serialized_end=432,
)

_RESULT = _descriptor.Descriptor(
  name='Result',
  full_name='Result',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='imageInfo', full_name='Result.imageInfo', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='detections', full_name='Result.detections', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_RESULT_ORIGIN, _RESULT_SIZE, _RESULT_RECT, _RESULT_IMAGEINFO, _RESULT_DETECTION, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=27,
  serialized_end=432,
)


_RESULTS = _descriptor.Descriptor(
  name='Results',
  full_name='Results',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='results', full_name='Results.results', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=434,
  serialized_end=469,
)

_RESULT_ORIGIN.containing_type = _RESULT
_RESULT_SIZE.containing_type = _RESULT
_RESULT_RECT.fields_by_name['origin'].message_type = _RESULT_ORIGIN
_RESULT_RECT.fields_by_name['size'].message_type = _RESULT_SIZE
_RESULT_RECT.containing_type = _RESULT
_RESULT_IMAGEINFO.containing_type = _RESULT
_RESULT_DETECTION.fields_by_name['boundingBox'].message_type = _RESULT_RECT
_RESULT_DETECTION.containing_type = _RESULT
_RESULT.fields_by_name['imageInfo'].message_type = _RESULT_IMAGEINFO
_RESULT.fields_by_name['detections'].message_type = _RESULT_DETECTION
_RESULTS.fields_by_name['results'].message_type = _RESULT
DESCRIPTOR.message_types_by_name['Result'] = _RESULT
DESCRIPTOR.message_types_by_name['Results'] = _RESULTS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Result = _reflection.GeneratedProtocolMessageType('Result', (_message.Message,), dict(

  Origin = _reflection.GeneratedProtocolMessageType('Origin', (_message.Message,), dict(
    DESCRIPTOR = _RESULT_ORIGIN,
    __module__ = 'maskrcnn.results_pb2'
    # @@protoc_insertion_point(class_scope:Result.Origin)
    ))
  ,

  Size = _reflection.GeneratedProtocolMessageType('Size', (_message.Message,), dict(
    DESCRIPTOR = _RESULT_SIZE,
    __module__ = 'maskrcnn.results_pb2'
    # @@protoc_insertion_point(class_scope:Result.Size)
    ))
  ,

  Rect = _reflection.GeneratedProtocolMessageType('Rect', (_message.Message,), dict(
    DESCRIPTOR = _RESULT_RECT,
    __module__ = 'maskrcnn.results_pb2'
    # @@protoc_insertion_point(class_scope:Result.Rect)
    ))
  ,

  ImageInfo = _reflection.GeneratedProtocolMessageType('ImageInfo', (_message.Message,), dict(
    DESCRIPTOR = _RESULT_IMAGEINFO,
    __module__ = 'maskrcnn.results_pb2'
    # @@protoc_insertion_point(class_scope:Result.ImageInfo)
    ))
  ,

  Detection = _reflection.GeneratedProtocolMessageType('Detection', (_message.Message,), dict(
    DESCRIPTOR = _RESULT_DETECTION,
    __module__ = 'maskrcnn.results_pb2'
    # @@protoc_insertion_point(class_scope:Result.Detection)
    ))
  ,
  DESCRIPTOR = _RESULT,
  __module__ = 'maskrcnn.results_pb2'
  # @@protoc_insertion_point(class_scope:Result)
  ))
_sym_db.RegisterMessage(Result)
_sym_db.RegisterMessage(Result.Origin)
_sym_db.RegisterMessage(Result.Size)
_sym_db.RegisterMessage(Result.Rect)
_sym_db.RegisterMessage(Result.ImageInfo)
_sym_db.RegisterMessage(Result.Detection)

Results = _reflection.GeneratedProtocolMessageType('Results', (_message.Message,), dict(
  DESCRIPTOR = _RESULTS,
  __module__ = 'maskrcnn.results_pb2'
  # @@protoc_insertion_point(class_scope:Results)
  ))
_sym_db.RegisterMessage(Results)


# @@protoc_insertion_point(module_scope)
