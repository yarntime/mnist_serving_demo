// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/util/saved_tensor_slice.proto

package org.tensorflow.util;

public interface SavedSliceMetaOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.SavedSliceMeta)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Name of the tensor.
   * </pre>
   *
   * <code>optional string name = 1;</code>
   */
  java.lang.String getName();
  /**
   * <pre>
   * Name of the tensor.
   * </pre>
   *
   * <code>optional string name = 1;</code>
   */
  com.google.protobuf.ByteString
      getNameBytes();

  /**
   * <pre>
   * Shape of the tensor
   * </pre>
   *
   * <code>optional .tensorflow.TensorShapeProto shape = 2;</code>
   */
  boolean hasShape();
  /**
   * <pre>
   * Shape of the tensor
   * </pre>
   *
   * <code>optional .tensorflow.TensorShapeProto shape = 2;</code>
   */
  org.tensorflow.framework.TensorShapeProto getShape();
  /**
   * <pre>
   * Shape of the tensor
   * </pre>
   *
   * <code>optional .tensorflow.TensorShapeProto shape = 2;</code>
   */
  org.tensorflow.framework.TensorShapeProtoOrBuilder getShapeOrBuilder();

  /**
   * <pre>
   * Type of the tensor
   * </pre>
   *
   * <code>optional .tensorflow.DataType type = 3;</code>
   */
  int getTypeValue();
  /**
   * <pre>
   * Type of the tensor
   * </pre>
   *
   * <code>optional .tensorflow.DataType type = 3;</code>
   */
  org.tensorflow.framework.DataType getType();

  /**
   * <pre>
   * Explicit list of slices saved in the checkpoint file.
   * </pre>
   *
   * <code>repeated .tensorflow.TensorSliceProto slice = 4;</code>
   */
  java.util.List<org.tensorflow.framework.TensorSliceProto> 
      getSliceList();
  /**
   * <pre>
   * Explicit list of slices saved in the checkpoint file.
   * </pre>
   *
   * <code>repeated .tensorflow.TensorSliceProto slice = 4;</code>
   */
  org.tensorflow.framework.TensorSliceProto getSlice(int index);
  /**
   * <pre>
   * Explicit list of slices saved in the checkpoint file.
   * </pre>
   *
   * <code>repeated .tensorflow.TensorSliceProto slice = 4;</code>
   */
  int getSliceCount();
  /**
   * <pre>
   * Explicit list of slices saved in the checkpoint file.
   * </pre>
   *
   * <code>repeated .tensorflow.TensorSliceProto slice = 4;</code>
   */
  java.util.List<? extends org.tensorflow.framework.TensorSliceProtoOrBuilder> 
      getSliceOrBuilderList();
  /**
   * <pre>
   * Explicit list of slices saved in the checkpoint file.
   * </pre>
   *
   * <code>repeated .tensorflow.TensorSliceProto slice = 4;</code>
   */
  org.tensorflow.framework.TensorSliceProtoOrBuilder getSliceOrBuilder(
      int index);
}
