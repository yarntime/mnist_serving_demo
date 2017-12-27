// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/graph.proto

package org.tensorflow.framework;

public interface GraphDefOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.GraphDef)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>repeated .tensorflow.NodeDef node = 1;</code>
   */
  java.util.List<org.tensorflow.framework.NodeDef> 
      getNodeList();
  /**
   * <code>repeated .tensorflow.NodeDef node = 1;</code>
   */
  org.tensorflow.framework.NodeDef getNode(int index);
  /**
   * <code>repeated .tensorflow.NodeDef node = 1;</code>
   */
  int getNodeCount();
  /**
   * <code>repeated .tensorflow.NodeDef node = 1;</code>
   */
  java.util.List<? extends org.tensorflow.framework.NodeDefOrBuilder> 
      getNodeOrBuilderList();
  /**
   * <code>repeated .tensorflow.NodeDef node = 1;</code>
   */
  org.tensorflow.framework.NodeDefOrBuilder getNodeOrBuilder(
      int index);

  /**
   * <pre>
   * Compatibility versions of the graph.  See core/public/version.h for version
   * history.  The GraphDef version is distinct from the TensorFlow version, and
   * each release of TensorFlow will support a range of GraphDef versions.
   * </pre>
   *
   * <code>optional .tensorflow.VersionDef versions = 4;</code>
   */
  boolean hasVersions();
  /**
   * <pre>
   * Compatibility versions of the graph.  See core/public/version.h for version
   * history.  The GraphDef version is distinct from the TensorFlow version, and
   * each release of TensorFlow will support a range of GraphDef versions.
   * </pre>
   *
   * <code>optional .tensorflow.VersionDef versions = 4;</code>
   */
  org.tensorflow.framework.VersionDef getVersions();
  /**
   * <pre>
   * Compatibility versions of the graph.  See core/public/version.h for version
   * history.  The GraphDef version is distinct from the TensorFlow version, and
   * each release of TensorFlow will support a range of GraphDef versions.
   * </pre>
   *
   * <code>optional .tensorflow.VersionDef versions = 4;</code>
   */
  org.tensorflow.framework.VersionDefOrBuilder getVersionsOrBuilder();

  /**
   * <pre>
   * Deprecated single version field; use versions above instead.  Since all
   * GraphDef changes before "versions" was introduced were forward
   * compatible, this field is entirely ignored.
   * </pre>
   *
   * <code>optional int32 version = 3 [deprecated = true];</code>
   */
  @java.lang.Deprecated int getVersion();

  /**
   * <pre>
   * EXPERIMENTAL. DO NOT USE OR DEPEND ON THIS YET.
   * "library" provides user-defined functions.
   * Naming:
   *   * library.function.name are in a flat namespace.
   *     NOTE: We may need to change it to be hierarchical to support
   *     different orgs. E.g.,
   *     { "/google/nn", { ... }},
   *     { "/google/vision", { ... }}
   *     { "/org_foo/module_bar", { ... }}
   *     map&lt;string, FunctionDefLib&gt; named_lib;
   *   * If node[i].op is the name of one function in "library",
   *     node[i] is deemed as a function call. Otherwise, node[i].op
   *     must be a primitive operation supported by the runtime.
   * Function call semantics:
   *   * The callee may start execution as soon as some of its inputs
   *     are ready. The caller may want to use Tuple() mechanism to
   *     ensure all inputs are ready in the same time.
   *   * The consumer of return values may start executing as soon as
   *     the return values the consumer depends on are ready.  The
   *     consumer may want to use Tuple() mechanism to ensure the
   *     consumer does not start until all return values of the callee
   *     function are ready.
   * </pre>
   *
   * <code>optional .tensorflow.FunctionDefLibrary library = 2;</code>
   */
  boolean hasLibrary();
  /**
   * <pre>
   * EXPERIMENTAL. DO NOT USE OR DEPEND ON THIS YET.
   * "library" provides user-defined functions.
   * Naming:
   *   * library.function.name are in a flat namespace.
   *     NOTE: We may need to change it to be hierarchical to support
   *     different orgs. E.g.,
   *     { "/google/nn", { ... }},
   *     { "/google/vision", { ... }}
   *     { "/org_foo/module_bar", { ... }}
   *     map&lt;string, FunctionDefLib&gt; named_lib;
   *   * If node[i].op is the name of one function in "library",
   *     node[i] is deemed as a function call. Otherwise, node[i].op
   *     must be a primitive operation supported by the runtime.
   * Function call semantics:
   *   * The callee may start execution as soon as some of its inputs
   *     are ready. The caller may want to use Tuple() mechanism to
   *     ensure all inputs are ready in the same time.
   *   * The consumer of return values may start executing as soon as
   *     the return values the consumer depends on are ready.  The
   *     consumer may want to use Tuple() mechanism to ensure the
   *     consumer does not start until all return values of the callee
   *     function are ready.
   * </pre>
   *
   * <code>optional .tensorflow.FunctionDefLibrary library = 2;</code>
   */
  org.tensorflow.framework.FunctionDefLibrary getLibrary();
  /**
   * <pre>
   * EXPERIMENTAL. DO NOT USE OR DEPEND ON THIS YET.
   * "library" provides user-defined functions.
   * Naming:
   *   * library.function.name are in a flat namespace.
   *     NOTE: We may need to change it to be hierarchical to support
   *     different orgs. E.g.,
   *     { "/google/nn", { ... }},
   *     { "/google/vision", { ... }}
   *     { "/org_foo/module_bar", { ... }}
   *     map&lt;string, FunctionDefLib&gt; named_lib;
   *   * If node[i].op is the name of one function in "library",
   *     node[i] is deemed as a function call. Otherwise, node[i].op
   *     must be a primitive operation supported by the runtime.
   * Function call semantics:
   *   * The callee may start execution as soon as some of its inputs
   *     are ready. The caller may want to use Tuple() mechanism to
   *     ensure all inputs are ready in the same time.
   *   * The consumer of return values may start executing as soon as
   *     the return values the consumer depends on are ready.  The
   *     consumer may want to use Tuple() mechanism to ensure the
   *     consumer does not start until all return values of the callee
   *     function are ready.
   * </pre>
   *
   * <code>optional .tensorflow.FunctionDefLibrary library = 2;</code>
   */
  org.tensorflow.framework.FunctionDefLibraryOrBuilder getLibraryOrBuilder();
}
