package com.skycloud.controller;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.List;


@Controller
@RequestMapping(value = "/")
public class controller {

    class Param {
        String host;
        Integer port;
        List<Integer> data;
    }

    @RequestMapping(value = "/",
            method = RequestMethod.GET)
    public void index(HttpServletRequest request, HttpServletResponse response) throws IOException {
        response.sendRedirect("view/index.html");
    }

    @RequestMapping(value = "/api/mnist",
            method = RequestMethod.POST)
    public void predict(HttpServletRequest request, HttpServletResponse response,
                        @RequestBody Param param) throws IOException {
        String modelName = "mnist";
        Integer result = doPredict(param.host, param.port, modelName, param.data);
        response.getWriter().print(result);
    }

    public Integer doPredict(String host, int port, String modelName, List<Integer> data) throws IOException {

        // Initialize gRPC client
        ManagedChannel channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext(true).build();
        PredictionServiceGrpc.PredictionServiceBlockingStub blockingStub = PredictionServiceGrpc.newBlockingStub(channel);

        //mnist data
        DataSetIterator mnistTest = new MnistDataSetIterator(1, false, (int)System.currentTimeMillis());

        TensorProto.Builder featuresTensorBuilder = TensorProto.newBuilder();
        for (int i = 0; i < data.size(); ++i)
            featuresTensorBuilder.addFloatVal((255 - data.get(i)) / Float.valueOf(255));

        TensorShapeProto.Dim featuresDim1 = TensorShapeProto.Dim.newBuilder().setSize(1).build();
        TensorShapeProto.Dim featuresDim2 = TensorShapeProto.Dim.newBuilder().setSize(data.size()).build();
        TensorShapeProto featuresShape = TensorShapeProto.newBuilder().addDim(featuresDim1).addDim(featuresDim2).build();
        featuresTensorBuilder.setDtype(org.tensorflow.framework.DataType.DT_FLOAT).setTensorShape(featuresShape);
        TensorProto featuresTensorProto = featuresTensorBuilder.build();

        // Generate gRPC request
        com.google.protobuf.Int64Value version = com.google.protobuf.Int64Value.newBuilder().build();
        Model.ModelSpec modelSpec = Model.ModelSpec.newBuilder().setSignatureName("predict_images").setName(modelName).build();
        Predict.PredictRequest request = Predict.PredictRequest.newBuilder().setModelSpec(modelSpec)
                .putInputs("images", featuresTensorProto).build();

        // Request gRPC server
        try {
            Predict.PredictResponse response = blockingStub.predict(request);
            java.util.Map<java.lang.String, org.tensorflow.framework.TensorProto> outputs = response.getOutputsMap();
            TensorProto tp = outputs.get("scores");

            List<Float> probs = tp.getFloatValList();
            Float maxValue = -Float.MAX_VALUE;
            Integer result = 0;
            for (int i = 0; i < probs.size(); i++) {
                if (probs.get(i) > maxValue) {
                    maxValue = probs.get(i);
                    result = i;
                }
            }
            return result;
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }
}
