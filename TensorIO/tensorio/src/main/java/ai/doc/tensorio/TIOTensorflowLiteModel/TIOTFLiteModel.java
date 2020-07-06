/*
 * TIOTFLiteModel.java
 * TensorIO
 *
 * Created by Philip Dow on 7/6/2020
 * Copyright (c) 2020 - Present doc.ai (http://doc.ai)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.doc.tensorio.TIOTensorflowLiteModel;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.experimental.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.doc.tensorio.TIOLayerInterface.TIOLayerDescription;
import ai.doc.tensorio.TIOLayerInterface.TIOLayerInterface;
import ai.doc.tensorio.TIOLayerInterface.TIOVectorLayerDescription;
import ai.doc.tensorio.TIOModel.TIOModel;
import ai.doc.tensorio.TIOModel.TIOModelBundle;
import ai.doc.tensorio.TIOModel.TIOModelException;
import ai.doc.tensorio.TIOModel.TIOModelIO;

public class TIOTFLiteModel extends TIOModel {

    private Interpreter tflite;
    private MappedByteBuffer tfliteModel;
    private GpuDelegate gpuDelegate = null;

    private int numThreads = 1;
    private boolean useGPU = false;
    private boolean useNNAPI = false;
    private boolean use16bit = false;

    public TIOTFLiteModel(Context context, TIOModelBundle bundle) {
        super(context, bundle);
    }

    //region Lifecycle

    @Override
    public void load() throws TIOModelException {
        try {
            tfliteModel = loadModelFile(getContext(), getBundle().getModelFilePath());
        } catch (IOException e) {
            throw new TIOModelException("Error loading model file", e);
        }
        tflite = new Interpreter(tfliteModel);
        super.load();
    }

    @Override
    public void unload() {
        if (tflite != null) {
            tflite.close();
            this.tflite = null;
        }
        if (this.gpuDelegate != null){
            this.gpuDelegate.close();
            this.gpuDelegate = null;
        }
        super.unload();
    }

    //endRegion

    //region Run

    private Map<String, Object> runMultipleInputMultipleOutput(Map input) throws TIOModelException {
        super.runOn(input);

        // Fetch the input and output layer descriptions from the model

        TIOModelIO.TIOModelIOList inputList = getIO().getInputs();
        TIOModelIO.TIOModelIOList outputList = getIO().getOutputs();

        // Prepare input buffers

        Object[] inputs = new Object[getIO().getInputs().size()];

        for (int i = 0; i < inputList.size(); i++){
            TIOLayerInterface layer = inputList.get(i);
            ByteBuffer inputBuffer = layer.getDataDescription().toByteBuffer(input.get(layer.getName()));
            inputs[i] = inputBuffer;
        }

        // Prepare output buffers

        Map<Integer, Object> outputs = new HashMap<>(getIO().getOutputs().size());

        for (int i = 0; i < outputList.size(); i++){
            TIOLayerInterface layer = outputList.get(i);
            ByteBuffer outputBuffer = layer.getDataDescription().getBackingByteBuffer();
            outputBuffer.rewind();
            outputs.put(i, outputBuffer);
        }

        // Run the model on the input buffers, store the output in the output buffers

        tflite.runForMultipleInputsOutputs(inputs, outputs);

        // Convert output buffers to user land objects

        return captureOutputs(outputs);
    }

    private Map<String, Object> runSingleInputSingleOutput(Object input) throws TIOModelException {
        super.runOn(input);

        // Fetch the input and output layer descriptions from the model

        TIOLayerDescription inputLayer = getIO().getInputs().get(0).getDataDescription();
        TIOLayerDescription outputLayer = getIO().getOutputs().get(0).getDataDescription();

        // Prepare input buffer

        ByteBuffer inputBuffer = inputLayer.toByteBuffer(input);

        // Prepare output buffer

        ByteBuffer outputBuffer = outputLayer.getBackingByteBuffer();
        outputBuffer.rewind();

        // Run the model on the input buffer, store the output in the output buffer

        tflite.run(inputBuffer, outputBuffer);

        // Convert output buffers to user land objects

        Map<Integer, Object> outputs = new HashMap<>(getIO().getOutputs().size()); // Always size 1
        outputs.put(0, outputBuffer);

        return captureOutputs(outputs);
    }

    /**
     * Converts captured ByteBuffers to user land Objects
     * @param outputs The indexed output buffers
     * @return A Map of keys to user land objects capturing the model's outputs
     */

    private Map<String, Object> captureOutputs(Map<Integer, Object> outputs) {
        TIOModelIO.TIOModelIOList outputList = getIO().getOutputs();
        Map<String, Object> outputMap = new HashMap<>(outputList.size());

        for (int i = 0; i < outputList.size(); i++){
            TIOLayerInterface layer = outputList.get(i);
            Object o = layer.getDataDescription().fromByteBuffer((ByteBuffer)outputs.get(i));

            // Perform any additional transformations on the captured output

            if (layer.getDataDescription() instanceof TIOVectorLayerDescription) {
                TIOVectorLayerDescription vectorLayer = (TIOVectorLayerDescription)layer.getDataDescription();

                // If the vector's output is labeled, return a Map of keys to values rather than raw values

                if (vectorLayer.isLabeled()) {
                    o = vectorLayer.labeledValues((float[])o);
                }

                // If the output vector is single valued, return the value directly; See #33 and #34

                // if (((float[])o).length == 1) {
                //    o = ((float[])o)[0];
                // }
            }

            outputMap.put(layer.getName(), o);
        }

        return outputMap;
    }

    @Override
    public Map<String, Object> runOn(Object input) throws TIOModelException{
        super.runOn(input);

        int numInputs = getIO().getInputs().size();
        int numOutputs = getIO().getOutputs().size();

        if (numInputs > 1) {
            Map<String, Object>  output = runMultipleInputMultipleOutput((Map<String, Object>)input);
            return output;
        }
        else {
            if (input instanceof Map) {
                Map<String, Object>  output = runMultipleInputMultipleOutput((Map<String, Object>)input);
                return output;
            }
            else {
                if (numOutputs == 1) {
                    return runSingleInputSingleOutput(input);
                }
                else {
                    Map<String, Object> inputMap = new HashMap<>();
                    TIOLayerInterface inputLayer = getIO().getInputs().get(0);
                    inputMap.put(inputLayer.getName(), input);
                    return runMultipleInputMultipleOutput(inputMap);
                }
            }
        }
    }

    //endRegion

    //region Utilities

    private MappedByteBuffer loadModelFile(Context context, String path) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(path);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void recreateInterpreter() {
        unload();
        Interpreter.Options tfliteOptions = new Interpreter.Options();
        tfliteOptions.setAllowFp16PrecisionForFp32(use16bit);
        tfliteOptions.setUseNNAPI(useNNAPI);
        tfliteOptions.setNumThreads(numThreads);
        if (useGPU && GpuDelegateHelper.isGpuDelegateAvailable()){
            tfliteOptions.addDelegate((GpuDelegate)GpuDelegateHelper.createGpuDelegate());
        }

        tflite = new Interpreter(tfliteModel, tfliteOptions);

    }

    public void useGPU() {
        if (GpuDelegateHelper.isGpuDelegateAvailable()){
            useGPU = true;
            recreateInterpreter();
        }
    }

    public void useCPU() {
        useGPU = false;
        useNNAPI = false;
        recreateInterpreter();
    }

    public void useNNAPI() {
        useGPU = false;
        useNNAPI = true;
        recreateInterpreter();
    }

    public void setNumThreads(int numThreads) {
        this.numThreads = numThreads;
        recreateInterpreter();
    }

    public void setAllow16BitPrecision(boolean use16Bit) {
        this.use16bit = use16Bit;
        recreateInterpreter();
    }

    public long getLastInferenceDuration(){
        return tflite.getLastNativeInferenceDurationNanoseconds();
    }

    public void setOptions(boolean use16bit, boolean useGPU, boolean useNNAPI, int numThreads){
        this.use16bit = use16bit;
        if (useGPU && GpuDelegateHelper.isGpuDelegateAvailable()){
            this.useGPU = true;
            this.useNNAPI = false;
        }
        else if (useNNAPI){
            this.useGPU = false;
            this.useNNAPI = true;
        }
        this.numThreads = numThreads;
        recreateInterpreter();
    }

    //endRegion
}
