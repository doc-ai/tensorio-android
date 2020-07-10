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

package ai.doc.tensorio.TIOTFLiteModel;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.experimental.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import ai.doc.tensorio.TIOLayerInterface.TIOLayerInterface;
import ai.doc.tensorio.TIOModel.TIOModel;
import ai.doc.tensorio.TIOModel.TIOModelBundle;
import ai.doc.tensorio.TIOModel.TIOModelException;
import ai.doc.tensorio.TIOModel.TIOModelIO;
import ai.doc.tensorio.TIOTFLiteData.TIOTFLitePixelDataConverter;
import ai.doc.tensorio.TIOTFLiteData.TIOTFLiteVectorDataConverter;

public class TIOTFLiteModel extends TIOModel {

    // TFLite Backend

    private Interpreter interpreter;
    private MappedByteBuffer tfliteModel;
    private GpuDelegate gpuDelegate = null;

    // TFLite Backend Options

    private int numThreads = 1;
    private boolean useGPU = false;
    private boolean useNNAPI = false;
    private boolean use16bit = false;

    // Buffer Caching

    private boolean cacheBuffers = true;
    private Map<TIOLayerInterface, ByteBuffer> bufferCache = null;

    // Data Converters

    final private TIOTFLiteVectorDataConverter vectorDataConverter = new TIOTFLiteVectorDataConverter();
    final private TIOTFLitePixelDataConverter pixelDataConverter = new TIOTFLitePixelDataConverter();

    // Constructor

    public TIOTFLiteModel(Context context, TIOModelBundle bundle) {
        super(context, bundle);
    }

    //region Lifecycle

    @Override
    public void load() throws TIOModelException {
        if (isLoaded()) {
            return;
        }

        // Prepare Model

        try {
            tfliteModel = loadModelFile(getContext(), getBundle().getModelFilePath());
        } catch (IOException e) {
            throw new TIOModelException("Error loading model file", e);
        }

        interpreter = new Interpreter(tfliteModel);

        // Prepare Buffer Cache

        if (cacheBuffers) {
            prepareBufferCache();
        }

        super.load();
    }

    @Override
    public void unload() {
        if (!isLoaded()) {
            return;
        }

        if (interpreter != null ) {
            interpreter.close();
            this.interpreter = null;
        }

        if (this.gpuDelegate != null) {
            this.gpuDelegate.close();
            this.gpuDelegate = null;
        }

        if (this.bufferCache != null) {
            this.bufferCache = null;
        }

        super.unload();
    }

    /**
     * Create buffer caches that are used for model inputs and outputs
     */

    void prepareBufferCache() {
        bufferCache = new HashMap<>();

        List<TIOLayerInterface> layers = new ArrayList<>();
        layers.addAll(getIO().getInputs().all());
        layers.addAll(getIO().getOutputs().all());

        for (TIOLayerInterface layer : layers) {
            layer.doCase((vectorLayer) -> {
                bufferCache.put(layer, vectorDataConverter.createBackingBuffer(vectorLayer));
            }, (pixelLayer) -> {
                bufferCache.put(layer, pixelDataConverter.createBackingBuffer(pixelLayer));
            });
        }
    }

    //endRegion

    //region Run

    @Override
    public Map<String, Object> runOn(float[] input) throws TIOModelException, IllegalArgumentException {
        validateInput(input);
        load();

        if (hasMultipleInputsOrOutputs()) {
            return runMultipleInputMultipleOutput(mappedInput(input));
        } else {
            return runSingleInputSingleOutput(input);
        }
    }

    @Override
    public Map<String, Object> runOn(byte[] input) throws TIOModelException, IllegalArgumentException {
        validateInput(input);
        load();

        if (hasMultipleInputsOrOutputs()) {
            return runMultipleInputMultipleOutput(mappedInput(input));
        } else {
            return runSingleInputSingleOutput(input);
        }
    }

    @Override
    public Map<String, Object> runOn(Bitmap input) throws TIOModelException, IllegalArgumentException {
        validateInput(input);
        load();

        if (hasMultipleInputsOrOutputs()) {
            return runMultipleInputMultipleOutput(mappedInput(input));
        } else {
            return runSingleInputSingleOutput(input);
        }
    }

    @Override
    public Map<String, Object> runOn(Map<String, Object> input) throws TIOModelException, IllegalArgumentException {
        validateInput(input);
        load();

        if (hasMultipleInputsOrOutputs()) {
            return runMultipleInputMultipleOutput(input);
        } else {
            return runSingleInputSingleOutput(unmappedInput(input));
        }
    }

    /**
     * Used to determined if an unmapped input should be mapped and a mapped input unmapped
     * @return true if models has either of more than one input or output, false otherwise
     */

    private boolean hasMultipleInputsOrOutputs() {
        return getIO().getInputs().size() > 0 || getIO().getOutputs().size() > 0;
    }

    /**
     * Converts a single input to a mapped input using the single input layer's name.
     *
     * Used to convert a single input to a map when a model has multiple outputs.
     * Check that the model only has a single input before calling this method.
     *
     * @param input One of the supported input types, e.g. byte[], float[], or Bitmap
     * @return A map from the input layer's name to the input
     */

    private Map<String, Object> mappedInput(Object input) {
        Map<String, Object> map = new HashMap<>();
        map.put(getIO().getInputs().get(0).getName(), input);
        return map;
    }

    /**
     * Extracts the value from a mapped input using the single input layer's name.
     *
     * Used to convert a mapped input to its raw value when a model has a single output.
     * Check that the model only has a single input before calling this method.
     *
     * @param input A mapping from an input layer's name to a value
     * @return The value in the map
     */

    private Object unmappedInput(Map<String, Object> input) {
        return input.get(getIO().getInputs().get(0).getName());
    }

    /**
     * Actually performs inference on a single input with a single output
     * @param input An input in one of the supported types, e.g. byte[], float[], or Bitmap
     * @return The model's single output mapped by the output layers name
     * @throws IllegalArgumentException raised if the input cannot be transformed to the format
     *                                  expected by the model
     */

    private Map<String, Object> runSingleInputSingleOutput(Object input) throws IllegalArgumentException {

        // Fetch the input and output layer descriptions from the model

        TIOLayerInterface inputLayer = getIO().getInputs().get(0);
        TIOLayerInterface outputLayer = getIO().getOutputs().get(0);

        // Prepare input buffer

        ByteBuffer inputBuffer = prepareInputBuffer(input, inputLayer);

        // Prepare output buffer

        ByteBuffer outputBuffer = prepareOutputBuffer(outputLayer);

        // Run the model on the input buffer, store the output in the output buffer

        interpreter.run(inputBuffer, outputBuffer);

        // Convert output buffers to user land objects

        Map<Integer, Object> outputs = new HashMap<>(getIO().getOutputs().size()); // Always size 1
        outputs.put(0, outputBuffer);

        return captureOutputs(outputs);
    }

    /**
     * Actually performs inference on multiple inputs or multiple outputs
     * @param inputs A mapping from input layer names to input values
     * @return The model's outputs mapped by the output layer names
     * @throws IllegalArgumentException raised if the input cannot be transformed to the format
     *                                  expected by the model
     */

    private Map<String, Object> runMultipleInputMultipleOutput(Map<String, Object> inputs) throws IllegalArgumentException {

        // Fetch the input and output layer descriptions from the model

        TIOModelIO.TIOModelIOList inputList = getIO().getInputs();
        TIOModelIO.TIOModelIOList outputList = getIO().getOutputs();

        // Prepare input buffers

        Object[] inputBuffers = new Object[inputList.size()];

        for (int i = 0; i < inputList.size(); i++){
            TIOLayerInterface inputLayer = inputList.get(i);
            Object input = inputs.get(inputLayer.getName());
            ByteBuffer inputBuffer = prepareInputBuffer(input, inputLayer);

            inputBuffers[i] = inputBuffer;
        }

        // Prepare output buffers

        Map<Integer, Object> outputBuffers = new HashMap<>(outputList.size());

        for (int i = 0; i < outputList.size(); i++){
            TIOLayerInterface outputLayer = outputList.get(i);
            ByteBuffer outputBuffer = prepareOutputBuffer(outputLayer);

            outputBuffers.put(i, outputBuffer);
        }

        // Run the model on the input buffers, store the output in the output buffers

        interpreter.runForMultipleInputsOutputs(inputBuffers, outputBuffers);

        // Convert output buffers to user land objects

        return captureOutputs(outputBuffers);
    }

    /**
     * Prepares a ByteBuffer that will be used for input to a model. If buffer caching is used
     * then buffers that have been associated with each layer will be resued.
     *
     * @param input The input to convert to a byte buffer
     * @param inputLayer The interface to the layer that this buffer will be used with
     * @return ByteBuffer ready for input to a model
     * @throws IllegalArgumentException raised if the input cannot be transformed to the format
     *                                  expected by the model
     */

    private ByteBuffer prepareInputBuffer(Object input, TIOLayerInterface inputLayer) throws IllegalArgumentException {
        final AtomicReference<ByteBuffer> inputBuffer = new AtomicReference<>();
        final ByteBuffer cachedBuffer = cacheBuffers ? bufferCache.get(inputLayer) : null;

        inputLayer.doCase((vectorLayer) -> {
            ByteBuffer buffer = vectorDataConverter.toByteBuffer(input, vectorLayer, cachedBuffer);
            inputBuffer.set(buffer);
        }, (pixelLayer) -> {
            ByteBuffer buffer = pixelDataConverter.toByteBuffer(input, pixelLayer, cachedBuffer);
            inputBuffer.set(buffer);
        });
        
        return inputBuffer.get();
    }

    /**
     * Prepares a ByteBuffer that can be used for output from a model. If buffer caching is used
     * then buffers that have been associated with each layer will be resued.
     *
     * @param outputLayer The interface to the layer layer that this buffer will be used with
     * @return ByteBuffer ready for model output
     */

    private ByteBuffer prepareOutputBuffer(TIOLayerInterface outputLayer) {
        if (cacheBuffers) {
            ByteBuffer cached = bufferCache.get(outputLayer);
            cached.rewind();
            return cached;
        }

        final AtomicReference<ByteBuffer> outputBuffer = new AtomicReference<>();

        outputLayer.doCase((vectorLayer) -> {
            ByteBuffer buffer = vectorDataConverter.createBackingBuffer(vectorLayer);
            outputBuffer.set(buffer);
        }, (pixelLayer) -> {
            ByteBuffer buffer = pixelDataConverter.createBackingBuffer(pixelLayer);
            outputBuffer.set(buffer);
        });

        return outputBuffer.get();
    }

    /**
     * Converts captured ByteBuffers from a model's output to user land Objects
     * @param outputs The indexed output buffers
     * @return A Map of keys to user land objects capturing the model's outputs
     */

    private Map<String, Object> captureOutputs(Map<Integer, Object> outputs) {
        TIOModelIO.TIOModelIOList outputList = getIO().getOutputs();
        Map<String, Object> outputMap = new HashMap<>(outputList.size());

        for (int i = 0; i < outputList.size(); i++){
            TIOLayerInterface layer = outputList.get(i);
            String name = layer.getName();

            ByteBuffer buffer = (ByteBuffer)outputs.get(i);
            Object o = captureOutput(buffer, layer);

            outputMap.put(name, o);
        }

        return outputMap;
    }

    /**
     * Converts a single ByteBuffer to a user land object
     * @param buffer The buffer to capture and convert
     * @param layer The interface to the output
     * @return An object in accordance with the layer description, usually one of byte[], float[],
     * or Bitmap
     */

    private Object captureOutput(ByteBuffer buffer, TIOLayerInterface layer) {
        final AtomicReference<Object> output = new AtomicReference<>();

        layer.doCase((vectorLayer) -> {
            Object o = vectorDataConverter.fromByteBuffer(buffer, vectorLayer);

            // If the vector's output is labeled, return a Map of keys to values rather than raw values

            if (vectorLayer.isLabeled()) {
                o = vectorLayer.labeledValues((float[])o);
            }

            // If the output vector is single valued, return the value directly; See #33 and #34

            // if (((float[])o).length == 1) {
            //    o = ((float[])o)[0];
            // }

            output.set(o);
        }, (pixelLayer) -> {
            Object o = pixelDataConverter.fromByteBuffer(buffer, pixelLayer);
            output.set(o);
        });

        return output.get();
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

        interpreter = new Interpreter(tfliteModel, tfliteOptions);
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
        return interpreter.getLastNativeInferenceDurationNanoseconds();
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
