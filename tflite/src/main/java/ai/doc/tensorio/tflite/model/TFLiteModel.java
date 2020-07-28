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

package ai.doc.tensorio.tflite.model;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import ai.doc.tensorio.core.layerinterface.LayerInterface;
import ai.doc.tensorio.core.model.Model;
import ai.doc.tensorio.core.modelbundle.ModelBundle;
import ai.doc.tensorio.core.model.ModelException;
import ai.doc.tensorio.core.model.IO;
import ai.doc.tensorio.tflite.data.BitmapConverter;
import ai.doc.tensorio.tflite.data.StringConverter;
import ai.doc.tensorio.tflite.data.VectorConverter;
import androidx.annotation.NonNull;

public class TFLiteModel extends Model {

    public enum HardwareBacking {
        CPU,
        GPU,
        NNAPI
    }

    // TFLite Backend

    private Interpreter interpreter;
    private MappedByteBuffer tfliteModel;
    private GpuDelegate gpuDelegate = null;
    private NnApiDelegate nnApiDelegate = null;

    // TFLite Backend Options

    private HardwareBacking hardwareBacking = HardwareBacking.CPU;
    private boolean use16BitPrecision = false;
    private int numThreads = -1;

    // Buffer Caching

    private boolean cacheBuffers = true;
    private Map<LayerInterface, ByteBuffer> bufferCache = null;

    // Data Converters

    final private VectorConverter vectorConverter = new VectorConverter();
    final private BitmapConverter bitmapConverter = new BitmapConverter();
    final private StringConverter stringConverter = new StringConverter();

    // Backend Options Getters and Setters

    /** Sets hardware backend. After calling this method you must call reload() for changes to take effect. */

    public void setHardwareBacking(HardwareBacking hardwareBacking) {
        this.hardwareBacking = hardwareBacking;
    }

    public HardwareBacking getHardwareBacking() {
        return hardwareBacking;
    }

    /** Sets number of model threads, default is -1. After calling this method you must call reload() for changes to take effect. */

    public void setNumThreads(int numThreads) {
        this.numThreads = numThreads;
    }

    public int getNumThreads() {
        return numThreads;
    }

    /** Sets floating point precision. After calling this method you must call reload() for changes to take effect. */

    public void setUse16BitPrecision(boolean use16bitPrecision) {
        this.use16BitPrecision = use16bitPrecision;
    }

    public boolean use16BitPrecision() {
        return use16BitPrecision;
    }

    // Constructor

    public TFLiteModel(@NonNull ModelBundle bundle) {
        super(bundle);
    }

    //region Lifecycle

    @Override
    public void load() throws ModelException {
        if (isLoaded()) {
            return;
        }

        // Prepare Buffer Cache

        if (cacheBuffers) {
            prepareBufferCache();
        }

        // Load Model

        try {
            tfliteModel = loadModelFile();
        } catch (IOException e) {
            throw new ModelException("Error loading model file", e);
        }

        // Prepare Interpreter

        createInterpreter();

        super.load();
    }

    /** Recreate the interpreter without reloading the model from disk **/

    @Override
    public void reload() throws ModelException {
        if (interpreter != null ) {
            interpreter.close();
            interpreter = null;
        }

        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }

        if (nnApiDelegate != null) {
            nnApiDelegate.close();
            nnApiDelegate = null;
        }

        createInterpreter();
        super.reload();
    }

    @Override
    public void unload() {
        if (!isLoaded()) {
            return;
        }

        if (interpreter != null ) {
            interpreter.close();
            interpreter = null;
        }

        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }

        if (nnApiDelegate != null) {
            nnApiDelegate.close();
            nnApiDelegate = null;
        }

        if (bufferCache != null) {
            bufferCache = null;
        }

        super.unload();
    }

    private void createInterpreter() {

        // Options

        Interpreter.Options options = new Interpreter.Options();

        options.setNumThreads(numThreads);
        options.setUseNNAPI(hardwareBacking == HardwareBacking.NNAPI);
        options.setAllowFp16PrecisionForFp32(use16BitPrecision);

        // GPU Delegate

        if (hardwareBacking == HardwareBacking.GPU && GpuDelegateHelper.isGpuDelegateAvailable()) {
            gpuDelegate = (GpuDelegate) GpuDelegateHelper.createGpuDelegate();
            options.addDelegate(gpuDelegate);
        }

        // NNAPI Delegate

        if (hardwareBacking == HardwareBacking.NNAPI && NnApiDelegateHelper.isNnApiDelegateAvailable()) {
            nnApiDelegate = (NnApiDelegate) NnApiDelegateHelper.createNnApiDelegate();
            options.addDelegate(nnApiDelegate);
        }

        // Interpreter

        interpreter = new Interpreter(tfliteModel, options);
    }

    /** Create buffer caches that are used for model inputs and outputs */

    void prepareBufferCache() {
        bufferCache = new HashMap<>();

        List<LayerInterface> layers = new ArrayList<>();
        layers.addAll(getIO().getInputs().all());
        layers.addAll(getIO().getOutputs().all());

        for (LayerInterface layer : layers) {
            layer.doCase((vectorLayer) -> {
                bufferCache.put(layer, vectorConverter.createBackingBuffer(vectorLayer));
            }, (pixelLayer) -> {
                bufferCache.put(layer, bitmapConverter.createBackingBuffer(pixelLayer));
            }, (stringLayer) -> {
                bufferCache.put(layer, stringConverter.createBackingBuffer(stringLayer));
            });
        }
    }

    //endRegion

    //region Run

    @Override
    public Map<String, Object> runOn(float[] input) throws ModelException, IllegalArgumentException {
        validateInput(input);
        load();

        if (hasMultipleInputsOrOutputs()) {
            return runMultipleInputMultipleOutput(mappedInput(input));
        } else {
            return runSingleInputSingleOutput(input);
        }
    }

    @Override
    public Map<String, Object> runOn(byte[] input) throws ModelException, IllegalArgumentException {
        validateInput(input);
        load();

        if (hasMultipleInputsOrOutputs()) {
            return runMultipleInputMultipleOutput(mappedInput(input));
        } else {
            return runSingleInputSingleOutput(input);
        }
    }

    @Override
    public Map<String, Object> runOn(@NonNull Bitmap input) throws ModelException, IllegalArgumentException {
        validateInput(input);
        load();

        if (hasMultipleInputsOrOutputs()) {
            return runMultipleInputMultipleOutput(mappedInput(input));
        } else {
            return runSingleInputSingleOutput(input);
        }
    }

    @Override
    public Map<String, Object> runOn(@NonNull Map<String, Object> input) throws ModelException, IllegalArgumentException {
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

    private Map<String, Object> mappedInput(@NonNull Object input) {
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

    private Object unmappedInput(@NonNull Map<String, Object> input) {
        return input.get(getIO().getInputs().get(0).getName());
    }

    /**
     * Actually performs inference on a single input with a single output
     * @param input An input in one of the supported types, e.g. byte[], float[], or Bitmap
     * @return The model's single output mapped by the output layers name
     * @throws IllegalArgumentException raised if the input cannot be transformed to the format
     *                                  expected by the model
     */

    private Map<String, Object> runSingleInputSingleOutput(@NonNull Object input) throws IllegalArgumentException {

        // Fetch the input and output layer descriptions from the model

        LayerInterface inputLayer = getIO().getInputs().get(0);
        LayerInterface outputLayer = getIO().getOutputs().get(0);

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

    private Map<String, Object> runMultipleInputMultipleOutput(@NonNull Map<String, Object> inputs) throws IllegalArgumentException {

        // Fetch the input and output layer descriptions from the model

        IO.IOList inputList = getIO().getInputs();
        IO.IOList outputList = getIO().getOutputs();

        // Prepare input buffers

        Object[] inputBuffers = new Object[inputList.size()];

        for (int i = 0; i < inputList.size(); i++){
            LayerInterface inputLayer = inputList.get(i);
            Object input = inputs.get(inputLayer.getName());
            ByteBuffer inputBuffer = prepareInputBuffer(input, inputLayer);

            inputBuffers[i] = inputBuffer;
        }

        // Prepare output buffers

        Map<Integer, Object> outputBuffers = new HashMap<>(outputList.size());

        for (int i = 0; i < outputList.size(); i++){
            LayerInterface outputLayer = outputList.get(i);
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

    private ByteBuffer prepareInputBuffer(@NonNull Object input, @NonNull LayerInterface inputLayer) throws IllegalArgumentException {
        final AtomicReference<ByteBuffer> inputBuffer = new AtomicReference<>();
        final ByteBuffer cachedBuffer = cacheBuffers ? bufferCache.get(inputLayer) : null;

        inputLayer.doCase((vectorLayer) -> {
            ByteBuffer buffer = vectorConverter.toByteBuffer(input, vectorLayer, cachedBuffer);
            inputBuffer.set(buffer);
        }, (pixelLayer) -> {
            ByteBuffer buffer = bitmapConverter.toByteBuffer(input, pixelLayer, cachedBuffer);
            inputBuffer.set(buffer);
        }, (stringLayer) -> {
            ByteBuffer buffer = stringConverter.toByteBuffer(input, stringLayer, cachedBuffer);
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

    private ByteBuffer prepareOutputBuffer(@NonNull LayerInterface outputLayer) {
        if (cacheBuffers) {
            ByteBuffer cached = bufferCache.get(outputLayer);
            cached.rewind();
            return cached;
        }

        final AtomicReference<ByteBuffer> outputBuffer = new AtomicReference<>();

        outputLayer.doCase((vectorLayer) -> {
            ByteBuffer buffer = vectorConverter.createBackingBuffer(vectorLayer);
            outputBuffer.set(buffer);
        }, (pixelLayer) -> {
            ByteBuffer buffer = bitmapConverter.createBackingBuffer(pixelLayer);
            outputBuffer.set(buffer);
        }, (stringLayer) -> {
            ByteBuffer buffer = stringConverter.createBackingBuffer(stringLayer);
            outputBuffer.set(buffer);
        });

        return outputBuffer.get();
    }

    /**
     * Converts captured ByteBuffers from a model's output to user land Objects
     * @param outputs The indexed output buffers
     * @return A Map of keys to user land objects capturing the model's outputs
     */

    private Map<String, Object> captureOutputs(@NonNull Map<Integer, Object> outputs) {
        IO.IOList outputList = getIO().getOutputs();
        Map<String, Object> outputMap = new HashMap<>(outputList.size());

        for (int i = 0; i < outputList.size(); i++){
            LayerInterface layer = outputList.get(i);
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

    private Object captureOutput(@NonNull ByteBuffer buffer, @NonNull LayerInterface layer) {
        final AtomicReference<Object> output = new AtomicReference<>();

        layer.doCase((vectorLayer) -> {
            Object o = vectorConverter.fromByteBuffer(buffer, vectorLayer);

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
            Object o = bitmapConverter.fromByteBuffer(buffer, pixelLayer);
            output.set(o);
        }, (stringLayer) -> {
            Object o = stringConverter.fromByteBuffer(buffer, stringLayer);
            output.set(o);
        });

        return output.get();
    }

    //endRegion

    //region Utilities

    public long getLastInferenceDuration(){
        return interpreter.getLastNativeInferenceDurationNanoseconds();
    }

    private MappedByteBuffer loadModelFile() throws IOException {

        // So barf
        switch (getBundle().getSource()) {
            case Asset: {
                AssetFileDescriptor fileDescriptor = getBundle().getContext().getAssets().openFd(getBundle().getModelFilename());
                FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
                FileChannel fileChannel = inputStream.getChannel();

                long startOffset = fileDescriptor.getStartOffset();
                long length = fileDescriptor.getDeclaredLength();

                return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, length);
            }
            case File: {
                FileInputStream inputStream = new FileInputStream(getBundle().getModelFile());
                FileChannel fileChannel = inputStream.getChannel();

                long startOffset = 0;
                long length = fileChannel.size();

                return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, length);
            }
            default:
                throw new FileNotFoundException();
        }

    }

    //endRegion
}
