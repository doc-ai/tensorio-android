/*
 * TensorFlowModel.java
 * TensorIO
 *
 * Created by Philip Dow
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

package ai.doc.tensorio.tensorflow.model;

import android.graphics.Bitmap;

import java.util.Map;

import ai.doc.tensorio.core.model.Model;
import java.io.File;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

import ai.doc.tensorio.core.layerinterface.LayerInterface;
import ai.doc.tensorio.core.model.IO;

import ai.doc.tensorio.core.modelbundle.FileModelBundle;
import ai.doc.tensorio.core.modelbundle.ModelBundle;
import androidx.annotation.NonNull;

import ai.doc.tensorflow.SavedModelBundle.Mode;
import ai.doc.tensorflow.SavedModelBundle;
import ai.doc.tensorflow.DataType;
import ai.doc.tensorflow.Tensor;

import ai.doc.tensorio.tensorflow.BuildConfig;
import ai.doc.tensorio.tensorflow.data.BitmapConverter;
import ai.doc.tensorio.tensorflow.data.StringConverter;
import ai.doc.tensorio.tensorflow.data.VectorConverter;

public class TensorFlowModel extends Model {

    // TensorFlow Backend

    SavedModelBundle interpreter;

    // Buffer Caching

    private boolean cacheBuffers = true;
    private Map<LayerInterface, ByteBuffer> bufferCache = null;

    // Data Converters

    final private VectorConverter vectorConverter = new VectorConverter();
    final private BitmapConverter bitmapConverter = new BitmapConverter();
    final private StringConverter stringConverter = new StringConverter();

    public TensorFlowModel(@NonNull ModelBundle bundle) {
        super(bundle);
    }

    @Override
    public void load() throws ModelException {
        if (BuildConfig.DEBUG && getBundle().getClass() != FileModelBundle.class) {
            throw new AssertionError("Assertion failed: ModelBundle must use File and not Asset");
        }

        if (isLoaded()) {
            return;
        }

        // Prepare Buffer Cache

        if (cacheBuffers) {
            prepareBufferCache();
        }

        // Load Model

        Mode tagset = Mode.Serve;
        if (getBundle().getModes().trains()) {
            tagset = Mode.Train;
        }

        File modelDir = Objects.requireNonNull(((FileModelBundle) getBundle()).getModelFile());
        interpreter = new SavedModelBundle(modelDir, tagset);

        super.load();
    }

    public void unload() {
        if (!isLoaded()) {
            return;
        }

        if (interpreter != null ) {
            interpreter = null;
        }

        super.unload();
    }

    /** Create buffer caches that are used for model inputs and outputs */

    // TODO: We're only actually caching input buffers at the moment
    // TODO: Consider caching tensors as well but then batch size must remain consistent

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

    @Override
    public void reload() throws ModelException {
        super.reload();
    }

    @Override
    public Map<String, Object> runOn(float[] input) throws ModelException {
        validateInput(input);
        load();

        return runOn(mappedInput(input));
    }

    @Override
    public Map<String, Object> runOn(byte[] input) throws ModelException {
        validateInput(input);
        load();

        return runOn(mappedInput(input));
    }

    @Override
    public Map<String, Object> runOn(@NonNull Bitmap input) throws ModelException {
        validateInput(input);
        load();

        return runOn(mappedInput(input));
    }

    @Override
    public Map<String, Object> runOn(@NonNull Map<String, Object> inputs) throws ModelException, IllegalArgumentException {
        validateInput(inputs);
        load();

        // Fetch the input and output layer descriptions from the model

        IO.IOList inputList = getIO().getInputs();
        IO.IOList outputList = getIO().getOutputs();

        // Prepare input tensors

        Tensor[] inputTensors = new Tensor[inputList.size()];

        for (int i = 0; i < inputList.size(); i++){
            LayerInterface inputLayer = inputList.get(i);

            String name = inputLayer.getName();
            int[] shape = inputLayer.getTensorShape();
            // TODO: layer data types: it's been just uint8 and float32 for tflite
            DataType dtype = tensorDataType(ai.doc.tensorio.core.layerinterface.DataType.Float32);

            // Batch size of 1
            if (shape[0] == -1) {
                shape[0] = 1;
            }

            Object input = Objects.requireNonNull(inputs.get(name));
            ByteBuffer inputBuffer = prepareInputBuffer(input, inputLayer);
            Tensor tensor = new Tensor(dtype, shape, name);
            tensor.setBytes(inputBuffer);
            inputTensors[i] = tensor;
        }

        // Prepare output tensors

        Tensor[] outputTensors = new Tensor[outputList.size()];

        for (int i = 0; i < outputList.size(); i++){
            LayerInterface outputLayer = outputList.get(i);

            String name = outputLayer.getName();
            int[] shape = outputLayer.getTensorShape();
            // TODO: layer data types: it's been just uint8 and float32 for tflite
            DataType dtype = tensorDataType(ai.doc.tensorio.core.layerinterface.DataType.Float32);

            // Batch size of 1
            if (shape[0] == -1) {
                shape[0] = 1;
            }

            Tensor tensor = new Tensor(dtype, shape, name);
            outputTensors[i] = tensor;
        }

        // Run the model on the input tensors, store the output in the output tensors

        interpreter.run(inputTensors, outputTensors);

        // Convert output buffers to user land objects

        return captureOutputs(outputTensors);
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
     * Converts a Tensor/IO DataType to a TensorFlow Data Type
     */

    private DataType tensorDataType(ai.doc.tensorio.core.layerinterface.DataType dataType) {
        switch (dataType) {
            case UInt8:
                return DataType.UINT8;
            case Int32:
                return DataType.INT32;
            case Int64:
                return DataType.INT64;
            case Float32:
            case Unknown:
                return DataType.FLOAT32;

        }

        return null;
    }

    /**
     * Converts captured ByteBuffers from a model's output to user land Objects
     * @param tensors The output tensors containing the output byte buffers
     * @return A Map of keys to user land objects capturing the model's outputs
     */

    private Map<String, Object> captureOutputs(@NonNull Tensor[] tensors) {
        IO.IOList outputList = getIO().getOutputs();
        Map<String, Object> outputMap = new HashMap<>(outputList.size());

        for (int i = 0; i < outputList.size(); i++){
            LayerInterface layer = outputList.get(i);
            String name = layer.getName();

            ByteBuffer buffer = tensors[i].getBytes();
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

    // TODO: Add abstract training classes to Model

    //region Train

    // The train methods are the primary interface to a concrete training implementation

    /**
     * Perform training on an map of objects
     * @param inputs A mapping of layer names to arbitrary objects
     * @return results of running the model mapped from the output layer names to the values
     * @throws ModelException Raised if the model has not yet been loaded and the attempt to
     *                           load it fails
     * @throws IllegalArgumentException Raised if the input to the model does not conform to the
     *                                  expected inputs
     */

    public Map<String, Object> trainOn(@NonNull Map<String, Object> inputs) throws ModelException, IllegalArgumentException {
        validateInput(inputs);
        load();

        // Fetch the input and output layer descriptions from the model

        IO.IOList inputList = getIO().getInputs();
        IO.IOList outputList = getIO().getOutputs();

        // Prepare input tensors

        Tensor[] inputTensors = new Tensor[inputList.size()];

        for (int i = 0; i < inputList.size(); i++){
            LayerInterface inputLayer = inputList.get(i);

            String name = inputLayer.getName();
            int[] shape = inputLayer.getTensorShape();
            // TODO: layer data types: it's been just uint8 and float32 for tflite
            DataType dtype = tensorDataType(ai.doc.tensorio.core.layerinterface.DataType.Float32);
            // TODO: Obviously no
            if (name.equals("labels")) {
                dtype = tensorDataType(ai.doc.tensorio.core.layerinterface.DataType.Int32);
            }

            // Batch size of 1
            if (shape[0] == -1) {
                shape[0] = 1;
            }

            Object input = Objects.requireNonNull(inputs.get(name));
            ByteBuffer inputBuffer = prepareInputBuffer(input, inputLayer);
            Tensor tensor = new Tensor(dtype, shape, name);
            tensor.setBytes(inputBuffer);
            inputTensors[i] = tensor;
        }

        // Prepare output tensors

        Tensor[] outputTensors = new Tensor[outputList.size()];

        for (int i = 0; i < outputList.size(); i++){
            LayerInterface outputLayer = outputList.get(i);

            String name = outputLayer.getName();
            int[] shape = outputLayer.getTensorShape();
            // TODO: layer data types: it's been just uint8 and float32 for tflite
            DataType dtype = tensorDataType(ai.doc.tensorio.core.layerinterface.DataType.Float32);

            // Batch size of 1
            if (shape[0] == -1) {
                shape[0] = 1;
            }

            Tensor tensor = new Tensor(dtype, shape, name);
            outputTensors[i] = tensor;
        }

        // Prepare training op names

        // TODO: read from model.json
        String[] trainingOps = new String[1];
        trainingOps[0] = "train";

        // Run the model on the input tensors, store the output in the output tensors

        interpreter.train(inputTensors, outputTensors, trainingOps);

        // Convert output buffers to user land objects

        return captureOutputs(outputTensors);
    }

    //endRegion
}
