package ai.doc.tensorio.pytorch.model;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;


import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import ai.doc.tensorio.core.data.Placeholders;
import ai.doc.tensorio.core.layerinterface.LayerInterface;
import ai.doc.tensorio.core.model.IO;
import ai.doc.tensorio.core.model.Model;
import ai.doc.tensorio.core.modelbundle.AssetModelBundle;
import ai.doc.tensorio.core.modelbundle.FileModelBundle;
import ai.doc.tensorio.core.modelbundle.ModelBundle;
import ai.doc.tensorio.pytorch.data.BitmapConverter;
import ai.doc.tensorio.pytorch.data.StringConverter;
import ai.doc.tensorio.pytorch.data.VectorConverter;

public class PytorchModel extends Model {

    // Pytorch Backend

    private Module pytorchModule;

    // Buffer Caching

    private boolean cacheBuffers = true;
    private Map<LayerInterface, ByteBuffer> bufferCache = null;

    // Data Converters

    final private BitmapConverter bitmapConverter = new BitmapConverter();
    final private VectorConverter vectorConverter = new VectorConverter();
    final private StringConverter stringConverter = new StringConverter();


    /**
     * The designated initializer for conforming classes.
     * <p>
     * You should not need to call this method directly. Instead, acquire an instance of a `ModelBundle`
     * associated with this model by way of the model's identifier. Then the `ModelBundle` class
     * calls this `initWithBundle:` factory initialization method, which conforming classes may override
     * to support custom initialization.
     *
     * @param bundle `ModelBundle` containing information about the model and its path
     * @return instancetype An instance of the conforming class, may be `null`.
     */
    public PytorchModel(@NonNull ModelBundle bundle) {
        super(bundle);
    }

    @Override
    public void load() throws ModelException {
        if (isLoaded()) {
            return;
        }

        // Prepare Buffer Cache

        if (cacheBuffers) {
            prepareBufferCache();
        }

        try {
            pytorchModule = loadModelFile();
        } catch (IOException e) {
            throw new ModelException("Error loading model file", e);
        }

        super.load();

    }

    @Override
    public void reload() throws ModelException {
        if (pytorchModule != null) {
            pytorchModule.destroy();
            pytorchModule = null;
        }

        load();
        super.reload();
    }

    @Override
    public void unload() {
        if (!isLoaded()) {
            return;
        }

        if (pytorchModule != null) {
            pytorchModule.destroy();
            pytorchModule = null;
        }

        super.unload();
    }

    /**
     * Create buffer caches that are used for model inputs and outputs
     */

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
    public Map<String, Object> runOn(float[] input) throws ModelException {
        validateInput(input);
        load();

        if (hasMultipleInputsOrOutputs()) {
            return runMultipleInputMultipleOutput(mappedInput(input));
        } else {
            return runSingleInputSingleOutput(input);
        }
    }

    @Override
    public Map<String, Object> runOn(byte[] input) throws ModelException {
        validateInput(input);
        load();

        if (hasMultipleInputsOrOutputs()) {
            return runMultipleInputMultipleOutput(mappedInput(input));
        } else {
            return runSingleInputSingleOutput(input);
        }
    }

    @Override
    public Map<String, Object> runOn(int[] input) throws ModelException, IllegalArgumentException {
        validateInput(input);
        load();

        if (hasMultipleInputsOrOutputs()) {
            return runMultipleInputMultipleOutput(mappedInput(input));
        } else {
            return runSingleInputSingleOutput(input);
        }
    }

    @Override
    public Map<String, Object> runOn(long[] input) throws ModelException, IllegalArgumentException {
        validateInput(input);
        load();

        if (hasMultipleInputsOrOutputs()) {
            return runMultipleInputMultipleOutput(mappedInput(input));
        } else {
            return runSingleInputSingleOutput(input);
        }
    }

    @Override
    public Map<String, Object> runOn(ByteBuffer input) throws ModelException, IllegalArgumentException {
        throw new IllegalArgumentException("ByteBuffer inputs are not supported by Pytorch");
    }

    @Override
    public Map<String, Object> runOn(@NonNull Bitmap input) throws ModelException {
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

    @Override
    public Map<String, Object> runOn(@NonNull Map<String, Object> input, @Nullable Placeholders placeholders) throws ModelException, IllegalArgumentException {
        throw new ModelException("Placeholders not supported by Pytorch models");
    }

    /**
     * Used to determined if an unmapped input should be mapped and a mapped input unmapped
     *
     * @return true if models has either of more than one input or output, false otherwise
     */

    private boolean hasMultipleInputsOrOutputs() {
        return getIO().getInputs().size() > 1 || getIO().getOutputs().size() > 1;
    }

    /**
     * Converts a single input to a mapped input using the single input layer's name.
     * <p>
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
     * <p>
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
     *
     * @param input An input in one of the supported types, e.g. byte[], float[], or Bitmap
     * @return The model's single output mapped by the output layers name
     * @throws IllegalArgumentException raised if the input cannot be transformed to the format
     *                                  expected by the model
     */

    private Map<String, Object> runSingleInputSingleOutput(@NonNull Object input) throws IllegalArgumentException {

        // Fetch the input layer descriptions from the model

        LayerInterface inputLayer = getIO().getInputs().get(0);

        // Prepare input tensor
        Tensor inputTensor = prepareInputTensor(input, inputLayer);


        IValue outputTensor = pytorchModule.forward(IValue.from(inputTensor));

        // Convert output buffers to user land objects
        Map<Integer, IValue> outputs = new HashMap<>(getIO().getOutputs().size()); // Always size 1
        outputs.put(0, outputTensor);

        return captureOutputs(outputs);
    }

    /**
     * Actually performs inference on multiple inputs or multiple outputs
     *
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
        Map<String, IValue> inputMap = new HashMap<>();
        IValue[] inputTensors = new IValue[inputList.size()];

        for (int i = 0; i < inputList.size(); i++){
            LayerInterface inputLayer = inputList.get(i);
            Object input = inputs.get(inputLayer.getName());
            Tensor inputTensor = prepareInputTensor(input, inputLayer);

            inputTensors[i] = IValue.from(inputTensor);
            inputMap.put(inputLayer.getName(), inputTensors[i]);
        }

        IValue temp = pytorchModule.forward(inputTensors);
        Map<Integer, IValue> outputMap = new HashMap<>(outputList.size());

        if (temp.isTuple()){
            IValue[] output = temp.toTuple();
            for (int i=0; i< output.length; i++){
                outputMap.put(i, output[i]);
            }
        }
        else if (temp.isDictStringKey()){
            Map<String, IValue> tempMap = temp.toDictStringKey();
            for (String n: temp.toDictStringKey().keySet()){
                outputMap.put(outputList.indexFor(n), tempMap.get(n));
            }
        }
        else {
            outputMap.put(0, temp);
        }



        return captureOutputs(outputMap);
    }

    /**
     * Prepares a Tensor that will be used for input to a model. If buffer caching is used
     * then buffers that have been associated with each layer will be reused.
     *
     * @param input The input to convert to a Tensor
     * @param inputLayer The interface to the layer that this buffer will be used with
     * @return ByteBuffer ready for input to a model
     * @throws IllegalArgumentException raised if the input cannot be transformed to the format
     *                                  expected by the model
     */
    private Tensor prepareInputTensor(Object input, LayerInterface inputLayer) throws IllegalArgumentException {
        final AtomicReference<Tensor> inputBuffer = new AtomicReference<>();
        final ByteBuffer cachedBuffer = cacheBuffers ? bufferCache.get(inputLayer) : null;

        inputLayer.doCase((vectorLayer) -> {
            Tensor tensor = vectorConverter.toTensor(input, vectorLayer, cachedBuffer);
            inputBuffer.set(tensor);
        }, (pixelLayer) -> {
            Tensor tensor = bitmapConverter.toTensor(input, pixelLayer, cachedBuffer);
            inputBuffer.set(tensor);
        }, (stringLayer) -> {
            Tensor tensor = stringConverter.toTensor(input, stringLayer, cachedBuffer);
            inputBuffer.set(tensor);
        });

        return inputBuffer.get();
    }

    /**
     * Converts captured Tensors from a model's output to user land Objects
     *
     * @param outputs The indexed output buffers
     * @return A Map of keys to user land objects capturing the model's outputs
     */

    private Map<String, Object> captureOutputs(@NonNull Map<Integer, IValue> outputs) {
        IO.IOList outputList = getIO().getOutputs();
        Map<String, Object> outputMap = new HashMap<>(outputList.size());

        for (int i = 0; i < outputList.size(); i++) {
            LayerInterface layer = outputList.get(i);
            String name = layer.getName();

            Object o = captureOutput(outputs.get(i), layer);

            outputMap.put(name, o);
        }

        return outputMap;
    }

    /**
     * Converts a single Tensor to a user land object
     *
     * @param tensor The tensor to capture and convert
     * @param layer  The interface to the output
     * @return An object in accordance with the layer description, usually one of byte[], float[],
     * or Bitmap
     */

    private Object captureOutput(@NonNull IValue tensor, @NonNull LayerInterface layer) {
        final AtomicReference<Object> output = new AtomicReference<>();

        layer.doCase((vectorLayer) -> {
            Object o = vectorConverter.fromTensor(tensor.toTensor(), vectorLayer);

            // If the vector's output is labeled, return a Map of keys to values rather than raw values
            if (vectorLayer.isLabeled()) {
                o = vectorLayer.labeledValues((float[]) o);
            }

            // If the output vector is single valued, return the value directly; See #33 and #34

            // if (((float[])o).length == 1) {
            //    o = ((float[])o)[0];
            // }

            output.set(o);
        }, (pixelLayer) -> {
            Object o = bitmapConverter.fromTensor(tensor.toTensor(), pixelLayer);
            output.set(o);
        }, (stringLayer) -> {
            Object o = stringConverter.fromTensor(tensor.toTensor(), stringLayer);
            output.set(o);
        });

        return output.get();
    }

    //endRegion

    //region Utilities

    private Module loadModelFile() throws IOException {
        String filename;

        if (getBundle() instanceof AssetModelBundle) {
            AssetModelBundle bundle = (AssetModelBundle) getBundle();
            InputStream inputStream = bundle.getContext().getAssets().open(bundle.getModelFilename());


            File file = new File(bundle.getContext().getFilesDir(), bundle.getFilename());
            if (file.exists() && file.length() > 0) {
                filename = file.getAbsolutePath();
            } else {

                try (OutputStream os = new FileOutputStream(file)) {
                    byte[] buffer = new byte[4 * 1024];
                    int read;
                    while ((read = inputStream.read(buffer)) != -1) {
                        os.write(buffer, 0, read);
                    }
                    os.flush();
                }
                filename = file.getAbsolutePath();
            }
        } else if (getBundle() instanceof FileModelBundle) {
            FileModelBundle bundle = (FileModelBundle) getBundle();
            filename = bundle.getModelFile().getAbsolutePath();
        } else {
            throw new FileNotFoundException();
        }

        return Module.load(filename);
    }

    //endRegion

}
