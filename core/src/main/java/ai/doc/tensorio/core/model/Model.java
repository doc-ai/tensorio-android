/*
 * Model.java
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

package ai.doc.tensorio.core.model;

import android.graphics.Bitmap;

import ai.doc.tensorio.core.data.Batch;
import ai.doc.tensorio.core.data.Placeholders;
import ai.doc.tensorio.core.modelbundle.ModelBundle;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.nio.ByteBuffer;
import java.util.Map;

import ai.doc.tensorio.core.layerinterface.LayerInterface;

/**
 * A Java wrapper around lower level, usually C++ model implementations. This is the primary
 * API provided by the TensorIO framework.
 *
 * A `Model` is built from a bundle folder that contains the underlying model, a json description
 * of the model's input and output layers, and any additional assets required by the model, for
 * example, output labels.
 *
 * A conforming `Model` begins by parsing a json description of the model's input and output
 * layers, producing a `LayerInterface` for each layer. Each layer is fully described by a
 * conforming `LayerDescription`, which describes the data the layer expects or produces, for
 * example, whether it is quantized, any transformations that should be applied to it, and the
 * number of bytes the layer expects.
 *
 * Note that, currently, only TensorFlow Lite (TFLite) models are supported.
 *
 * @warning
 * Models are not thread safe. Models may be used on separate threads, so that you can perform
 * inference off the main thread, but you should not use the same model from multiple threads.
 */

public abstract class Model {

    public static class ModelException extends Exception {
        public ModelException(@NonNull String message) {
            super(message);
        }

        public ModelException(@NonNull String message, @NonNull Throwable cause) {
            super(message, cause);
        }
    }

    /**
     * The `ModelBundle` object from which this model was instantiated.
     */

    private ModelBundle bundle;

    /**
     * Options associated with this model.
     */

    private Options options;

    /**
     * Modes associated with the model, e.g. whether it has support for prediction, training, and evaluation
     */

    private Modes modes;

    /**
     * A string uniquely identifying this model, taken from the model bundle.
     */

    private String identifier;

    /**
     * Human readable name of the model, taken from the model bundle.
     */

    private String name;

    /**
     * Additional information about the model, taken from the model bundle.
     */

    private String details;

    /**
     * The model's authors, taken from the model bundle.
     */

    private String author;

    /**
     * The model's license, taken from the model bundle.
     */

    private String license;

    /**
     * A boolean value indicating if this is a placeholder bundle.
     * <p>
     * A placeholder bundle has no underlying model and instantiates a `Model` that does nothing.
     * Placeholders bundles are used to collect labeled data for models that haven't been trained yet.
     */

    private boolean placeholder;

    /**
     * A boolean value indicating if the model is quantized or not.
     * <p>
     * Quantized models have 8 bit `uint8_t` interfaces while unquantized modesl have 32 bit, `float_t`
     * interfaces.
     */

    private boolean quantized;

    /**
     * A string indicating the kind of model this is, e.g. "image.classification.imagenet"
     */

    private String type;

    /**
     * A boolean value indicating whether the model has been loaded or not. Conforming classes may want
     * to wrap the underlying models such that they can be aggressively loaded and unloaded from memory,
     * as some models contain hundreds of megabytes of paramters.
     */

    protected boolean loaded;

    /**
     * Contains the descriptions of the model's inputs, outputs, and placeholders
     * accessible by numeric index or by name. Not all model backends support
     * placeholders.
     *
     * @code
     * io.inputs[0]
     * io.inputs[@"image"]
     * io.outputs[0]
     * io.outputs[@"label"]
     * io.placeholders[0]
     * io.placeholders[@"label"]
     * @endcode
     */

    private IO io;

    /**
     * The designated initializer for conforming classes.
     *
     * You should not need to call this method directly. Instead, acquire an instance of a `ModelBundle`
     * associated with this model by way of the model's identifier. Then the `ModelBundle` class
     * calls this `initWithBundle:` factory initialization method, which conforming classes may override
     * to support custom initialization.
     *
     * @param bundle `ModelBundle` containing information about the model and its path
     * @return instancetype An instance of the conforming class, may be `nil`.
     */

    public Model(@NonNull ModelBundle bundle) {
        this.bundle = bundle;

        this.options = bundle.getOptions();
        this.modes = bundle.getModes();
        this.identifier = bundle.getIdentifier();
        this.name = bundle.getName();
        this.details = bundle.getDetails();
        this.author = bundle.getAuthor();
        this.license = bundle.getLicense();
        this.placeholder = bundle.isPlaceholder();
        this.quantized = bundle.isQuantized();
        this.type = bundle.getType();
        this.io = bundle.getIO();
    }

    //region Getters and Setters

    public ModelBundle getBundle() {
        return bundle;
    }

    public Options getOptions() {
        return options;
    }

    public String getIdentifier() {
        return identifier;
    }

    public String getName() {
        return name;
    }

    public String getDetails() {
        return details;
    }

    public String getAuthor() {
        return author;
    }

    public String getLicense() {
        return license;
    }

    public boolean isPlaceholder() {
        return placeholder;
    }

    public boolean isQuantized() {
        return quantized;
    }

    public String getType() {
        return type;
    }

    public boolean isLoaded() {
        return loaded;
    }

    public IO getIO() {
        return io;
    }

    //endRegion

    //region Lifecycle

    /**
     * Loads a model into memory.
     *
     * A model should load itself prior to running on any input, but consumers of the model may want
     * more control over when a model is loaded in order to avoid placing parameters into memory
     * before they are needed.
     *
     * Conforming classes should override this method to perform custom loading and set loaded=true
     * or call super's implementation after loading has been successful.
     *
     * @@throws ModelException
     */

    public void load() throws ModelException {
        loaded = true;
    }

    /**
     * Reloads a model. Use this method when you have changed model configuration and must reload
     * the model in order for those changes to take effect.
     *
     * Conforming classes should override this method to perform custom reloading and set loaded=true
     * or call super's implementation after loading has been successful.
     *
     * @throws ModelException
     */

    public void reload() throws ModelException {
        loaded = true;
    }

    /**
     * Unloads a model from memory
     *
     * A model will unload its resources automatically when it is deallocated, but the unload function
     * may do this as well in order to provide finer grained control to consumers.
     *
     * Conforming classes should override this method to perform custom unloading and set loaded=false
     * or call super's implementation after unloading has been successful.
     */

    public void unload() {
        loaded = false;
    }

    //endRegion

    //region Run

    // The run methods are the primary interface to a concrete implementation

    /**
     * Perform inference on an array of floats for a single input layer.
     *
     * @param input an array of floats
     * @return results of running the model mapped from the output layer names to the values
     * @throws ModelException If the model has not yet been loaded and the attempt to load it fails
     * @throws IllegalArgumentException If the input to the model does not conform to the expected inputs
     * or if the backend does not support float32s
     */

    public abstract Map<String, Object> runOn(float[] input) throws ModelException, IllegalArgumentException;

    /**
     * Perform inference on an array of bytes (uint8) for a single input layer.
     *
     * @param input an array of bytes
     * @return results of running the model mapped from the output layer names to the values
     * @throws ModelException If the model has not yet been loaded and the attempt to load it fails
     * @throws IllegalArgumentException If the input to the model does not conform to the expected inputs
     * or if the backend does not support bytes
     */

    public abstract Map<String, Object> runOn(byte[] input) throws ModelException, IllegalArgumentException;

    /**
     * Perform inference on an array of int32s for a single input layer. Not all backends support
     * int32 inputs and will throw an exception if this method is not supported.
     *
     * @param input An array of int32s
     * @return results of running the model mapped from the output layer names to the values
     * @throws ModelException If the model has not yet been loaded and the attempt to load it fails
     * @throws IllegalArgumentException If the input to the model does not conform to the expected inputs
     * or if the backend does not support int32s
     */

    public abstract Map<String, Object> runOn(int[] input) throws ModelException, IllegalArgumentException;

    /**
     * Perform inference on a ByteBuffer for a single input layer. Not all backends support
     * ByteBuffer inputs and will throw an exception if this method is not supported. ByteBuffers
     * should only be used with `string` type inputs.
     *
     * @param input A ByteBuffer that will be sent directly to the model with no additional preprocessing
     * @return results of running the model mapped from the output layer names to the values
     * @throws ModelException If the model has not yet been loaded and the attempt to load it fails
     * @throws IllegalArgumentException If the input to the model does not conform to the expected inputs
     * or if the backend does not support ByteBuffers
     */

    public abstract Map<String, Object> runOn(ByteBuffer input) throws ModelException, IllegalArgumentException;

    /**
     * Perform inference on a Bitmap for a single input layer.
     *
     * @param input A Bitmap
     * @return results of running the model mapped from the output layer names to the values
     * @throws ModelException If the model has not yet been loaded and the attempt to load it fails
     * @throws IllegalArgumentException If the input to the model does not conform to the expected inputs
     * or if the backend does not support Bitmaps
     */

    public abstract Map<String, Object> runOn(@NonNull Bitmap input) throws ModelException, IllegalArgumentException;

    /**
     * Perform inference on an map of objects
     *
     * @param input A mapping of layer names to arbitrary objects
     * @return results of running the model mapped from the output layer names to the values
     * @throws ModelException If the model has not yet been loaded and the attempt to load it fails
     * @throws IllegalArgumentException If the input to the model does not conform to the expected inputs
     */

    public abstract Map<String, Object> runOn(@NonNull Map<String, Object> input) throws ModelException, IllegalArgumentException;

    /**
     * Perform inference on an map of objects with placeholders. Not all backends support placeholders,
     * in which case concrete implementations should raise an exception.
     *
     * @param input A mapping of layer names to arbitrary objects
     * @param placeholders A mapping of placeholder layer names to arbitrary objects. May be nil, in
     *                     which case calling this method should be no different from calling runOn
     *                     without placeholders.
     * @return results of running the model mapped from the output layer names to the values
     * @throws ModelException If the model has not yet been loaded and the attempt to load it fails
     * @throws IllegalArgumentException If the input to the model does not conform to the expected inputs
     */

    public abstract Map<String, Object> runOn(@NonNull Map<String, Object> input, @Nullable Placeholders placeholders) throws ModelException, IllegalArgumentException;

    //endRegion

    //region Input Validation

    // TODO: Write unit tests for these methods

    protected void validateInput(float[] input) throws IllegalArgumentException {
        if (io.getInputs().size() != 1) {
            throw InputCountMismatchException(1, io.getInputs().size());
        }
    }

    protected void validateInput(byte[] input) throws IllegalArgumentException {
        if (io.getInputs().size() != 1) {
            throw InputCountMismatchException(1, io.getInputs().size());
        }
    }

    protected void validateInput(int[] input) throws IllegalArgumentException {
        if (io.getInputs().size() != 1) {
            throw InputCountMismatchException(1, io.getInputs().size());
        }
    }

    protected void validateInput(@NonNull ByteBuffer input) throws IllegalArgumentException {
        if (io.getInputs().size() != 1) {
            throw InputCountMismatchException(1, io.getInputs().size());
        }
    }

    protected void validateInput(@NonNull Bitmap input) throws IllegalArgumentException {
        if (io.getInputs().size() != 1) {
            throw InputCountMismatchException(1, io.getInputs().size());
        }
    }

    protected void validateInput(@NonNull Map<String, Object> input) throws IllegalArgumentException {
        int expectedSize = io.getInputs().size();
        int receivedSize = input.size();

        if (expectedSize != receivedSize) {
            throw InputCountMismatchException(expectedSize, receivedSize);
        }

        if ( !input.keySet().equals(io.getInputs().keys()) ) {
            for (LayerInterface layer : io.getInputs().all()) {
                if ( !input.containsKey(layer.getName()) ) {
                    throw MissingInput(layer.getName());
                }
            }
        }
    }

    protected void validateInput(@NonNull Batch batch) throws IllegalArgumentException {
        int expectedSize = io.getInputs().size();
        int receivedSize = batch.getKeys().length;

        if (expectedSize != receivedSize) {
            throw InputCountMismatchException(expectedSize, receivedSize);
        }

        if ( !batch.getKeyset().equals(io.getInputs().keys()) ) {
            for (LayerInterface layer : io.getInputs().all()) {
                if ( !batch.getKeyset().contains(layer.getName()) ) {
                    throw MissingInput(layer.getName());
                }
            }
        }
    }

    protected void validatePlaceholders(@Nullable Placeholders placeholders) throws IllegalArgumentException {
        int expectedSize = io.getPlaceholders().size();
        int receivedSize = placeholders == null ? 0 : placeholders.size();

        if (expectedSize != receivedSize) {
            throw PlaceholdersCountMismatchException(expectedSize, receivedSize);
        }

        if (placeholders == null) {
            return;
        }

        if ( !placeholders.keySet().equals(io.getPlaceholders().keys()) ) {
            for (LayerInterface layer : io.getPlaceholders().all()) {
                if ( !placeholders.containsKey(layer.getName()) ) {
                    throw MissingPlaceholder(layer.getName());
                }
            }
        }
    }

    //endRegion

    //region Utilities

    @NonNull
    @Override
    public String toString() {
        return "Model{" +
                " options=" + options +
                ", identifier='" + identifier + '\'' +
                ", name='" + name + '\'' +
                ", details='" + details + '\'' +
                ", author='" + author + '\'' +
                ", license='" + license + '\'' +
                ", placeholder=" + placeholder +
                ", quantized=" + quantized +
                ", type='" + type + '\'' +
                ", loaded=" + loaded +
                ", inputs=" + io.getInputs().toString() +
                ", outputs=" + io.getOutputs().toString() +
                '}';
    }

    //endRegion

    //region Exceptions

    private static IllegalArgumentException InputCountMismatchException(int expected, int received) {
        return new IllegalArgumentException("The model has " + expected + " input layers but received " + received + " inputs");
    }

    private static IllegalArgumentException MissingInput(@NonNull String name) {
        return new IllegalArgumentException("The model received no input for layer \"" + name + "\"");
    }

    private static IllegalArgumentException PlaceholdersCountMismatchException(int expected, int received) {
        return new IllegalArgumentException("The model has " + expected + " placeholders but received " + received + " placeholders");
    }

    private static IllegalArgumentException MissingPlaceholder(@NonNull String name) {
        return new IllegalArgumentException("The model received no placeholder for layer \"" + name + "\"");
    }

    // endRegion

}
