/*
 * TIOModel.java
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

package ai.doc.tensorio.TIOModel;

import android.content.Context;
import android.graphics.Bitmap;
import androidx.annotation.NonNull;

import java.util.Map;

import ai.doc.tensorio.TIOLayerInterface.TIOLayerInterface;

/**
 * A Java wrapper around lower level, usually C++ model implementations. This is the primary
 * API provided by the TensorIO framework.
 *
 * A `TIOModel` is built from a bundle folder that contains the underlying model, a json description
 * of the model's input and output layers, and any additional assets required by the model, for
 * example, output labels.
 *
 * A conforming `TIOModel` begins by parsing a json description of the model's input and output
 * layers, producing a `TIOLayerInterface` for each layer. Each layer is fully described by a
 * conforming `TIOLayerDescription`, which describes the data the layer expects or produces, for
 * example, whether it is quantized, any transformations that should be applied to it, and the
 * number of bytes the layer expects.
 *
 * To perform inference with the underlying model, call `runOn:` with a conforming `TIOData` object.
 * `TIOData` objects simply know how to copy bytes to and receive bytes from a model's input
 * and output layers. Internally, this method matches `TIOData` objects with their corresponding
 * layers and ensures that bytes are copied to the right place. The `runOn:` method then returns a
 * conforming `TIOData` object, which is the result of performing inference with the model.
 * Objects that conform to the `TIOData` protocol include `NSNumber`, `NSArray`, `NSData`,
 * `NSDictionary`, and `TIOPixelBuffer`, which wraps a `CVPixelBuffer` for computer vision models.
 *
 * For more information about a model's interface, refer to the `TIOLayerInterface` and
 * `TIOLayerDescription` classes. For more information about the kinds of Objective-C data a
 * `TIOModel` can work with, refer to the `TIOData` protocol and its conforming classes. For more
 * information about the JSON file which describes a model, see TIOModelBundleJSONSchema.h
 *
 * Note that, currently, only TensorFlow Lite (TFLite) models are supported.
 *
 * @warning
 * Models are not thread safe. Models may be used on separate threads, so that you can perform
 * inference off the main thread, but you should not use the same model from multiple threads.
 */

public abstract class TIOModel {

    /**
     * The application or activity context
     */

    private final Context context;

    /**
     * The `TIOModelBundle` object from which this model was instantiated.
     */

    private TIOModelBundle bundle;

    /**
     * Options associated with this model.
     */

    private TIOModelOptions options;

    /**
     * Modes associated with the model, e.g. whether it has support for prediction, training, and evaluation
     */

    private TIOModelModes modes;

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
     * A placeholder bundle has no underlying model and instantiates a `TIOModel` that does nothing.
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

    private TIOModelIO io;

    /**
     * The designated initializer for conforming classes.
     *
     * You should not need to call this method directly. Instead, acquire an instance of a `TIOModelBundle`
     * associated with this model by way of the model's identifier. Then the `TIOModelBundle` class
     * calls this `initWithBundle:` factory initialization method, which conforming classes may override
     * to support custom initialization.
     *
     * @param context The application or activity context
     * @param bundle `TIOModelBundle` containing information about the model and its path
     * @return instancetype An instance of the conforming class, may be `nil`.
     */

    public TIOModel(Context context, TIOModelBundle bundle) {
        this.context = context;
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

    public Context getContext() {
        return context;
    }

    public TIOModelBundle getBundle() {
        return bundle;
    }

    public TIOModelOptions getOptions() {
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

    public TIOModelIO getIO() {
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
     * @@throws TIOModelException
     */

    public void load() throws TIOModelException {
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
     * @param input an array of floats
     * @return results of running the model mapped from the output layer names to the values
     * @throws TIOModelException
     */

    public abstract Map<String, Object> runOn(float[] input) throws TIOModelException;

    /**
     * Perform inference on an array of bytes for a single input layer.
     * @param input an array of bytes
     * @return results of running the model mapped from the output layer names to the values
     * @throws TIOModelException
     */

    public abstract Map<String, Object> runOn(byte[] input) throws TIOModelException;

    /**
     * Perform inference on a Bitmap for a single input layer.
     * @param input A Bitmap
     * @return results of running the model mapped from the output layer names to the values
     * @throws TIOModelException
     */

    public abstract Map<String, Object> runOn(Bitmap input) throws TIOModelException;

    /**
     * Perform inference on an map of bytes
     * @param input A mapping of layer names to arbitrary objects
     * @return results of running the model mapped from the output layer names to the values
     * @throws TIOModelException
     */

    public abstract Map<String, Object> runOn(Map<String, Object> input) throws TIOModelException;

    //endRegion

    //region Input Validation

    protected void validateInput(float[] input) throws TIOModelException {
        if (io.getInputs().size() != 1) {
            throw TIOModelException.InputCountMismatchException(1, io.getInputs().size());
        }
    }

    protected void validateInput(byte[] input) throws TIOModelException {
        if (io.getInputs().size() != 1) {
            throw TIOModelException.InputCountMismatchException(1, io.getInputs().size());
        }
    }

    protected void validateInput(Bitmap input) throws TIOModelException {
        if (io.getInputs().size() != 1) {
            throw TIOModelException.InputCountMismatchException(1, io.getInputs().size());
        }
    }

    protected void validateInput(Map<String, Object> input) throws TIOModelException {
        int expectedSize = io.getInputs().size();
        int actualSize = input.size();

        if (expectedSize != actualSize) {
            throw TIOModelException.InputCountMismatchException(actualSize, expectedSize);
        }

        if ( !input.keySet().equals(io.getInputs().keys()) ) {
            for (TIOLayerInterface layer : io.getInputs().all()) {
                if ( !input.containsKey(layer.getName()) ) {
                    throw TIOModelException.MissingInput(layer.getName());
                }
            }
        }
    }

    //endRegion

    //region Utilities

    @NonNull
    @Override
    public String toString() {
        return "TIOModel{" +
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

}
