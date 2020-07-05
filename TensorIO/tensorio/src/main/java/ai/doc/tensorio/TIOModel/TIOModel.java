package ai.doc.tensorio.TIOModel;

import android.content.Context;
import android.support.annotation.NonNull;

import java.util.Map;

import ai.doc.tensorio.TIOLayerInterface.TIOLayerInterface;

/**
 * This is the primary API provided by the TensorIO framework.
 * <p>
 * A TIOModel is built from a bundle folder that contains the underlying model, a json description of the model’s input and output layers, and any additional assets required by the model, for example, output labels.
 * <p>
 * A conforming TIOModel begins by parsing a json description of the model’s input and output layers, producing a TIOLayerInterface for each layer. Each layer is fully described by a conforming TIOLayerDescription, which describes the data the layer expects or produces, for example, whether it is quantized, any transformations that should be applied to it, and the number of bytes the layer expects.
 * <p>
 * To perform inference with the underlying model, call runOn: with a conforming TIOData object. TIOData objects simply know how to copy bytes to and receive bytes from a model’s input and output layers. Internally, this method matches TIOData objects with their corresponding layers and ensures that bytes are copied to the right place. The runOn: method then returns a conforming TIOData object, which is the result of performing inference with the model. Objects that conform to the TIOData protocol include NSNumber, NSArray, NSData, NSDictionary, and TIOPixelBuffer, which wraps a CVPixelBuffer for computer vision models.
 * <p>
 * For more information about a model’s interface, refer to the TIOLayerInterface and TIOLayerDescription classes. For more information about the kinds of Objective-C data a TIOModel can work with, refer to the TIOData protocol and its conforming classes.
 * <p>
 * Note that, currently, only TensorFlow Lite (TFLite) models are supported.
 * <p>
 * WARNING: Models are not thread safe. Models may be used on separate threads, so that you can perform inference off the main thread, but you should not use the same model from multiple threads.
 */
public abstract class TIOModel {

    private TIOModelBundle bundle;

    /**
     * Options associated with this model.
     */
    private TIOModelOptions options;

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

    private boolean loaded;

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

    private Context context;

    /**
     * The designated initializer for conforming classes.
     * <p>
     * You should not need to call this method directly. Instead, acquire an instance of a `TIOModelBundle`
     * associated with this model by way of the model's identifier. Then the `TIOModelBundle` class
     * calls this `initWithBundle:` factory initialization method, which conforming classes may override
     * to support custom initialization.
     *
     * @param bundle `TIOModelBundle` containing information about the model and its path
     */
    public TIOModel(Context context, TIOModelBundle bundle) {
        this.context = context;
        this.bundle = bundle;

        this.options = bundle.getOptions();
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

    /**
     * Loads a model into memory.
     * <p>
     * A model should load itself prior to running on any input, but consumers of the model may want
     * more control over when a model is loaded in order to avoid placing parameters into memory
     * before they are needed.
     * <p>
     */
    public void load() throws TIOModelException {
        loaded = true;
    }

    /**
     * Unloads a model from memory
     * <p>
     * A model will unload its resources automatically when it is deallocated, but the unload function
     * may do this as well in order to provide finer grained control to consumers.
     * <p>
     */
    public void unload() {
        loaded = false;
    }

    /**
     * Performs inference on the provided input and returns the results. The primary interface to a
     * conforming class.
     */
    public Object runOn(Object input) throws TIOModelException {

        if (input instanceof Map) {
            Map<String, Object> inputMap = (Map<String, Object>) input;
            if (io.getInputs().size() != inputMap.size()) {
                throw new TIOModelException("The model has " + io.getInputs().size() + " input layers but received " + inputMap.size() + " inputs");
            }

            if (!inputMap.keySet().equals(io.getInputs().keys())) {
                for (TIOLayerInterface layer : io.getInputs().all()) {
                    if (!inputMap.containsKey(layer.getName())) {
                        throw new TIOModelException("The model received no input for layer \"" + layer.getName() + "\"");
                    }
                }
            }
        } else {
            if (io.getInputs().size() != 1) {
                throw new TIOModelException("The model has " + io.getInputs().size() + " input layers but only received one input");
            }
        }

        return null;
    }

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

    public Context getContext() {
        return context;
    }
}
