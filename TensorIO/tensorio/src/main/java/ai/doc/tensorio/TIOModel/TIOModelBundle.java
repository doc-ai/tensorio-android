package ai.doc.tensorio.TIOModel;

import android.content.Context;
import android.support.annotation.NonNull;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;

import ai.doc.tensorio.TIOData.TIODataDequantizer;
import ai.doc.tensorio.TIOData.TIODataQuantizer;
import ai.doc.tensorio.TIOData.TIOPixelDenormalizer;
import ai.doc.tensorio.TIOData.TIOPixelNormalizer;
import ai.doc.tensorio.TIOLayerInterface.TIOLayerInterface;
import ai.doc.tensorio.TIOLayerInterface.TIOPixelBufferLayerDescription;
import ai.doc.tensorio.TIOLayerInterface.TIOVectorLayerDescription;
import ai.doc.tensorio.TIOModel.TIOVisionModel.TIOImageVolume;
import ai.doc.tensorio.TIOModel.TIOVisionModel.TIOPixelFormat;
import ai.doc.tensorio.TIOTensorflowLiteModel.TIOTFLiteModel;
import ai.doc.tensorio.utils.FileIO;

/**
 * Encapsulates information about a @see TIOModel without actually loading the model.
 * <p>
 * A TIOModelBundle is used by the UI to show model details and is used to instantiate model instances as a model factory. There is currently a one-to-one correspondence between a TIOModelBundle and a .mobilenet_V1_0.25_128.tfbundle folder in the models directory.
 * <p>
 * A model bundle folder must contain at least a model.json file, which contains information about the model. Some information is required, such as the identifier and name field, while other information may be added as needed by your use case.
 */

public class TIOModelBundle {

    private static final String TFMODEL_INFO_FILE = "model.json";
    private static final String TFMODEL_ASSETS_DIRECTORY = "assets";
    private static final String TENSOR_TYPE_VECTOR = "array";
    private static final String TENSOR_TYPE_IMAGE = "image";
    private final Context context;

    /**
     * The deserialized information contained in the model.json file.
     */
    private String info;

    /**
     * The full path to the model bundle folder.
     */
    private String path;

    /**
     * A string uniquely identifying the model represented by this bundle.
     */
    private String identifier;

    /**
     * Human readable name of the model represented by this bundle
     */
    private String name;

    /**
     * The version of the model reprsented by this bundle.
     * <p>
     * A model’s unique identifier may remain the same as the version is incremented.
     */
    private String version;

    /**
     * Additional information about the model represented by this bundle.
     */
    private String details;

    /**
     * The authors of the model represented by this bundle.
     */
    private String author;

    /**
     * The license of the model represented by this bundle.
     */
    private String license;

    /**
     * boolean value indicating if this is a placeholder bundle.
     * <p>
     * A placeholder bundle has no underlying model and instantiates a @see TIOModel that does nothing. Placeholders bundles are used to collect labeled data for models that haven’t been trained yet.
     */
    private boolean placeholder;

    /**
     * A boolean value indicating if the model represented by this bundle is quantized or not.
     */
    private boolean quantized;

    /**
     * A string indicating the kind of model this is, e.g. image.classification.imagenet
     */
    private String type;

    /**
     * Options associated with the model represented by this bundle.
     */
    private TIOModelOptions options;

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
     * The file path to the actual underlying model contained in this bundle.
     * <p>
     * Currently, only tflite models are supported. If this placeholder is YES this property returns nil.
     */

    private String modelFilePath;

    /**
     * The class name of the @see TIOModel that should be used to implement this network.
     */

    private String modelClassName;

    /**
     * @param path Fully qualified path to the model bundle folder.
     */

    public TIOModelBundle(Context context, String path) throws TIOModelBundleException {
        this.context = context;

        String json;
        JSONObject bundle;

        try {
            json = FileIO.readFile(context, path + "/" + TFMODEL_INFO_FILE);
        } catch (IOException e) {
            throw new TIOModelBundleException("Error reading model file", e);
        }

        try {
            bundle = new JSONObject(json);
        } catch (JSONException e) {
            throw new TIOModelBundleException("Error parsing model file as JSON", e);
        }

        this.path = path;
        this.info = json;

        JSONObject modelJsonObject;

        try {
            this.identifier = bundle.getString("id");
            this.name = bundle.getString("name");
            this.version = bundle.getString("version");
            this.details = bundle.getString("details");
            this.author = bundle.getString("author");
            this.license = bundle.getString("license");

            modelJsonObject = bundle.getJSONObject("model");

            this.quantized = modelJsonObject.getBoolean("quantized");

        } catch (JSONException e) {
            throw new TIOModelBundleException("Incomplete JSON model file", e);
        }

        try {
            if (bundle.has("options")) {
                String devicePosition = bundle.getJSONObject("options").getString("device_position");
                this.options = new TIOModelOptions(devicePosition);
            } else {
                this.options = new TIOModelOptions("back");
            }
        } catch (JSONException e) {
            throw new TIOModelBundleException("Incomplete options field, expected 'device_position' entry");
        }


        this.type = modelJsonObject.optString("type", "unknown");
        this.modelClassName = modelJsonObject.optString("class", TIOTFLiteModel.class.getName());
        this.placeholder = modelJsonObject.optBoolean("placeholder", false);


        if (!this.placeholder) {
            try {
                this.modelFilePath = path + "/" + modelJsonObject.getString("file");
            } catch (JSONException e) {
                throw new TIOModelBundleException("Incomplete JSON model file, could not find model file declaration", e);
            }
        }

        // Parse Inputs and Outputs

        List<TIOLayerInterface> inputs;
        List<TIOLayerInterface> outputs;

        try {
            inputs = parseInputs(context, bundle.getJSONArray("inputs"));
        } catch (JSONException e) {
            throw new TIOModelBundleException("Error parsing inputs field", e);
        } catch (IOException e) {
            throw new TIOModelBundleException("Error reading labels file", e);
        }

        try {
            outputs = parseOutputs(context, bundle.getJSONArray("outputs"));
        } catch (JSONException e) {
            throw new TIOModelBundleException("Error parsing outputs field", e);
        } catch (IOException e) {
            throw new TIOModelBundleException("Error reading labels file", e);
        }

        this.io = new TIOModelIO(inputs, outputs);
    }

    /**
     * @return a new instance of the TIOModel represented by this bundle.
     */

    public TIOModel newModel() throws TIOModelBundleException {
        try {
            return (TIOModel) Class.forName(modelClassName).getConstructor(Context.class, TIOModelBundle.class).newInstance(context, this);
        } catch (IllegalAccessException | InstantiationException | InvocationTargetException | NoSuchMethodException | ClassNotFoundException e) {
            throw new TIOModelBundleException("Error creating TIOModel", e);
        }
    }

    /**
     * Returns the path to an asset in the bundle
     *
     * @param filename Asset’s filename, including extension
     * @return The full path to the file
     */

    public String pathToAsset(String filename) {
        return "assets" + path + "/" + TFMODEL_ASSETS_DIRECTORY + "/" + filename;
    }

    @NonNull
    @Override
    public String toString() {
        return "TIOModelBundle{" +
                "info='" + info + '\'' +
                ", path='" + path + '\'' +
                ", identifier='" + identifier + '\'' +
                ", name='" + name + '\'' +
                ", version='" + version + '\'' +
                ", details='" + details + '\'' +
                ", author='" + author + '\'' +
                ", license='" + license + '\'' +
                ", placeholder=" + placeholder +
                ", quantized=" + quantized +
                ", type='" + type + '\'' +
                ", options=" + options +
                ", modelFilePath='" + modelFilePath + '\'' +
                ", modelClassName='" + modelClassName + '\'' +
                '}';
    }

    public String getPath() {
        return path;
    }

    public String getIdentifier() {
        return identifier;
    }

    public String getName() {
        return name;
    }

    public String getVersion() {
        return version;
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

    public TIOModelOptions getOptions() {
        return options;
    }

    public TIOModelIO getIO() {
        return io;
    }

    public String getModelFilePath() {
        return modelFilePath;
    }


    private List<TIOLayerInterface> parseInputs(Context c, JSONArray inputs) throws JSONException, TIOModelBundleException, IOException {
        ArrayList<TIOLayerInterface> indexedInputInterfaces = new ArrayList<>();
        boolean isQuantized = this.isQuantized();

        for (int i = 0; i < inputs.length(); i++) {
            JSONObject inputObject = inputs.getJSONObject(i);
            String type = inputObject.getString("type");
            String name = inputObject.getString("name");

            TIOLayerInterface tioLayerInterface;

            switch (type) {
                case TENSOR_TYPE_VECTOR:
                    tioLayerInterface = parseTIOVectorDescription(c, inputObject, true, isQuantized, this);
                    break;
                case TENSOR_TYPE_IMAGE:
                    tioLayerInterface = parseTIOPixelBufferDescription(inputObject, true, isQuantized);
                    break;
                default:
                    throw new TIOModelBundleException("Unsupported input layer type: " + type);
            }

            indexedInputInterfaces.add(tioLayerInterface);
        }

        return indexedInputInterfaces;
    }

    private List<TIOLayerInterface> parseOutputs(Context c, JSONArray outputs) throws JSONException, TIOModelBundleException, IOException {
        ArrayList<TIOLayerInterface> indexedOutputInterfaces = new ArrayList<>();
        boolean isQuantized = this.isQuantized();

        for (int i = 0; i < outputs.length(); i++) {
            JSONObject outputObject = outputs.getJSONObject(i);
            String type = outputObject.getString("type");
            String name = outputObject.getString("name");

            TIOLayerInterface tioLayerInterface;

            switch (type) {
                case TENSOR_TYPE_VECTOR:
                    tioLayerInterface = parseTIOVectorDescription(c, outputObject, false, isQuantized, this);
                    break;
                case TENSOR_TYPE_IMAGE:
                    tioLayerInterface = parseTIOPixelBufferDescription(outputObject, false, isQuantized);
                    break;
                default:
                    throw new TIOModelBundleException("Unsupported input layer type: " + type);
            }

            indexedOutputInterfaces.add(tioLayerInterface);
        }

        return indexedOutputInterfaces;
    }

    private TIOLayerInterface parseTIOVectorDescription(Context c, JSONObject dict, boolean isInput, boolean quantized, TIOModelBundle bundle) throws JSONException, TIOModelBundleException, IOException {
        int[] shape = parseIntArray(dict.getJSONArray("shape"));
        String name = dict.getString("name");
        boolean isOutput = !isInput;

        // Labels
        String[] labels = null;
        if (dict.optString("labels", null) != null) {
            try {
                String contents = FileIO.readFile(c, path + "/" + TFMODEL_ASSETS_DIRECTORY + "/" + dict.getString("labels"));
                contents = contents.trim();
                labels = contents.split("\\n");
            }
            catch (IOException e){
                throw new TIOModelBundleException("There was a problem reading the labels file, no labels were loaded", e);
            }
        }

        // Quantization
        TIODataQuantizer quantizer = null;
        if (isInput && dict.has("quantize")) {
            quantizer = TIODataQuantizerForDict(dict.getJSONObject("quantize"));
        }

        // Dequantization
        TIODataDequantizer dequantizer = null;
        if (isOutput && dict.has("dequantize")) {
            dequantizer = TIODataDequantizerForDict(dict.getJSONObject("dequantize"));
        }

        // Interface
        return new TIOLayerInterface(
                name,
                isInput,
                new TIOVectorLayerDescription(shape, labels, quantized, quantizer, dequantizer)
        );
    }

    private TIOLayerInterface parseTIOPixelBufferDescription(JSONObject dict, boolean isInput, boolean quantized) throws TIOModelBundleException, JSONException {

        String name = dict.getString("name");
        boolean isOutput = !isInput;

        // Image Volume
        TIOImageVolume imageVolume;
        try {
            int[] shape = parseIntArray(dict.getJSONArray("shape"));
            imageVolume = TIOImageVolumeForShape(shape);
        } catch (JSONException e) {
            throw new TIOModelBundleException("Expected input.shape array field in model.json, none found", e);
        }

        // Pixel Format
        TIOPixelFormat pixelFormat;
        try {
            pixelFormat = TIOPixelFormatForString(dict.getString("format"));
        } catch (JSONException e) {
            throw new TIOModelBundleException("Expected input.format string in model.json, none found", e);
        }


        // Normalization
        TIOPixelNormalizer normalizer = null;
        if (isInput && dict.has("normalize")) {
                normalizer = TIOPixelNormalizerForDictionary(dict.getJSONObject("normalize"));
        }

        // Denormalization

        TIOPixelDenormalizer denormalizer = null;

        if (isOutput && dict.has("denormalize")) {
            denormalizer = TIOPixelDenormalizerForDictionary(dict.getJSONObject("denormalize"));
        }

        // Description
        TIOLayerInterface layerInterface = new TIOLayerInterface(
                name,
                isInput,
                new TIOPixelBufferLayerDescription(
                        pixelFormat,
                        imageVolume,
                        normalizer,
                        denormalizer,
                        quantized
                )
        );

        return layerInterface ;
    }



    private TIOPixelFormat TIOPixelFormatForString(String format) throws TIOModelBundleException {
        switch (format) {
            case "RGB":
                return TIOPixelFormat.RGB;
            case "BGR":
                return TIOPixelFormat.BGR;
        }
        throw new TIOModelBundleException("Expected dict.format string to be RGB or BGR in model.json, found " + format);
    }

    private TIODataQuantizer TIODataQuantizerForDict(JSONObject dict) throws JSONException, TIOModelBundleException {
        if (dict == null) {
            return null;
        }

        String standard = dict.optString("standard", null);

        if (standard != null) {
            switch (standard) {
                case "[0,1]":
                    return TIODataQuantizer.TIODataQuantizerZeroToOne();
                case "[-1,1]":
                    return TIODataQuantizer.TIODataQuantizerNegativeOneToOne();
                default:
                    throw new TIOModelBundleException("Invalid Quantizer, expected standard quantization to be [0,1] or [1,1]");
            }
        } else {
            if (dict.has("scale") && dict.has("bias")) {
                float scale = (float) dict.getDouble("scale");
                float bias = (float) dict.getDouble("bias");
                return TIODataQuantizer.TIODataQuantizerWithQuantization(scale, bias);
            } else {
                throw new TIOModelBundleException("Invalid Quantizer, expected scale and bias for quantizer");
            }
        }
    }

    private TIODataDequantizer TIODataDequantizerForDict(JSONObject dict) throws TIOModelBundleException, JSONException {
        if (dict == null) {
            return null;
        }

        String standard = dict.optString("standard", null);

        if (standard != null) {
            switch (standard) {
                case "[0,1]":
                    return TIODataDequantizer.TIODataDequantizerZeroToOne();
                case "[-1,1]":
                    return TIODataDequantizer.TIODataDequantizerNegativeOneToOne();
                default:
                    throw new TIOModelBundleException("Invalid Dequantizer, expected standard dequantization to be [0,1] or [1,1]");
            }
        } else {
            if (dict.has("scale") && dict.has("bias")) {
                float scale = (float) dict.getDouble("scale");
                float bias = (float) dict.getDouble("bias");
                return TIODataDequantizer.TIODataDequantizerWithDequantization(scale, bias);
            } else {
                throw new TIOModelBundleException("Invalid Dequantizer, expected scale and bias for quantizer");
            }
        }
    }


    private int[] parseIntArray(JSONArray a) throws JSONException {
        int[] result = new int[a.length()];
        for (int i = 0; i < a.length(); i++) {
            result[i] = a.getInt(i);
        }
        return result;
    }

    private TIOImageVolume TIOImageVolumeForShape(int[] shape) throws TIOModelBundleException {
        if (shape.length != 3) {
            throw new TIOModelBundleException("Expected shape with three elements, actual count is " + shape.length);
        }
        if (shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0) {
            throw new TIOModelBundleException("Invalid image input shape, shape elements can not be <= 0");
        }
        return new TIOImageVolume(shape[0], shape[1], shape[2]);
    }

    /**
     * Returns the TIOPixelNormalizer given an input dictionary.
     */

    private TIOPixelNormalizer TIOPixelNormalizerForDictionary(JSONObject dict) throws TIOModelBundleException {
        if (dict == null) {
            return null;
        }

        String normalizerString = dict.optString("standard", null);

        if (normalizerString != null) {
            switch (normalizerString) {
                case "[0,1]":
                    return TIOPixelNormalizer.TIOPixelNormalizerZeroToOne();
                case "[-1,1]":
                    return TIOPixelNormalizer.TIOPixelNormalizerNegativeOneToOne();
                default:
                    throw new TIOModelBundleException("Expected input.normalizer string to be '[0,1]' or '[-1,1]', actual value is " + normalizerString);
            }
        } else if (dict.has("scale") || dict.has("bias")) {
            float scale = (float)dict.optDouble("scale", 1.0);
            float redBias = (float)dict.optDouble("r", 0.0);
            float greenBias = (float)dict.optDouble("g", 0.0);
            float blueBias = (float)dict.optDouble("b", 0.0);
            return TIOPixelNormalizer.TIOPixelNormalizerPerChannelBias(scale, redBias, greenBias, blueBias);
        } else {
            return null;
        }
    }

    private TIOPixelDenormalizer TIOPixelDenormalizerForDictionary(JSONObject dict) throws TIOModelBundleException {
        if (dict == null) {
            return null;
        }

        String normalizerString = dict.optString("standard", null);

        if (normalizerString != null) {
            switch (normalizerString) {
                case "[0,1]":
                    return TIOPixelDenormalizer.TIOPixelDenormalizerZeroToOne();
                case "[-1,1]":
                    return TIOPixelDenormalizer.TIOPixelDenormalizerNegativeOneToOne();
                default:
                    throw new TIOModelBundleException("Expected input.denormalizer string to be '[0,1]' or '[-1,1]', actual value is " + normalizerString);
            }
        } else if (dict.has("scale") || dict.has("bias")) {
            float scale = (float)dict.optDouble("scale", 1.0);
            float redBias = (float)dict.optDouble("r", 0.0);
            float greenBias = (float)dict.optDouble("g", 0.0);
            float blueBias = (float)dict.optDouble("b", 0.0);
            return TIOPixelDenormalizer.TIOPixelDenormalizerPerChannelBias(scale, redBias, greenBias, blueBias);
        } else {
            return null;
        }
    }

}
