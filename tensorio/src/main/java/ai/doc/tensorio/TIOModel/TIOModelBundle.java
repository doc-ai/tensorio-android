/*
 * TIOModelBundle.java
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
import androidx.annotation.NonNull;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.List;

import ai.doc.tensorio.TIOLayerInterface.TIOLayerInterface;
import ai.doc.tensorio.TIOTFLiteModel.TIOTFLiteModel;
import ai.doc.tensorio.TIOUtilities.FileIO;

/**
 * Encapsulates information about a `TIOModel` without actually loading the model.
 *
 * A `TIOModelBundle` is used by the UI to show model details and is used to instantiate model
 * instances as a model factory. There is currently a one-to-one correspondence between a
 * `TIOModelBundle` and a .tiobundle folder in the models directory.
 *
 * A model bundle folder must contain at least a model.json file, which contains information
 * about the model. Some information is required, such as the identifier and name field,
 * while other information may be added as needed by your use case.
 */

public class TIOModelBundle {

    /**
     * The name of the file inside a TensorIO bundle that contains the model spec, currently 'model.json'.
     */

    private static final String TFMODEL_INFO_FILE = "model.json";

    /**
     * The name of the directory inside a TensorIO bundle that contains additional data, currently 'assets'.
     */

    private static final String TFMODEL_ASSETS_DIRECTORY = "assets";

    /**
     * The directory extension for TF bundles, considered deprecated, using .tiobundle instead
     */

    public static final String TF_BUNDLE_EXTENSION = ".tfbundle";

    /**
     * The directory extension for TIO bundles
     */

    public static final String TIO_BUNDLE_EXTENSION = ".tiobundle";

    /**
     * The application or activity context
     */

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
     * The version of the model represented by this bundle.
     *
     * A model's unique identifier may remain the same as the version is incremented.
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
     * A boolean value indicating if this is a placeholder bundle.
     *
     * A placeholder bundle has no underlying model and instantiates a `TIOModel` that does nothing.
     * Placeholders bundles are used to collect labeled data for models that haven't been trained yet.
     */

    private boolean placeholder;

    /**
     * A boolean value indicating if the model represnted by this bundle is quantized or not.
     */

    private boolean quantized;

    /**
     * A string indicating the kind of model this is, e.g. "image.classification.imagenet"
     */

    private String type;

    /**
     * Options associated with the model represented by this bundle.
     */

    private TIOModelOptions options;

    /**
     * Modes associated with the model, e.g. whether it has support for prediction, training, and evaluation
     */

    private TIOModelModes modes;

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
     *
     * Currently, only tflite models are supported. If this `placeholder` is `YES` this property
     * returns `nil`.
     */

    private String modelFilePath;

    /**
     * The class name of the @see TIOModel that should be used to implement this network.
     */

    private String modelClassName;

    /**
     * The designated initializer, responsible for parsing a bundle's model.json and especially
     * for setting up the description of a model's inputs and outputs.
     * @param context The application or activity context
     * @param path Fully qualified path to the model bundle folder.
     * @throws TIOModelBundleException
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

        // Parse basic top level properties

        try {
            this.identifier = bundle.getString("id");
            this.name = bundle.getString("name");
            this.version = bundle.getString("version");
            this.details = bundle.getString("details");
            this.author = bundle.getString("author");
            this.license = bundle.getString("license");
        } catch (JSONException e) {
            throw new TIOModelBundleException("Incomplete JSON model file", e);
        }

        // Parse optional options

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

        // Parse model properties

        try {
            JSONObject modelJsonObject = bundle.getJSONObject("model");

            this.quantized = modelJsonObject.getBoolean("quantized");
            this.type = modelJsonObject.optString("type", "unknown");
            this.modelClassName = modelJsonObject.optString("class", TIOTFLiteModel.class.getName());
            this.placeholder = modelJsonObject.optBoolean("placeholder", false);

            if (modelJsonObject.has("modes")) {
                this.modes = new TIOModelModes(modelJsonObject.getJSONArray("modes"));
            } else {
                this.modes = new TIOModelModes();
            }

            // Determine model file path

            if (!this.placeholder) {
                this.modelFilePath = path + "/" + modelJsonObject.getString("file");
            }

        } catch (JSONException e) {
            throw new TIOModelBundleException("Incomplete JSON model file", e);
        }

        // Parse Inputs and Outputs

        List<TIOLayerInterface> inputs;
        List<TIOLayerInterface> outputs;

        try {
            inputs = TIOModelJSONParsing.parseIO(this, bundle.getJSONArray("inputs"), TIOLayerInterface.Mode.Input);
        } catch (JSONException e) {
            throw new TIOModelBundleException("Error parsing inputs field", e);
        } catch (IOException e) {
            throw new TIOModelBundleException("Error reading labels file", e);
        }

        try {
            outputs = TIOModelJSONParsing.parseIO(this, bundle.getJSONArray("outputs"), TIOLayerInterface.Mode.Output);
        } catch (JSONException e) {
            throw new TIOModelBundleException("Error parsing outputs field", e);
        } catch (IOException e) {
            throw new TIOModelBundleException("Error reading labels file", e);
        }

        // Parse Placeholders, may be null

        List<TIOLayerInterface> placeholders = null;

        if ( bundle.has("placeholders") ) {

            try {
                placeholders = TIOModelJSONParsing.parseIO(this, bundle.getJSONArray("placeholders"), TIOLayerInterface.Mode.Placeholder);
            } catch (JSONException e) {
                throw new TIOModelBundleException("Error parsing outputs field", e);
            } catch (IOException e) {
                throw new TIOModelBundleException("Error reading labels file", e);
            }

        }

        this.io = new TIOModelIO(inputs, outputs, placeholders);
    }

    //region Getters and Setters

    public Context getContext() {
        return context;
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

    public TIOModelModes getModes() {
        return modes;
    }

    public TIOModelIO getIO() {
        return io;
    }

    public String getModelFilePath() {
        return modelFilePath;
    }

    //endregion

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

}
