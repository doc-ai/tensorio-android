/*
 * ModelBundle.java
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

package ai.doc.tensorio.core.modelbundle;

import android.content.Context;

import ai.doc.tensorio.core.model.Model;
import ai.doc.tensorio.core.model.IO;
import ai.doc.tensorio.core.model.Modes;
import ai.doc.tensorio.core.model.Options;
import androidx.annotation.NonNull;

import ai.doc.tensorio.core.layerinterface.LayerInterface;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.List;


/**
 * Encapsulates information about a `Model` without actually loading the model.
 *
 * A `ModelBundle` is used by the UI to show model details and is used to instantiate model
 * instances as a model factory. There is currently a one-to-one correspondence between a
 * `ModelBundle` and a .tiobundle folder in the models directory.
 *
 * A model bundle folder must contain at least a model.json file, which contains information
 * about the model. Some information is required, such as the identifier and name field,
 * while other information may be added as needed by your use case.
 *
 * This is an abstract class. Use one of the static methods @see bundleWithAsset or
 * @see bundleWithFile to get a concrete instance from a package asset or File.
 */

public abstract class ModelBundle {

    private static final String TF_LITE_MODEL_CLASS_NAME = "ai.doc.tensorio.tflite.model.TFLiteModel";

    public static class ModelBundleException extends Exception {
        public ModelBundleException(@NonNull String message, @NonNull Throwable cause) {
            super(message, cause);
        }

        public ModelBundleException(@NonNull String message) {
            super(message);
        }
    }

    /**
     * The name of the file inside a TensorIO bundle that contains the model spec, currently 'model.json'.
     */

    public static final String TFMODEL_INFO_FILE = "model.json";

    /**
     * The name of the directory inside a TensorIO bundle that contains additional data, currently 'assets'.
     */

    public static final String TFMODEL_ASSETS_DIRECTORY = "assets";

    /**
     * The directory extension for TF bundles, considered deprecated, using .tiobundle instead
     */

    public static final String TF_BUNDLE_EXTENSION = ".tfbundle";

    /**
     * The directory extension for TIO bundles
     */

    public static final String TIO_BUNDLE_EXTENSION = ".tiobundle";

    /**
     * Creates and returns a new ModelBundle from an asset
     *
     * @param context The application or activity context
     * @param filename The filename of the model bundle in the assets, including subdirectories
     * @return An @see AssetModelBundle
     * @throws ModelBundleException On any problem reading the ModelBundle
     */

    public static ModelBundle bundleWithAsset(@NonNull Context context, @NonNull String filename) throws ModelBundleException {
        return new AssetModelBundle(context, filename);
    }

    /**
     * Creates and returns a new ModelBundle from a File
     * @param f The File to the model bundle
     * @return An @see FileModelBundle
     * @throws ModelBundleException On any problem reading the ModelBundle
     */

    public static ModelBundle bundleWithFile(@NonNull File f) throws ModelBundleException {
        return new FileModelBundle(f);
    }

    /**
     * The deserialized information contained in the model.json file.
     */

    protected JSONObject info;

    /**
     * A string uniquely identifying the model represented by this bundle.
     */

    protected String identifier;

    /**
     * Human readable name of the model represented by this bundle
     */

    protected String name;

    /**
     * The version of the model represented by this bundle.
     *
     * A model's unique identifier may remain the same as the version is incremented.
     */

    protected String version;

    /**
     * Additional information about the model represented by this bundle.
     */

    protected String details;

    /**
     * The authors of the model represented by this bundle.
     */

    protected String author;

    /**
     * The license of the model represented by this bundle.
     */

    protected String license;

    /**
     * A boolean value indicating if this is a placeholder bundle.
     *
     * A placeholder bundle has no underlying model and instantiates a `Model` that does nothing.
     * Placeholders bundles are used to collect labeled data for models that haven't been trained yet.
     */

    protected boolean placeholder;

    /**
     * A boolean value indicating if the model represented by this bundle is quantized or not.
     */

    protected boolean quantized;

    /**
     * A string indicating the kind of model this is, e.g. "image.classification.imagenet"
     */

    protected String type;

    /**
     * Options associated with the model represented by this bundle.
     */

    protected Options options;

    /**
     * Modes associated with the model, e.g. whether it has support for prediction, training, and evaluation
     */

    protected Modes modes;

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

    protected IO io;

    /**
     * The class name of the @see Model that should be used to implement this network.
     */

    protected String modelClassName;

    /**
     * Initializes the bundle with a JSON representation of the model.json file. Concrete subclasses
     * should call this method after they have read the JSON file.
     *
     * @param bundle A JSON representation of the modle.json file
     * @throws ModelBundleException On any problem parsing the JSON Object
     */

    protected void initBundle(JSONObject bundle) throws ModelBundleException {
        this.info = bundle;

        // Parse basic top level properties

        try {
            this.identifier = bundle.getString("id");
            this.name = bundle.getString("name");
            this.version = bundle.getString("version");
            this.details = bundle.getString("details");
            this.author = bundle.getString("author");
            this.license = bundle.getString("license");
        } catch (JSONException e) {
            throw new ModelBundleException("Incomplete JSON model file", e);
        }

        // Parse optional options

        try {
            if (bundle.has("options")) {
                String devicePosition = bundle.getJSONObject("options").getString("device_position");
                this.options = new Options(devicePosition);
            } else {
                this.options = new Options("back");
            }
        } catch (JSONException e) {
            throw new ModelBundleException("Incomplete options field, expected 'device_position' entry");
        }

        // Parse model properties

        try {
            JSONObject modelJsonObject = bundle.getJSONObject("model");

            this.quantized = modelJsonObject.getBoolean("quantized");
            this.type = modelJsonObject.optString("type", "unknown");
            this.modelClassName = modelJsonObject.optString("class", TF_LITE_MODEL_CLASS_NAME);
            this.placeholder = modelJsonObject.optBoolean("placeholder", false);

            if (modelJsonObject.has("modes")) {
                this.modes = new Modes(modelJsonObject.getJSONArray("modes"));
            } else {
                this.modes = new Modes();
            }
        } catch (JSONException e) {
            throw new ModelBundleException("Incomplete JSON model file", e);
        }

        // Parse Inputs and Outputs

        List<LayerInterface> inputs;
        List<LayerInterface> outputs;

        try {
            inputs = JSONParsing.parseIO(this, bundle.getJSONArray("inputs"), LayerInterface.Mode.Input);
        } catch (JSONException e) {
            throw new ModelBundleException("Error parsing inputs field", e);
        } catch (IOException e) {
            throw new ModelBundleException("Error reading labels file", e);
        }

        try {
            outputs = JSONParsing.parseIO(this, bundle.getJSONArray("outputs"), LayerInterface.Mode.Output);
        } catch (JSONException e) {
            throw new ModelBundleException("Error parsing outputs field", e);
        } catch (IOException e) {
            throw new ModelBundleException("Error reading labels file", e);
        }

        // Parse Placeholders, may be null

        List<LayerInterface> placeholders = null;

        if ( bundle.has("placeholders") ) {

            try {
                placeholders = JSONParsing.parseIO(this, bundle.getJSONArray("placeholders"), LayerInterface.Mode.Placeholder);
            } catch (JSONException e) {
                throw new ModelBundleException("Error parsing outputs field", e);
            } catch (IOException e) {
                throw new ModelBundleException("Error reading labels file", e);
            }

        }

        this.io = new IO(inputs, outputs, placeholders);
    }

    //region Getters and Setters

    public JSONObject getInfo() {
        return info;
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

    public Options getOptions() {
        return options;
    }

    public Modes getModes() {
        return modes;
    }

    public IO getIO() {
        return io;
    }

    //endregion

    /**
     * @return a new instance of the Model represented by this bundle.
     */

    public Model newModel() throws ModelBundleException {
        try {
            return (Model) Class.forName(modelClassName).getConstructor(ModelBundle.class).newInstance( this);
        } catch (IllegalAccessException | InstantiationException | InvocationTargetException | NoSuchMethodException | ClassNotFoundException e) {
            throw new ModelBundleException("Error creating Model", e);
        }
    }

    /**
     * Reads a text file from the model bundle's assets directory and returns its contents
     *
     * @param filename The filename within the model bundle's assets directory
     * @return The String contents of the file
     * @throws IOException On any error reading the file
     */

    public abstract String readTextFile(String filename) throws IOException;

}
