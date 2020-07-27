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

package ai.doc.tensorio.core.modelbundle;

import android.content.Context;

import ai.doc.tensorio.core.model.Model;
import ai.doc.tensorio.core.model.IO;
import ai.doc.tensorio.core.model.Modes;
import ai.doc.tensorio.core.model.Options;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import ai.doc.tensorio.core.utilities.AndroidAssets;
import ai.doc.tensorio.core.utilities.FileIO;
import ai.doc.tensorio.core.layerinterface.LayerInterface;
import ai.doc.tensorio.tflite.model.TFLiteModel;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.List;


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

public class ModelBundle {

    /** Source is an asset from a context or a file. Barf */

    public enum Source {
        Asset,
        File
    };

    private Source source;

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
     * The application or activity context
     */

    private final Context context;

    /**
     * The deserialized information contained in the model.json file.
     */

    private JSONObject info;

    /**
     * The filename of the model bundle folder for a context.assets source. See also modelFilename
     */

    private @Nullable String filename;

    /**
     * The File corresponding to the model bundle folder when using a fully qualified path. See also modelFile
     */

    private @Nullable File file;

    /**
     * The file path to the actual underlying model contained in this bundle when using a
     * context.source model bundle. See also filename.
     *
     * Currently, only tflite models are supported. If `placeholder` is `true` this property
     * returns `null`.
     */

    private @Nullable String modelFilename;

    /**
     * The File corresponding to the actual underlying model contained in this bundle when using a
     * context.source model bundle. See also file.
     *
     * Currently, only tflite models are supported. If `placeholder` is `true` this property
     * returns `null`.
     */

    private @Nullable File modelFile;

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

    private Options options;

    /**
     * Modes associated with the model, e.g. whether it has support for prediction, training, and evaluation
     */

    private Modes modes;

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
     * The class name of the @see TIOModel that should be used to implement this network.
     */

    private String modelClassName;



    // TODO: Must also be able to initialize from a File that is not in context.getAssets

    /**
     * One of two designated initializers, responsible for parsing a bundle's model.json and especially
     * for setting up the description of a model's inputs and outputs.
     *
     * @param f The File pointing to this model bundle with a fully qualified filepath
     * @throws ModelBundleException On any failure to read the model bundle
     */

    public ModelBundle(@NonNull File f) throws ModelBundleException {
        this.source = Source.File;

        this.file = f;

        this.context = null;
        this.filename = filename;
        this.modelFilename = null;

        initBundle();
    }

    /**
     * One of two designated initializers, responsible for parsing a bundle's model.json and especially
     * for setting up the description of a model's inputs and outputs.
     *
     * @param context The application or activity context
     * @param filename Filename or path to the model bundle folder as a context.assets source
     * @throws ModelBundleException On any failure to read the model bundle
     */

    public ModelBundle(@NonNull Context context, @NonNull String filename) throws ModelBundleException {
        this.source = Source.Asset;

        this.context = context;
        this.filename = filename;

        this.file = null;
        this.modelFile = null;

        initBundle();
    }

    private void initBundle() throws ModelBundleException {
        String json = null;
        JSONObject bundle;

        try {
            // Barf
            switch (source) {
                case Asset:
                    json = AndroidAssets.readTextFile(context, filename + "/" + TFMODEL_INFO_FILE);
                    break;
                case File:
                    json = FileIO.readTextFile(new File(file, TFMODEL_INFO_FILE));
                    break;
            }
        } catch (IOException e) {
            throw new ModelBundleException("Error reading model file", e);
        }

        try {
            bundle = new JSONObject(json);
        } catch (JSONException e) {
            throw new ModelBundleException("Error parsing model file as JSON", e);
        }

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
            this.modelClassName = modelJsonObject.optString("class", TFLiteModel.class.getName());
            this.placeholder = modelJsonObject.optBoolean("placeholder", false);

            if (modelJsonObject.has("modes")) {
                this.modes = new Modes(modelJsonObject.getJSONArray("modes"));
            } else {
                this.modes = new Modes();
            }

            // Determine model file filename

            if (!this.placeholder) {
                String n = modelJsonObject.getString("file");
                // So barf
                switch (source) {
                    case Asset:
                        this.modelFilename = filename + "/" + n;
                        this.modelFile = null;
                        break;
                    case File:
                        this.modelFile = new File(file, n);
                        this.modelFilename = null;
                        break;
                }
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

    public Source getSource() {
        return source;
    }

    public Context getContext() {
        return context;
    }

    public @Nullable String getFilename() {
        return filename;
    }

    public @Nullable String getModelFilename() {
        return modelFilename;
    }

    public @Nullable File getFile() {
        return file;
    }

    public @Nullable File getModelFile() {
        return modelFile;
    }

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
     * @return a new instance of the TIOModel represented by this bundle.
     */

    public Model newModel() throws ModelBundleException {
        try {
            return (Model) Class.forName(modelClassName).getConstructor(ModelBundle.class).newInstance( this);
        } catch (IllegalAccessException | InstantiationException | InvocationTargetException | NoSuchMethodException | ClassNotFoundException e) {
            throw new ModelBundleException("Error creating TIOModel", e);
        }
    }

    /**
     * Returns the path to an asset in the bundle, used for a context.asset source
     *
     * @param assetName Asset’s filename, including extension
     * @return The relative path to the file
     */

    public String pathToAsset(String assetName) {
        return filename + "/" + TFMODEL_ASSETS_DIRECTORY + "/" + assetName;
    }

    /**
     * Returns the File to an asset in the bundle, used for a File source
     *
     * @param assetName Asset’s filename, including extension
     * @return The file for the asset
     */

    public File fileToAsset(String assetName) {
        return new File(new File(file, TFMODEL_ASSETS_DIRECTORY), assetName);
    }

    @NonNull @Override
    public String toString() {
        String fname = getSource() == Source.Asset
                ? ", filename='" + filename + '\''
                : ", file='" + file.getPath() + '\'';
        String modelfname = getSource() == Source.Asset
                ? ", filename='" + modelFilename + '\''
                : ", file='" + modelFile.getPath() + '\'';

        return "TIOModelBundle{" +
                "info='" + info + '\'' +
                fname +
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
                modelfname +
                ", modelClassName='" + modelClassName + '\'' +
                '}';
    }

}
