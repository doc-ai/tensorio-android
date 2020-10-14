package ai.doc.tensorio.core.modelbundle;

import android.content.Context;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;

import ai.doc.tensorio.core.utilities.AndroidAssets;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

public class AssetModelBundle extends ModelBundle {

    /**
     * The application or activity context
     */

    private final Context context;

    /**
     * The filename of the model bundle in the package's assets directory.
     */

    String filename;

    /**
     * The file path to the actual underlying model contained in this bundle, including the path
     * to the bundle.
     *
     * Currently, only tflite models are supported. If `placeholder` is `true` this property
     * returns `null`.
     */

    private @Nullable String modelFilename;

    /**
     * The designated initializer for a Model Bundle initialized with a packaged Asset. Responsible
     * for parsing a bundle's model.json and especially for setting up the description of a model's
     * inputs and outputs.
     *
     * @param context The application or activity context
     * @param filename Filename or path to the model bundle folder as a context.assets source
     * @throws ModelBundleException On any failure to read the model bundle
     */

    public AssetModelBundle(@NonNull Context context, @NonNull String filename) throws ModelBundleException {
        this.context = context;
        this.filename = filename;

        try {
            String json = AndroidAssets.readTextFile(context, filename + "/" + TFMODEL_INFO_FILE);
            JSONObject bundle = new JSONObject(json);
            initBundle(bundle);

            if (!this.placeholder) {
                String n = bundle.getJSONObject("model").getString("file");
                this.modelFilename = filename + "/" + n;
            }

        } catch (IOException e) {
            throw new ModelBundleException("Error reading model file", e);
        } catch (JSONException e) {
            throw new ModelBundleException("Error parsing model file as JSON", e);
        }
    }

    /**
     * Reads a text file from the model bundle's assets directory and returns its contents
     *
     * @param filename The filename within the model bundle's assets directory
     * @return The String contents of the file
     * @throws IOException On any error reading the file
     */

    public String readTextFile(String filename) throws IOException {
        return AndroidAssets.readTextFile(context, pathToAsset(filename));
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
    // Getters and Setters

    public Context getContext() {
        return context;
    }

    public String getFilename() {
        return filename;
    }

    public @Nullable String getModelFilename() {
        return modelFilename;
    }

    // Utilities

    @NonNull @Override
    public String toString() {
        return "ModelBundle{" +
                "info='" + info + '\'' +
                ", filename='" + filename + '\'' +
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
                ", model filename='" + modelFilename + '\'' +
                ", modelClassName='" + modelClassName + '\'' +
                '}';
    }
}
