package ai.doc.tensorio.core.modelbundle;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;

import ai.doc.tensorio.core.utilities.FileIO;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

public class FileModelBundle extends ModelBundle {

    /**
     * The File corresponding to the model bundle folder.
     */

    private File file;

    /**
     * The File corresponding to the actual underlying model contained in this bundle.
     *
     * Currently, only tflite models are supported. If `placeholder` is `true` this property
     * returns `null`.
     */

     private @Nullable File modelFile;

    /**
     * The designated initializer for a Model Bundle initialized with a File. Responsible for
     * parsing a bundle's model.json and especially for setting up the description of a model's
     * inputs and outputs.
     *
     * @param f The File pointing to this model bundle with a fully qualified filepath
     * @throws ModelBundleException On any failure to read the model bundle
     */

    public FileModelBundle(@NonNull File f) throws ModelBundleException {
        this.file = f;

        try {
            String json = FileIO.readTextFile(new File(file, TFMODEL_INFO_FILE));
            JSONObject bundle = new JSONObject(json);
            initBundle(bundle);

            if (!this.placeholder) {
                String n = bundle.getJSONObject("model").getString("file");
                this.modelFile = new File(file, n);
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
        return FileIO.readTextFile(fileToAsset(filename));
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

    // Getters and Setters

    public File getFile() {
        return file;
    }

    public @Nullable File getModelFile() {
        return modelFile;
    }

    // Utilities

    @NonNull @Override
    public String toString() {
        return "ModelBundle{" +
                "info='" + info + '\'' +
                ", file='" + file.getPath() + '\'' +
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
                ", model file='" + modelFile.getPath() + '\'' +
                ", modelClassName='" + modelClassName + '\'' +
                '}';
    }
}
