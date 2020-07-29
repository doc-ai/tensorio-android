package ai.doc.tensorio.core.modelbundle;

import android.content.Context;

import org.everit.json.schema.Schema;

import org.everit.json.schema.ValidationException;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.util.function.BiFunction;

import ai.doc.tensorio.core.utilities.FileIO;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

public class FileModelBundleValidator extends ModelBundleValidator {

    private File file;
    private File jsonFile;

    /**
     * Initializes the Model Bundle Validator with a File
     *
     * Model validation is done with json-schema-validator library using the JSON Schema Internet draft Version 4.
     * More info about the schema draft can be found at https://tools.ietf.org/html/draft-zyp-json-schema-04.
     *
     * @param context Context used to acquire the TFLite/model-schema.json
     * @param file Fully qualified File pointing to the model bundle
     */

    public FileModelBundleValidator(@NonNull Context context, @NonNull File file) {
        super(context);

        this.file = file;
        this.jsonFile = new File(file, ModelBundle.TFMODEL_INFO_FILE);
    }

    /**
     * Validates the Model Bundle with a custom validator
     *
     * @param customValidator A user supplied validation lambda that will be called after all other
     *                        validation is complete and which allows the user to apply additional
     *                        custom validation. The first String parameter is the relative
     *                        path to the model bundle provided to the validator and the second
     *                        JSONObject parameter is the model JSON.
     * @throws ValidatorException On any validation error. See the message for more details
     */

    public void validate(@Nullable BiFunction<String, JSONObject, Boolean> customValidator) throws ValidatorException {
        validateFile(customValidator);
    }

    /** Validates a model bundle at a fully qualified File location */

    private void validateFile(@Nullable BiFunction<String, JSONObject, Boolean> customValidator) throws ValidatorException {
        // Validate Path

        if (!file.exists()) {
            throw new ValidatorException("No model bundle at found at provided path");
        }

        // Validate Bundle Structure

        // Extension

        if ( !(file.getPath().endsWith(ModelBundle.TIO_BUNDLE_EXTENSION) || file.getPath().endsWith(ModelBundle.TF_BUNDLE_EXTENSION)) ) {
            throw new ValidatorException("No model bundle found with .tiobundle or .tfbundle extension at provided path");
        }

        // model.json

        if (!jsonFile.exists()) {
            throw new ValidatorException("No model.json at found in provided bundle");
        }

        // Validate if JSON Can Be Read

        String json = loadFileJson();
        JSONObject jsonObject = null;

        if (json == null) {
            throw new ValidatorException("Unable to read model.json");
        }

        try {
            jsonObject = new JSONObject(json);
        } catch (JSONException e) {
            throw new ValidatorException("Unable to load model.json", e);
        }

        // Acquire Backend

        String backend = "TFLite";

        // Validate JSON with Schema

        Schema schema = jsonSchemaForBackend(backend);

        if (schema == null) {
            throw new ValidatorException("Unable to acquire model.json schema");
        }

        try {
            schema.validate(jsonObject);
        } catch (ValidationException e) {
            throw new ValidatorException("model.json validation failed", e);
        }

        // Validate Assets

        try {
            validateFileModelAssets(jsonObject);
        } catch (Exception e) {
            throw new ValidatorException("Unable to locate model file in bundle or other named assets");
        }

        // Custom Validator

        if (customValidator != null && !customValidator.apply(file.getName(), jsonObject)) {
            throw new ValidatorException("Custom validator failed");
        }
    }

    /** Validates presence of model file if not placeholder model and any other model assets for a context.asset source */

    private void validateFileModelAssets(JSONObject jsonObject) throws JSONException, Exception {
        if ( !(jsonObject.has("placeholder") && jsonObject.getBoolean("placeholder")) ) {
            String modelFilename = jsonObject.getJSONObject("model").getString("file");
            File model = new File(file, modelFilename);

            if (!model.exists()) {
                throw new Exception();
            }
        }

        JSONArray outputs = jsonObject.getJSONArray("outputs");

        for (int i = 0; i < outputs.length(); i++) {
            JSONObject output = outputs.getJSONObject(i);
            if (!output.has("labels")) {
                continue;
            }

            String labelsFilename = output.getString("labels");
            File assetsDir = new File(file, ModelBundle.TFMODEL_ASSETS_DIRECTORY);
            File asset = new File(assetsDir, labelsFilename);

            if (!asset.exists()) {
                throw new Exception();
            }
        }
    }

    /** Loads the model JSON from a file source and returns null if unable */

    protected @Nullable String loadFileJson() {
        try {
            return FileIO.readTextFile(jsonFile);
        } catch (IOException e) {
            return null;
        }
    }
}
