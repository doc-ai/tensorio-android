/*
 * TIOModelBundleValidator.java
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

import com.fasterxml.jackson.databind.JsonNode;
import com.github.fge.jackson.JsonLoader;
import com.github.fge.jsonschema.core.exceptions.ProcessingException;
import com.github.fge.jsonschema.core.report.ProcessingReport;
import com.github.fge.jsonschema.main.JsonSchema;
import com.github.fge.jsonschema.main.JsonSchemaFactory;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.util.Arrays;
import java.util.function.BiFunction;

import ai.doc.tensorio.TIOUtilities.FileIO;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

// TODO: Must also be able to validate from a File that is not in context.getAssets

public class TIOModelBundleValidator {

    /** Used to validate model bundles in an assets directory */

    private Context context;
    private String filename;

    private String jsonPath;

    /**
     * Initializes the Model Bundle Validator
     *
     * Model validation is done with json-schema-validator library using the JSON Schema Internet draft Version 4.
     * More info about the schema draft can be found at https://tools.ietf.org/html/draft-zyp-json-schema-04.
     *
     * @param context Context used to acquire the TFLite/model-schema.json
     * @param filename The model bundle's filename relative to the assets directory
     */

    public TIOModelBundleValidator(@NonNull Context context, @NonNull String filename) {
        this.context = context;
        this.filename = filename;
        this.jsonPath = filename + "/" + TIOModelBundle.TFMODEL_INFO_FILE;
    }

    /**
     * Validates the Model Bundle without a custom validator.
     *
     * @throws TIOModelBundleValidatorException On any validation error. See the message for more details
     */

    public void validate() throws TIOModelBundleValidatorException {
        validate(null);
    }

    /**
     * Validates the Model Bundle with a custom validator
     *
     * @param customValidator A user supplied validation lambda that will be called after all other
     *                        validation is complete and which allows the user to apply additional
     *                        custom validation. The first String parameter is the relative
     *                        path to the model bundle provided to the validator and the second
     *                        JSONObject parameter is the model JSON.
     * @throws TIOModelBundleValidatorException On any validation error. See the message for more details
     */

    public void validate(@Nullable  BiFunction<String, JSONObject, Boolean> customValidator) throws TIOModelBundleValidatorException {

        // Validate Path

        try {
            if (!Arrays.asList(context.getResources().getAssets().list("")).contains(filename)) {
                throw new TIOModelBundleValidatorException("No model bundle at found at provided path");
            }
        } catch (Exception e) {
            throw new TIOModelBundleValidatorException("No model bundle at provided path");
        }

        // Validate Bundle Structure

        // Extension

        if ( !(filename.endsWith(TIOModelBundle.TIO_BUNDLE_EXTENSION) || filename.endsWith(TIOModelBundle.TF_BUNDLE_EXTENSION)) ) {
            throw new TIOModelBundleValidatorException("No model bundle found with .tiobundle or .tfbundle extension at provided path");
        }

        // model.json

        try {
            if (!Arrays.asList(context.getResources().getAssets().list(filename)).contains(TIOModelBundle.TFMODEL_INFO_FILE)) {
                throw new TIOModelBundleValidatorException("No model bundle at found at provided path");
            }
        } catch (Exception e) {
            throw new TIOModelBundleValidatorException("No model bundle at provided path");
        }

        // Validate if JSON Can Be Read

        String json = loadJson();
        JSONObject jsonObject = null;
        JsonNode jsonNode = null;

        if (json == null) {
            throw new TIOModelBundleValidatorException("Unable to read model.json");
        }

        try {
            jsonObject = new JSONObject(json);
            jsonNode = JsonLoader.fromString(json);
        } catch (JSONException | IOException e) {
            throw new TIOModelBundleValidatorException("Unable to load model.json", e);
        }

        // Acquire Backend

        String backend = "TFLite";

        // Validate JSON with Schema

        JsonSchema schema = jsonSchemaForBackend(backend);

        if (schema == null) {
            throw new TIOModelBundleValidatorException("Unable to acquire model.json schema");
        }

        try {
            ProcessingReport report = schema.validate(jsonNode);
            if (!report.isSuccess()) {
                throw new TIOModelBundleValidatorException("model.json validation failed");
            }
        } catch (ProcessingException e) {
            throw new TIOModelBundleValidatorException("Unknown error encounted during model.json validation", e);
        }

        // Validate Assets

        try {
            validateAssets(jsonObject);
        } catch (Exception e) {
            throw new TIOModelBundleValidatorException("Unable to locate model file in bundle or other named assets");
        }

        // Custom Validator

        if (customValidator != null && !customValidator.apply(filename, jsonObject)) {
            throw new TIOModelBundleValidatorException("Custom validator failed");
        }
    }

    /** Validates presence of model file if not placeholder model and any other assets */

    private void validateAssets(JSONObject jsonObject) throws JSONException, Exception {
        if ( !(jsonObject.has("placeholder") && jsonObject.getBoolean("placeholder")) ) {
            String modelFilename = jsonObject.getJSONObject("model").getString("file");
            if (!Arrays.asList(context.getResources().getAssets().list(filename)).contains(modelFilename)) {
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
            String assetPath = filename + "/" + TIOModelBundle.TFMODEL_ASSETS_DIRECTORY;

            if (!Arrays.asList(context.getResources().getAssets().list(assetPath)).contains(labelsFilename)) {
                throw new Exception();
            }
        }
    }

    /** Loads the model JSON and returns null if unable */

    private @Nullable String loadJson() {
        try {
            return FileIO.readFile(context, jsonPath);
        } catch (IOException e) {
            return null;
        }
    }

    /** Loads the JSON schema and returns null if unable */

    private @Nullable JsonSchema jsonSchemaForBackend(@NonNull String backend){
        try {
            String modelSchema = FileIO.readFile(context, backend + "/model-schema.json");
            JsonNode schemaNode = JsonLoader.fromString(modelSchema);
            JsonSchemaFactory factory = JsonSchemaFactory.byDefault();
            JsonSchema schema = factory.getJsonSchema(schemaNode);
            return schema;
        } catch (IOException | ProcessingException e) {
            return null;
        }
    }
}
