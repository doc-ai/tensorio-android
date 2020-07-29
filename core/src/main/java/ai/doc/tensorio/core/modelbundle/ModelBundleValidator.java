/*
 * ModelBundleValidator.java
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

import com.fasterxml.jackson.databind.JsonNode;
import com.github.fge.jackson.JsonLoader;
import com.github.fge.jsonschema.core.exceptions.ProcessingException;
import com.github.fge.jsonschema.main.JsonSchema;
import com.github.fge.jsonschema.main.JsonSchemaFactory;

import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.util.function.BiFunction;

import ai.doc.tensorio.core.utilities.AndroidAssets;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

public abstract class ModelBundleValidator {

    public static class ValidatorException extends Exception {
        public ValidatorException(@NonNull String message) {
            super(message);
        }

        public ValidatorException(@NonNull String message, @NonNull Throwable cause) {
            super(message, cause);
        }
    }

    /**
     * Creates and returns a new AssetModelBundleValidator for the bundle in the assets directory at path
     *
     * @param context The application or activity context
     * @param filename The filename for the model bundle that will be validated, including subdirectories
     * @return A @see AssetModelBundleValidator
     */

    public static ModelBundleValidator validatorWithAsset(@NonNull Context context, @NonNull String filename) {
        return new AssetModelBundleValidator(context, filename);
    }

    /**
     * Creates and returns a new FileModelBundleValidator for the bundle at File
     *
     * @param context The application or activity context
     * @param file The File to the model bundle
     * @return A @see FileModelBundleValidator
     */

    public static ModelBundleValidator validatorWithFile(@NonNull Context context, @NonNull File file) {
        return new FileModelBundleValidator(context, file);
    }

    /**
     * The application or activity context that contains the JSON schema
     */

    protected Context context;

    /**
     * Designated initializer although this is an abstract class so it just sets the context property
     *
     * @param context The application or activity context
     */

    public ModelBundleValidator(Context context) {
        this.context = context;
    }

    /**
     * Validates the Model Bundle without a custom validator.
     *
     * @throws ValidatorException On any validation error. See the message for more details
     */

    public void validate() throws ValidatorException {
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
     * @throws ValidatorException On any validation error. See the message for more details
     */

    public abstract void validate(@Nullable BiFunction<String, JSONObject, Boolean> customValidator) throws ValidatorException;

    /** Loads the JSON schema and returns null if unable */

    protected @Nullable JsonSchema jsonSchemaForBackend(@NonNull String backend){
        try {
            String modelSchema = AndroidAssets.readTextFile(context, backend + "/model-schema.json");
            JsonNode schemaNode = JsonLoader.fromString(modelSchema);
            JsonSchemaFactory factory = JsonSchemaFactory.byDefault();
            JsonSchema schema = factory.getJsonSchema(schemaNode);
            return schema;
        } catch (IOException | ProcessingException e) {
            return null;
        }
    }
}
