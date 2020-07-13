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

import java.io.IOException;
import java.io.InputStream;

import androidx.annotation.NonNull;

public abstract class TIOModelBundleValidator {

    /**
     * Validates the string representation of the json with the TFLite json schema file in the assets folder at TFLite/mode-schema.json.
     * The validation is done with json-schema-validator library using the JSON Schema Internet draft Version 4.
     * More info about the schema draft can be found at https://tools.ietf.org/html/draft-zyp-json-schema-04.
     *
     * @param context Context object. The context is needed to get access to assets folder where the mode-schema.json file is stored.
     * @param json The string representation of the json object to validate.
     * @return if the provided json is validated against the schema.
     * @throws IOException IOException is thrown if the TFLite/model-schema.json file is not found inside the assets folder.
     * @throws ProcessingException ProcessingException is thrown if the JsonSchemaFactory cannot get the JsonSchema from the schemaNode.
     */

    public static boolean ValidateTFLite(@NonNull Context context, @NonNull String json) throws IOException, ProcessingException {
        InputStream inputStream = context.getAssets().open("TFLite/model-schema.json");
        int size = inputStream.available();
        byte[] buffer = new byte[size];
        inputStream.read(buffer);
        inputStream.close();
        String modelSchema = new String(buffer, "UTF-8");

        JsonNode schemaNode = JsonLoader.fromString(modelSchema);
        JsonNode data = JsonLoader.fromString(json);
        JsonSchemaFactory factory = JsonSchemaFactory.byDefault();
        JsonSchema schema = factory.getJsonSchema(schemaNode);
        ProcessingReport report = schema.validate(data);
        return report.isSuccess();
    }
}
