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

    public static boolean ValidateTFLite(Context context, String json) throws IOException, ProcessingException {
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
