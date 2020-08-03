/*
 * ModelBundleValidatorTest.java
 * TensorIO
 *
 * Created by Philip Dow on 7/17/2020
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

package ai.doc.tensorio.tflite.modelbundle;

import android.content.Context;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystemException;

import ai.doc.tensorio.core.modelbundle.ModelBundleValidator;
import ai.doc.tensorio.core.modelbundle.ModelBundleValidator.ValidatorException;
import ai.doc.tensorio.core.utilities.AndroidAssets;
import androidx.test.platform.app.InstrumentationRegistry;

import static org.junit.Assert.*;

public class ModelBundleValidatorTest {
    private Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
    private Context testContext = InstrumentationRegistry.getInstrumentation().getContext();

    /** Set up a models directory to copy assets to for testing */

    @Before
    public void setUp() throws Exception {
        File f = new File(testContext.getFilesDir(), "models");
        if (!f.mkdirs()) {
            throw new FileSystemException("on create: " + f.getPath());
        }
    }

    /** Tear down the models directory */

    @After
    public void tearDown() throws Exception {
        File f = new File(testContext.getFilesDir(), "models");
        deleteRecursive(f);
    }

    /** Create an assets source validator from an asset */

    private ModelBundleValidator validatorForFilename(String filename) {
        return ModelBundleValidator.validatorWithAsset(testContext, filename);
    }

    /** Create a file source validator from a file, copying the asset to models */

    private ModelBundleValidator validatorForFile(String filename) throws IOException {
        File dir = new File(testContext.getFilesDir(), "models");
        File file = new File(dir, filename);

        AndroidAssets.copyAsset(testContext, filename, file);
        return ModelBundleValidator.validatorWithFile(testContext, file);
    }

   /** Delete a directory and all its contents */

    private void deleteRecursive(File f) throws FileSystemException {
        if (f.isDirectory())
            for (File child : f.listFiles())
                deleteRecursive(child);

        if (!f.delete()) {
            throw new FileSystemException("on delete: " + f.getPath());
        }
    }

    // ASSETS DIRECTORY

    // Path and Bundle Validation

    @Test
    public void testBundleAtInvalidPathDoesNotValidate_assets() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFilename("foo-foo.tiobundle");
            modelBundleValidator.validate();
            fail();
        } catch (ValidatorException e) {
        }
    }

    @Test
    public void testBundleWithoutTIOBundleExtensionDoesNotValidate_assets() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFilename("invalid-model-no-ext");
            modelBundleValidator.validate();
            fail();
        } catch (ValidatorException e) {
        }
    }

    @Test
    public void testBundleWithoutJSONDoesNotValidate_assets() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFilename("invalid-model-no-json.tiobundle");
            modelBundleValidator.validate();
            fail();
        } catch (ValidatorException e) {
        }
    }

    @Test
    public void testBundleWithBadJSONDoesNotValidate_assets() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFilename("invalid-model-bad-json.tiobundle");
            modelBundleValidator.validate();
            fail();
        } catch (ValidatorException e) {
        }
    }

    @Test
    public void testBundleWithDeprecatedExtensionStillValidates_assets() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFilename("deprecated.tfbundle");
            modelBundleValidator.validate();
        } catch (ValidatorException e) {
            fail();
        }
    }

    // Assets Validation

    @Test
    public void testAnIncorrectlyNamedModelFileDoesNotValidate_assets() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFilename("invalid-model-incorrect-model-file.tiobundle");
            modelBundleValidator.validate();
            fail();
        } catch (ValidatorException e) {
        }
    }

    @Test
    public void testAnIncorrectNamedLabelsFileDoesNotValidate_assets() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFilename("invalid-model-incorrect-labels-file.tiobundle");
            modelBundleValidator.validate();
            fail();
        } catch (ValidatorException e) {
        }
    }

    // Custom Validation

    @Test
    public void testInvalidCustomValidationDoesNotValidate_assets() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFilename("1_in_1_out_number_test.tiobundle");
            modelBundleValidator.validate((path, jsonObject) -> {
                return false;
            });
            fail();
        } catch (ValidatorException e) {
        }
    }

    @Test
    public void testValidCustomValidatorValidates_assets() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFilename("1_in_1_out_number_test.tiobundle");
            modelBundleValidator.validate((path, jsonObject) -> {
                return true;
            });
        } catch (ValidatorException e) {
            fail();
        }
    }

    // Valid Model Bundles

    @Test
    public void testValidModelsValidate_assets() {
        ModelBundleValidator modelBundleValidator = null;

        try {
            modelBundleValidator = validatorForFilename("1_in_1_out_number_test.tiobundle");
            modelBundleValidator.validate();

            modelBundleValidator = validatorForFilename("1_in_1_out_pixelbuffer_identity_test.tiobundle");
            modelBundleValidator.validate();

            modelBundleValidator = validatorForFilename("1_in_1_out_pixelbuffer_normalization_test.tiobundle");
            modelBundleValidator.validate();

            modelBundleValidator = validatorForFilename("1_in_1_out_pixelbuffer_test.tiobundle");
            modelBundleValidator.validate();

            modelBundleValidator = validatorForFilename("1_in_1_out_tensors_test.tiobundle");
            modelBundleValidator.validate();

            modelBundleValidator = validatorForFilename("1_in_1_out_vectors_test.tiobundle");
            modelBundleValidator.validate();

            modelBundleValidator = validatorForFilename("2_in_2_out_matrices_test.tiobundle");
            modelBundleValidator.validate();

            modelBundleValidator = validatorForFilename("2_in_2_out_vectors_test.tiobundle");
            modelBundleValidator.validate();

            modelBundleValidator = validatorForFilename("mobilenet_v1_1.0_224_quant.tiobundle");
            modelBundleValidator.validate();

            modelBundleValidator = validatorForFilename("mobilenet_v2_1.4_224.tiobundle");
            modelBundleValidator.validate();
        } catch (ValidatorException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testModelWithoutBackendValidates_assets() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFilename("no-backend.tiobundle");
            modelBundleValidator.validate();
        } catch (ValidatorException e) {
            fail();
        }
    }

    @Test
    public void testModelWithoutModesValidates_assets() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFilename("no-modes.tiobundle");
            modelBundleValidator.validate();
        } catch (ValidatorException e) {
            fail();
        }
    }

    @Test
    public void testPlaceholderModelValidates_assets() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFilename("placeholder.tiobundle");
            modelBundleValidator.validate();
        } catch (ValidatorException e) {
            fail();
        }
    }

    // FILE

    // Path and Bundle Validation

    @Test
    public void testBundleAtInvalidPathDoesNotValidate_file() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFile("foo-foo.tiobundle");
            modelBundleValidator.validate();
            fail();
        } catch (IOException e) {
        } catch (ValidatorException e) {
        }
    }

    @Test
    public void testBundleWithoutTIOBundleExtensionDoesNotValidate_file() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFile("invalid-model-no-ext");
            modelBundleValidator.validate();
            fail();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (ValidatorException e) {
        }
    }

    @Test
    public void testBundleWithoutJSONDoesNotValidate_file() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFile("invalid-model-no-json.tiobundle");
            modelBundleValidator.validate();
            fail();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (ValidatorException e) {
        }
    }

    @Test
    public void testBundleWithBadJSONDoesNotValidate_file() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFile("invalid-model-bad-json.tiobundle");
            modelBundleValidator.validate();
            fail();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (ValidatorException e) {
        }
    }

    @Test
    public void testBundleWithDeprecatedExtensionStillValidates_file() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFile("deprecated.tfbundle");
            modelBundleValidator.validate();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (ValidatorException e) {
            fail();
        }
    }

    // Assets Validation

    @Test
    public void testAnIncorrectlyNamedModelFileDoesNotValidate_file() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFile("invalid-model-incorrect-model-file.tiobundle");
            modelBundleValidator.validate();
            fail();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (ValidatorException e) {
        }
    }

    @Test
    public void testAnIncorrectNamedLabelsFileDoesNotValidate_file() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFile("invalid-model-incorrect-labels-file.tiobundle");
            modelBundleValidator.validate();
            fail();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (ValidatorException e) {
        }
    }

    // Custom Validation

    @Test
    public void testInvalidCustomValidationDoesNotValidate_file() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFile("1_in_1_out_number_test.tiobundle");
            modelBundleValidator.validate((path, jsonObject) -> {
                return false;
            });
            fail();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (ValidatorException e) {
        }
    }

    @Test
    public void testValidCustomValidatorValidates_file() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFile("1_in_1_out_number_test.tiobundle");
            modelBundleValidator.validate((path, jsonObject) -> {
                return true;
            });
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (ValidatorException e) {
            fail();
        }
    }

    // Valid Model Bundles

    @Test
    public void testValidModelsValidate_file() {

        ModelBundleValidator modelBundleValidator = null;

        try {
            modelBundleValidator = validatorForFile("1_in_1_out_number_test.tiobundle");
            modelBundleValidator.validate();

            modelBundleValidator = validatorForFile("1_in_1_out_pixelbuffer_identity_test.tiobundle");
            modelBundleValidator.validate();

            modelBundleValidator = validatorForFile("1_in_1_out_pixelbuffer_normalization_test.tiobundle");
            modelBundleValidator.validate();

            modelBundleValidator = validatorForFile("1_in_1_out_pixelbuffer_test.tiobundle");
            modelBundleValidator.validate();

            modelBundleValidator = validatorForFile("1_in_1_out_tensors_test.tiobundle");
            modelBundleValidator.validate();

            modelBundleValidator = validatorForFile("1_in_1_out_vectors_test.tiobundle");
            modelBundleValidator.validate();

            modelBundleValidator = validatorForFile("2_in_2_out_matrices_test.tiobundle");
            modelBundleValidator.validate();

            modelBundleValidator = validatorForFile("2_in_2_out_vectors_test.tiobundle");
            modelBundleValidator.validate();

            modelBundleValidator = validatorForFile("mobilenet_v1_1.0_224_quant.tiobundle");
            modelBundleValidator.validate();

            modelBundleValidator = validatorForFile("mobilenet_v2_1.4_224.tiobundle");
            modelBundleValidator.validate();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (ValidatorException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testModelWithoutBackendValidates_file() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFile("no-backend.tiobundle");
            modelBundleValidator.validate();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (ValidatorException e) {
            fail();
        }
    }

    @Test
    public void testModelWithoutModesValidates_file() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFile("no-modes.tiobundle");
            modelBundleValidator.validate();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (ValidatorException e) {
            fail();
        }
    }

    @Test
    public void testPlaceholderModelValidates_file() {
        try {
            ModelBundleValidator modelBundleValidator = validatorForFile("placeholder.tiobundle");
            modelBundleValidator.validate();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (ValidatorException e) {
            fail();
        }
    }
}
