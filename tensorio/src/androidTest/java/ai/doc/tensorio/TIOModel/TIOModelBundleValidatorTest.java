package ai.doc.tensorio.TIOModel;

import android.content.Context;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystemException;

import ai.doc.tensorio.TIOUtilities.TIOAndroidAssets;
import androidx.test.platform.app.InstrumentationRegistry;

import static org.junit.Assert.*;

public class TIOModelBundleValidatorTest {
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

    private TIOModelBundleValidator validatorForFilename(String filename) {
        return new TIOModelBundleValidator(testContext, filename);
    }

    /** Create a file source validator from an asset, copying the asset to models */

    private TIOModelBundleValidator validatorForFile(String filename) throws IOException {
        File dir = new File(testContext.getFilesDir(), "models");
        File file = new File(dir, filename);

        TIOAndroidAssets.copyAsset(testContext, filename, file);
        return new TIOModelBundleValidator(testContext, file);
    }

    // ASSETS DIRECTORY

    // Path and Bundle Validation

    @Test
    public void testBundleAtInvalidPathDoesNotValidate_assets() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("foo-foo.tiobundle");
            validator.validate();
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    @Test
    public void testBundleWithoutTIOBundleExtensionDoesNotValidate_assets() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("invalid-model-no-ext");
            validator.validate();
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    @Test
    public void testBundleWithoutJSONDoesNotValidate_assets() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("invalid-model-no-json.tiobundle");
            validator.validate();
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    @Test
    public void testBundleWithBadJSONDoesNotValidate_assets() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("invalid-model-bad-json.tiobundle");
            validator.validate();
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    @Test
    public void testBundleWithDeprecatedExtensionStillValidates_assets() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("deprecated.tfbundle");
            validator.validate();
        } catch (TIOModelBundleValidatorException e) {
            fail();
        }
    }

    // Assets Validation

    @Test
    public void testAnIncorrectlyNamedModelFileDoesNotValidate_assets() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("invalid-model-incorrect-model-file.tiobundle");
            validator.validate();
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    @Test
    public void testAnIncorrectNamedLabelsFileDoesNotValidate_assets() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("invalid-model-incorrect-labels-file.tiobundle");
            validator.validate();
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    // Custom Validation

    @Test
    public void testInvalidCustomValidationDoesNotValidate_assets() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("1_in_1_out_number_test.tiobundle");
            validator.validate((path, jsonObject) -> {
                return false;
            });
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    @Test
    public void testValidCustomValidatorValidates_assets() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("1_in_1_out_number_test.tiobundle");
            validator.validate((path, jsonObject) -> {
                return true;
            });
        } catch (TIOModelBundleValidatorException e) {
            fail();
        }
    }

    // Valid Model Bundles

    @Test
    public void testValidModelsValidate_assets() {
        TIOModelBundleValidator validator = null;

        try {
            validator = validatorForFilename("1_in_1_out_number_test.tiobundle");
            validator.validate();

            validator = validatorForFilename("1_in_1_out_pixelbuffer_identity_test.tiobundle");
            validator.validate();

            validator = validatorForFilename("1_in_1_out_pixelbuffer_normalization_test.tiobundle");
            validator.validate();

            validator = validatorForFilename("1_in_1_out_pixelbuffer_test.tiobundle");
            validator.validate();

            validator = validatorForFilename("1_in_1_out_tensors_test.tiobundle");
            validator.validate();

            validator = validatorForFilename("1_in_1_out_vectors_test.tiobundle");
            validator.validate();

            validator = validatorForFilename("2_in_2_out_matrices_test.tiobundle");
            validator.validate();

            validator = validatorForFilename("2_in_2_out_vectors_test.tiobundle");
            validator.validate();

            validator = validatorForFilename("mobilenet_v1_1.0_224_quant.tiobundle");
            validator.validate();

            validator = validatorForFilename("mobilenet_v2_1.4_224.tiobundle");
            validator.validate();
        } catch (TIOModelBundleValidatorException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testModelWithoutBackendValidates_assets() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("no-backend.tiobundle");
            validator.validate();
        } catch (TIOModelBundleValidatorException e) {
            fail();
        }
    }

    @Test
    public void testModelWithoutModesValidates_assets() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("no-modes.tiobundle");
            validator.validate();
        } catch (TIOModelBundleValidatorException e) {
            fail();
        }
    }

    @Test
    public void testPlaceholderModelValidates_assets() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("placeholder.tiobundle");
            validator.validate();
        } catch (TIOModelBundleValidatorException e) {
            fail();
        }
    }

    // FILE

    // Path and Bundle Validation

    @Test
    public void testBundleAtInvalidPathDoesNotValidate_file() {
        try {
            TIOModelBundleValidator validator = validatorForFile("foo-foo.tiobundle");
            validator.validate();
            fail();
        } catch (IOException e) {
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    @Test
    public void testBundleWithoutTIOBundleExtensionDoesNotValidate_file() {
        try {
            TIOModelBundleValidator validator = validatorForFile("invalid-model-no-ext");
            validator.validate();
            fail();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    @Test
    public void testBundleWithoutJSONDoesNotValidate_file() {
        try {
            TIOModelBundleValidator validator = validatorForFile("invalid-model-no-json.tiobundle");
            validator.validate();
            fail();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    @Test
    public void testBundleWithBadJSONDoesNotValidate_file() {
        try {
            TIOModelBundleValidator validator = validatorForFile("invalid-model-bad-json.tiobundle");
            validator.validate();
            fail();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    @Test
    public void testBundleWithDeprecatedExtensionStillValidates_file() {
        try {
            TIOModelBundleValidator validator = validatorForFile("deprecated.tfbundle");
            validator.validate();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (TIOModelBundleValidatorException e) {
            fail();
        }
    }

    // Assets Validation

    @Test
    public void testAnIncorrectlyNamedModelFileDoesNotValidate_file() {
        try {
            TIOModelBundleValidator validator = validatorForFile("invalid-model-incorrect-model-file.tiobundle");
            validator.validate();
            fail();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    @Test
    public void testAnIncorrectNamedLabelsFileDoesNotValidate_file() {
        try {
            TIOModelBundleValidator validator = validatorForFile("invalid-model-incorrect-labels-file.tiobundle");
            validator.validate();
            fail();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    // Custom Validation

    @Test
    public void testInvalidCustomValidationDoesNotValidate_file() {
        try {
            TIOModelBundleValidator validator = validatorForFile("1_in_1_out_number_test.tiobundle");
            validator.validate((path, jsonObject) -> {
                return false;
            });
            fail();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    @Test
    public void testValidCustomValidatorValidates_file() {
        try {
            TIOModelBundleValidator validator = validatorForFile("1_in_1_out_number_test.tiobundle");
            validator.validate((path, jsonObject) -> {
                return true;
            });
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (TIOModelBundleValidatorException e) {
            fail();
        }
    }

    // Valid Model Bundles

    @Test
    public void testValidModelsValidate_file() {

        TIOModelBundleValidator validator = null;

        try {
            validator = validatorForFile("1_in_1_out_number_test.tiobundle");
            validator.validate();

            validator = validatorForFile("1_in_1_out_pixelbuffer_identity_test.tiobundle");
            validator.validate();

            validator = validatorForFile("1_in_1_out_pixelbuffer_normalization_test.tiobundle");
            validator.validate();

            validator = validatorForFile("1_in_1_out_pixelbuffer_test.tiobundle");
            validator.validate();

            validator = validatorForFile("1_in_1_out_tensors_test.tiobundle");
            validator.validate();

            validator = validatorForFile("1_in_1_out_vectors_test.tiobundle");
            validator.validate();

            validator = validatorForFile("2_in_2_out_matrices_test.tiobundle");
            validator.validate();

            validator = validatorForFile("2_in_2_out_vectors_test.tiobundle");
            validator.validate();

            validator = validatorForFile("mobilenet_v1_1.0_224_quant.tiobundle");
            validator.validate();

            validator = validatorForFile("mobilenet_v2_1.4_224.tiobundle");
            validator.validate();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (TIOModelBundleValidatorException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testModelWithoutBackendValidates_file() {
        try {
            TIOModelBundleValidator validator = validatorForFile("no-backend.tiobundle");
            validator.validate();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (TIOModelBundleValidatorException e) {
            fail();
        }
    }

    @Test
    public void testModelWithoutModesValidates_file() {
        try {
            TIOModelBundleValidator validator = validatorForFile("no-modes.tiobundle");
            validator.validate();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (TIOModelBundleValidatorException e) {
            fail();
        }
    }

    @Test
    public void testPlaceholderModelValidates_file() {
        try {
            TIOModelBundleValidator validator = validatorForFile("placeholder.tiobundle");
            validator.validate();
        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (TIOModelBundleValidatorException e) {
            fail();
        }
    }

    // Utility Methods

    void deleteRecursive(File f) throws FileSystemException {
        if (f.isDirectory())
            for (File child : f.listFiles())
                deleteRecursive(child);

        if (!f.delete()) {
            throw new FileSystemException("on delete: " + f.getPath());
        }
    }

}