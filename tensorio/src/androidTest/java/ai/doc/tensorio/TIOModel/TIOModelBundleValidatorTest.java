package ai.doc.tensorio.TIOModel;

import android.content.Context;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.nio.file.FileSystemException;

import androidx.test.platform.app.InstrumentationRegistry;

import static org.junit.Assert.*;

public class TIOModelBundleValidatorTest {
    private Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
    private Context testContext = InstrumentationRegistry.getInstrumentation().getContext();

//    @Before
//    public void setUp() throws Exception {
//        File f = new File(testContext.getFilesDir(), "models");
//        if (!f.mkdirs()) {
//            throw new FileSystemException("on create: " + f.getPath());
//        }
//    }
//
//    @After
//    public void tearDown() throws Exception {
//        File f = new File(testContext.getFilesDir(), "models");
//        deleteRecursive(f);
//    }

    private TIOModelBundleValidator validatorForFilename(String filename) {
        return new TIOModelBundleValidator(testContext, filename);
    }

    // Path and Bundle Validation

    @Test
    public void testBundleAtInvalidPathDoesNotValidate() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("foo-foo.tiobundle");
            validator.validate();
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    @Test
    public void testBundleWithoutTIOBundleExtensionDoesNotValidate() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("invalid-model-ext");
            validator.validate();
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    @Test
    public void testBundleWithoutJSONDoesNotValidate() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("invalid-model-no-json.tiobundle");
            validator.validate();
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    @Test
    public void testBundleWithBadJSONDoesNotValidate() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("invalid-model-bad-json.tiobundle");
            validator.validate();
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    @Test
    public void testBundleWithDeprecatedExtensionStillValidates() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("deprecated.tfbundle");
            validator.validate();
        } catch (TIOModelBundleValidatorException e) {
            fail();
        }
    }

    // Assets Validation

    @Test
    public void testAnIncorrectlyNamedModelFileDoesNotValidate() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("invalid-model-incorrect-model-file.tiobundle");
            validator.validate();
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    @Test
    public void testAnIncorrectNamedLabelsFileDoesNotValidate() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("invalid-model-incorrect-labels-file.tiobundle");
            validator.validate();
            fail();
        } catch (TIOModelBundleValidatorException e) {
        }
    }

    // Custom Validation

    @Test
    public void testInvalidCustomValidationDoesNotValidate() {
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
    public void testValidCustomValidatorValidates() {
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
    public void testValidModelsValidate() {
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
    public void testModelWithoutBackendValidates() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("no-backend.tiobundle");
            validator.validate();
        } catch (TIOModelBundleValidatorException e) {
            fail();
        }
    }

    @Test
    public void testModelWithoutModesValidates() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("no-modes.tiobundle");
            validator.validate();
        } catch (TIOModelBundleValidatorException e) {
            fail();
        }
    }

    @Test
    public void testPlaceholderModelValidates() {
        try {
            TIOModelBundleValidator validator = validatorForFilename("placeholder.tiobundle");
            validator.validate();
        } catch (TIOModelBundleValidatorException e) {
            fail();
        }
    }

    // Utility Methods for Copying Assets to Data Directory

    void deleteRecursive(File f) throws FileSystemException {
        if (f.isDirectory())
            for (File child : f.listFiles())
                deleteRecursive(child);

        if (!f.delete()) {
            throw new FileSystemException("on delete: " + f.getPath());
        }
    }



}