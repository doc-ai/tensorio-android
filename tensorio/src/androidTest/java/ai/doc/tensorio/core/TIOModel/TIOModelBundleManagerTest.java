package ai.doc.tensorio.core.TIOModel;

import android.content.Context;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystemException;
import java.util.Set;

import ai.doc.tensorio.core.TIOUtilities.TIOAndroidAssets;
import androidx.test.platform.app.InstrumentationRegistry;

import static org.junit.Assert.*;

public class TIOModelBundleManagerTest {

    private static final int NUM_VALID_MODELS = 13;

    private static final String[] VALID_MODELS = {
            "1_in_1_out_number_test.tiobundle",
            "1_in_1_out_pixelbuffer_identity_test.tiobundle",
            "1_in_1_out_pixelbuffer_normalization_test.tiobundle",
            "1_in_1_out_pixelbuffer_test.tiobundle",
            "1_in_1_out_tensors_test.tiobundle",
            "1_in_1_out_vectors_test.tiobundle",
            "2_in_2_out_matrices_test.tiobundle",
            "2_in_2_out_vectors_test.tiobundle",
            "mobilenet_v1_1.0_224_quant.tiobundle",
            "mobilenet_v2_1.4_224.tiobundle",
            "no-backend.tiobundle",
            "no-modes.tiobundle",
            "placeholder.tiobundle"
    };

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

    /** Delete a directory and all its contents */

    private void deleteRecursive(File f) throws FileSystemException {
        if (f.isDirectory())
            for (File child : f.listFiles())
                deleteRecursive(child);

        if (!f.delete()) {
            throw new FileSystemException("on delete: " + f.getPath());
        }
    }

    /** Copies a model from assets to a filesDir location and returns the File */

    private File copyModelToFiles(String filename) throws IOException {
        File dir = new File(testContext.getFilesDir(), "models");
        File file = new File(dir, filename);

        TIOAndroidAssets.copyAsset(testContext, filename, file);
        return file;
    }

    @Test
    public void testLoadsModelBundlesInAssetsDirectory() {
        try {
            TIOModelBundleManager manager = new TIOModelBundleManager(testContext, "");
            Set<String> ids = manager.getBundleIds();

            assertEquals(ids.size(), NUM_VALID_MODELS);

        } catch (IOException e) {
            fail();
        }
    }

    @Test
    public void testLoadsModelBundlesInFileDirectory() {
        try {
            for (String filename : VALID_MODELS) {
                copyModelToFiles(filename);
            }
        } catch (IOException e) {
            fail();
        }
        try {
            File modelsDir = new File(testContext.getFilesDir(), "models");
            TIOModelBundleManager manager = new TIOModelBundleManager(modelsDir);
            Set<String> ids = manager.getBundleIds();

            assertEquals(ids.size(), NUM_VALID_MODELS);
            
        } catch (IOException e) {
            fail();
        }
    }
}