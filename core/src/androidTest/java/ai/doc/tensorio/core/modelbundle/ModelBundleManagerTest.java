/*
 * ModelBundleManagerTest.java
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

package ai.doc.tensorio.core.modelbundle;

import android.content.Context;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystemException;
import java.util.Set;

import ai.doc.tensorio.core.utilities.AndroidAssets;
import androidx.test.platform.app.InstrumentationRegistry;

import static org.junit.Assert.*;

public class ModelBundleManagerTest {

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

        AndroidAssets.copyAsset(testContext, filename, file);
        return file;
    }

    @Test
    public void testLoadsModelBundlesInAssetsDirectory() {
        try {
            ModelBundleManager modelBundleManager = ModelBundleManager.managerWithAssets(testContext, "");
            Set<String> ids = modelBundleManager.getBundleIds();

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
            ModelBundleManager modelBundleManager = ModelBundleManager.managerWithFiles(modelsDir);
            Set<String> ids = modelBundleManager.getBundleIds();

            assertEquals(ids.size(), NUM_VALID_MODELS);
            
        } catch (IOException e) {
            fail();
        }
    }
}