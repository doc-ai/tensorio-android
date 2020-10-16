/*
 * TensorFlowModelIntegrationTest.java
 * TensorIO
 *
 * Created by Philip Dow
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

package ai.doc.tensorio.tensorflow.model;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.FileSystemException;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import ai.doc.tensorio.core.model.Model;
import ai.doc.tensorio.core.modelbundle.ModelBundle;
import ai.doc.tensorio.core.utilities.AndroidAssets;
import androidx.test.platform.app.InstrumentationRegistry;

import static org.junit.Assert.*;

public class TensorFlowModelIntegrationTest {

    private Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
    private Context testContext = InstrumentationRegistry.getInstrumentation().getContext();

    private float epsilon = 0.01f;

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

    /** Create a model bundle from a file, copying the asset to models */

    private ModelBundle bundleForFile(String filename) throws IOException, ModelBundle.ModelBundleException {
        File dir = new File(testContext.getFilesDir(), "models");
        File file = new File(dir, filename);

        AndroidAssets.copyAsset(testContext, filename, file);
        return ModelBundle.bundleWithFile(file);
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

    // Single Valued Tests

    @Test
    public void test1x1NumberModel() {
        try {
            // Prepare Model

            ModelBundle tioBundle = bundleForFile("1_in_1_out_number_test.tiobundle");
            assertNotNull(tioBundle);

            Model model = tioBundle.newModel();
            assertNotNull(tioBundle);
            model.load();

            // Prepare Inputs

            float[] input = {2};

            // Run Model

            Map<String, Object> outputs = model.runOn(input);
            assertNotNull(outputs);

            // Check Output

            float[] output = (float[]) outputs.get("output");
            assertNotNull(output);

            float[] expectedOutput = {
                    25
            };

            assertArrayEquals(output, expectedOutput, epsilon);

        } catch (ModelBundle.ModelBundleException | Model.ModelException | IOException e) {
            e.printStackTrace();
            fail();
        }
    }

    // Vectors, Matrix, Tensors Tests

    @Test
    public void test1x1VectorsModel() {
        try {
            // Prepare Model

            ModelBundle tioBundle = bundleForFile("1_in_1_out_vectors_test.tiobundle");
            assertNotNull(tioBundle);

            Model model = tioBundle.newModel();
            assertNotNull(tioBundle);
            model.load();

            // Prepare Inputs

            float[] input = {
                    1, 2, 3, 4
            };

            // Run Model

            Map<String, Object> outputs = model.runOn(input);
            assertNotNull(outputs);

            // Check Output

            float[] output = (float[]) outputs.get("output");
            assertNotNull(output);

            float[] expectedOutput = {
                    2, 2, 4, 4
            };

            assertArrayEquals(output, expectedOutput, epsilon);

        } catch (ModelBundle.ModelBundleException | Model.ModelException | IOException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void test2x2VectorsModel() {
        try {
            // Prepare Model

            ModelBundle tioBundle = bundleForFile("2_in_2_out_vectors_test.tiobundle");
            assertNotNull(tioBundle);

            Model model = tioBundle.newModel();
            assertNotNull(tioBundle);
            model.load();

            // Prepare Inputs

            float[] input1 = {
                    1, 2, 3, 4
            };
            float[] input2 = {
                    10, 20, 30, 40
            };

            Map<String, Object> input = new HashMap<String, Object>();
            input.put("input1", input1);
            input.put("input2", input2);

            // Run Model

            Map<String, Object> outputs = model.runOn(input);
            assertNotNull(outputs);

            // Check Output

            float[] output1 = (float[]) outputs.get("output1");
            assertNotNull(output1);
            float[] output2 = (float[]) outputs.get("output2");
            assertNotNull(output2);

            float[] expectedOutput1 = {
                    240
            };
            float[] expectedOutput2 = {
                    64
            };

            assertArrayEquals(output1, expectedOutput1, epsilon);
            assertArrayEquals(output2, expectedOutput2, epsilon);

        } catch (ModelBundle.ModelBundleException | Model.ModelException | IOException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void test2x2MatricesModel() {
        try {
            // Prepare Model

            ModelBundle tioBundle = bundleForFile("2_in_2_out_matrices_test.tiobundle");
            assertNotNull(tioBundle);

            Model model = tioBundle.newModel();
            assertNotNull(tioBundle);
            model.load();

            // Prepare Inputs

            float[] input1 = {
                    1,    2,    3,    4,
                    10,   20,   30,   40,
                    100,  200,  300,  400,
                    1000, 2000, 3000, 4000
            };
            float[] input2 = {
                    5,    6,    7,    8,
                    50,   60,   70,   80,
                    500,  600,  700,  800,
                    5000, 6000, 7000, 8000
            };

            Map<String, Object> input = new HashMap<String, Object>();
            input.put("input1", input1);
            input.put("input2", input2);

            // Run Model

            Map<String, Object> outputs = model.runOn(input);
            assertNotNull(outputs);

            // Check Output

            float[] output1 = (float[]) outputs.get("output1");
            assertNotNull(output1);
            float[] output2 = (float[]) outputs.get("output2");
            assertNotNull(output2);

            float[] expectedOutput1 = {
                    56,       72,       56,       72,
                    5600,     7200,     5600,     7200,
                    560000,   720000,   560000,   720000,
                    56000000, 72000000, 56000000, 72000000,
            };
            float[] expectedOutput2 = {
                    18,    18,    18,    18,
                    180,   180,   180,   180,
                    1800,  1800,  1800,  1800,
                    18000, 18000, 18000, 18000
            };

            assertArrayEquals(output1, expectedOutput1, epsilon);
            assertArrayEquals(output2, expectedOutput2, epsilon);

        } catch (ModelBundle.ModelBundleException | Model.ModelException | IOException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void test1x1TensorsModel() {
        try {
            // Prepare Model

            ModelBundle tioBundle = bundleForFile("1_in_1_out_tensors_test.tiobundle");
            assertNotNull(tioBundle);

            Model model = tioBundle.newModel();
            assertNotNull(tioBundle);
            model.load();

            // Prepare Inputs

            float[] input = {
                    1,   2,   3,   4,   5,   6,   7,   8,   9,
                    10,  20,  30,  40,  50,  60,  70,  80,  90,
                    100, 200, 300, 400, 500, 600, 700, 800, 900
            };

            // Run Model

            Map<String, Object> outputs = model.runOn(input);
            assertNotNull(outputs);

            // Check Output

            float[] output = (float[]) outputs.get("output");
            assertNotNull(output);

            float[] expectedOutput = {
                    2,   3,   4,   5,   6,   7,   8,   9,   10,
                    12,  22,  32,  42,  52,  62,  72,  82,  92,
                    103, 203, 303, 403, 503, 603, 703, 803, 903
            };

            assertArrayEquals(output, expectedOutput, epsilon);

        } catch (ModelBundle.ModelBundleException | Model.ModelException | IOException e) {
            e.printStackTrace();
            fail();
        }
    }

    // String Tests

    // Pixel Buffer Tests

    // Int32 and Int64 Tests



    // Real Usage Tests

    @Test
    public void testCatsVsDogsPredict() {
        try {
            // Prepare Model

            ModelBundle bundle = bundleForFile("cats-vs-dogs-predict.tiobundle");
            assertNotNull(bundle);

            TensorFlowModel model = (TensorFlowModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            // Prepare Input

            InputStream stream = testContext.getAssets().open("cat.jpg");
            Bitmap bitmap = BitmapFactory.decodeStream(stream);

            // Run Model

            Map<String,Object> output = model.runOn(bitmap);
            assertNotNull(output);

            // Check Output

            float sigmoid = ((float[]) Objects.requireNonNull(output.get("sigmoid")))[0];
            assertTrue(sigmoid < 0.5);

        } catch (ModelBundle.ModelBundleException | Model.ModelException | IOException e) {
            fail();
        }
    }

    // TODO: Train

    @Test
    public void testCatsVsDogsTrain() {
        try {
            // Prepare Model

            ModelBundle bundle = bundleForFile("cats-vs-dogs-train.tiobundle");
            assertNotNull(bundle);

            TensorFlowModel model = (TensorFlowModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            // Prepare Input

            InputStream stream = testContext.getAssets().open("cat.jpg");
            Bitmap bitmap = BitmapFactory.decodeStream(stream);

            float[] labels = {
                    0
            };

            Map<String, Object> input = new HashMap<String, Object>();
            input.put("image", bitmap);
            input.put("labels", labels);

            // Train Model

            float[] losses = new float[4];
            int epochs = 4;

            for (int epoch = 0; epoch < epochs; epoch++) {

                Map<String,Object> output = model.trainOn(input);
                assertNotNull(output);

                float loss = ((float[]) Objects.requireNonNull(output.get("sigmoid_cross_entropy_loss/value")))[0];
                losses[epoch] = loss;
            }

            assertNotEquals(losses[0], losses[1]);
            assertNotEquals(losses[1], losses[2]);
            assertNotEquals(losses[2], losses[3]);

        } catch (ModelBundle.ModelBundleException | Model.ModelException | IOException e) {
            fail();
        }
    }

    // TODO: Batch Tests

    // TODO: Placeholder Tests

}