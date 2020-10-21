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
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.FileSystemException;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import ai.doc.tensorio.core.data.Placeholders;
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

    /** Create a direct native order byte buffer with floats */

    private ByteBuffer byteBufferWithFloats(float[] floats) {
        int size = floats.length * 4; // dims * bytes_per_float

        ByteBuffer buffer = ByteBuffer.allocateDirect(size);
        buffer.order(ByteOrder.nativeOrder());

        for (float f : floats) {
            buffer.putFloat(f);
        }

        return buffer;
    }

    /** Compares the contents of a float byte buffer to floats */

    private void assertByteBufferEqualToFloats(ByteBuffer buffer, float epsilon, float[] floats) {
        for (float f : floats) {
            assertEquals(buffer.getFloat(), f, epsilon);
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


    // Pixel Buffer Tests

    @Test
    public void testPixelBufferIdentityModel() {
        try {
            ModelBundle tioBundle = bundleForFile("1_in_1_out_pixelbuffer_identity_test.tiobundle");
            assertNotNull(tioBundle);

            Model model = tioBundle.newModel();
            assertNotNull(tioBundle);
            model.load();

            // Ensure inputs and outputs return correct count

            assertEquals(1, model.getIO().getInputs().size());
            assertEquals(1, model.getIO().getOutputs().size());

            int width = 224;
            int height = 224;

            Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            Canvas canvas = new Canvas(bmp);
            Paint paint = new Paint();

            paint.setColor(Color.rgb(89, 0, 84));
            canvas.drawRect(0F, 0F, width, height, paint);

            Map<String, Object> output = model.runOn(bmp);
            assertNotNull(output);

            Bitmap outputBitmap = (Bitmap) output.get("output");
            assertNotNull(outputBitmap);

            // Inspect pixel buffer bytes

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    assertEquals((bmp.getPixel(x, y)) & 0xFF, (outputBitmap.getPixel(x, y)) & 0xFF, epsilon = 1);
                    assertEquals((bmp.getPixel(x, y) >> 8) & 0xFF, (outputBitmap.getPixel(x, y) >> 8) & 0xFF, epsilon = 1);
                    assertEquals((bmp.getPixel(x, y) >> 16) & 0xFF, (outputBitmap.getPixel(x, y) >> 16) & 0xFF, epsilon = 1);
                }
            }

        } catch (ModelBundle.ModelBundleException | Model.ModelException | IOException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testPixelBufferNormalizationTransformationModel() {
        try {
            ModelBundle tioBundle = bundleForFile("1_in_1_out_pixelbuffer_normalization_test.tiobundle");
            assertNotNull(tioBundle);

            Model model = tioBundle.newModel();
            assertNotNull(tioBundle);
            model.load();

            // Ensure inputs and outputs return correct count

            assertEquals(1, model.getIO().getInputs().size());
            assertEquals(1, model.getIO().getOutputs().size());

            int width = 224;
            int height = 224;

            Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            Canvas canvas = new Canvas(bmp);
            Paint paint = new Paint();

            paint.setColor(Color.rgb(89, 0, 84));
            canvas.drawRect(0F, 0F, width, height, paint);

            Map<String,Object> output = model.runOn(bmp);
            assertNotNull(output);

            Bitmap outputBitmap = (Bitmap) output.get("output");
            assertNotNull(outputBitmap);

            // Inspect pixel buffer bytes

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    assertEquals((bmp.getPixel(x, y)) & 0xFF, (outputBitmap.getPixel(x, y)) & 0xFF, epsilon = 1);
                    assertEquals((bmp.getPixel(x, y) >> 8) & 0xFF, (outputBitmap.getPixel(x, y) >> 8) & 0xFF, epsilon = 1);
                    assertEquals((bmp.getPixel(x, y) >> 16) & 0xFF, (outputBitmap.getPixel(x, y) >> 16) & 0xFF, epsilon = 1);
                }
            }

        } catch (ModelBundle.ModelBundleException | Model.ModelException | IOException e) {
            e.printStackTrace();
            fail();
        }
    }

    // Int32 and Int64 Tests

    @Test
    public void testInt32Model() {
        try {
            // Prepare Model

            ModelBundle tioBundle = bundleForFile("int32io_test.tiobundle");
            assertNotNull(tioBundle);

            Model model = tioBundle.newModel();
            assertNotNull(tioBundle);
            model.load();

            // Prepare Inputs

            int[] input = {2};

            // Run Model

            Map<String, Object> outputs = model.runOn(input);
            assertNotNull(outputs);

            // Check Output

            int[] output = (int[]) outputs.get("output");
            assertNotNull(output);

            int[] expectedOutput = {
                    25
            };

            assertArrayEquals(output, expectedOutput);

        } catch (ModelBundle.ModelBundleException | Model.ModelException | IOException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testInt64Model() {
        fail();
    }

    // String Tests

    @Test
    public void test1In1OutStringModel() {
        try {
            // Prepare Model

            ModelBundle tioBundle = bundleForFile("1_in_1_out_string_test.tiobundle");
            assertNotNull(tioBundle);

            Model model = tioBundle.newModel();
            assertNotNull(tioBundle);
            model.load();

            // Prepare Inputs

            ByteBuffer input = byteBufferWithFloats(new float[]{2});

            // Run Model

            Map<String, Object> outputs = model.runOn(input);
            assertNotNull(outputs);

            // Check Output

            FloatBuffer output = (FloatBuffer) outputs.get("output");
            assertNotNull(output);

            assertEquals(25, output.get(), epsilon);

        } catch (ModelBundle.ModelBundleException | Model.ModelException | IOException e) {
            e.printStackTrace();
            fail();
        }
    }

    // Placeholder Tests

    @Test
    public void testPlaceholders() {
        try {
            // Prepare Model

            ModelBundle tioBundle = bundleForFile("1_in_1_placeholder_2_out_vectors_test.tiobundle");
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

            // Prepare Placeholders

            Placeholders placeholders = new Placeholders();
            placeholders.put("input2", input2);

            // Run Model

            Map<String, Object> outputs = model.runOn(input, placeholders);
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

    // Additional Tests

    @Test
    public void testModelWithoutSpecifiedBackendUsesAvailableBackend() {
    // Uses a copy of the 1_in_1_out_number_test without a model.backend field
        try {
            // Prepare Model

            ModelBundle bundle = bundleForFile("no-backend.tiobundle");
            assertNotNull(bundle);

            TensorFlowModel model = (TensorFlowModel) bundle.newModel();
            assertNotNull(model);
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
            fail();
        }
    }

    // Real Usage Tests

    @Test
    public void testCatsVsDogsPredictCat() {
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

    @Test
    public void testCatsVsDogsPredictDog() {
        try {
            // Prepare Model

            ModelBundle bundle = bundleForFile("cats-vs-dogs-predict.tiobundle");
            assertNotNull(bundle);

            TensorFlowModel model = (TensorFlowModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            // Prepare Input

            InputStream stream = testContext.getAssets().open("dog.jpg");
            Bitmap bitmap = BitmapFactory.decodeStream(stream);

            // Run Model

            Map<String,Object> output = model.runOn(bitmap);
            assertNotNull(output);

            // Check Output

            float sigmoid = ((float[]) Objects.requireNonNull(output.get("sigmoid")))[0];
            assertTrue(sigmoid > 0.5);

        } catch (ModelBundle.ModelBundleException | Model.ModelException | IOException e) {
            fail();
        }
    }

}