/*
 * TFLiteModelIntegrationTests.java
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

package ai.doc.tensorio.pytorch.modelbundle;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import androidx.test.platform.app.InstrumentationRegistry;


import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.FileSystemException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.doc.tensorio.core.model.Model.ModelException;
import ai.doc.tensorio.core.modelbundle.ModelBundle;
import ai.doc.tensorio.core.modelbundle.ModelBundle.ModelBundleException;
import ai.doc.tensorio.core.utilities.AndroidAssets;
import ai.doc.tensorio.core.utilities.ClassificationHelper;
import ai.doc.tensorio.pytorch.model.PytorchModel;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class PytorchModelIntegrationTests {
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

    private ModelBundle bundleForFile(String filename) throws IOException, ModelBundleException {
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

    @Test
    public void test1In1OutIntegerModel() {
        try {
            ModelBundle bundle = ModelBundle.bundleWithAsset(testContext, "1_in_1_out_integer_test.tiobundle");
            assertNotNull(bundle);

            PytorchModel model = (PytorchModel)bundle.newModel();
            assertNotNull(model);
            model.load();

            // Ensure inputs and outputs return correct count

            assertEquals(1, model.getIO().getInputs().size());
            assertEquals(1, model.getIO().getOutputs().size());

            Map<String, Object> output;
            int[] result;

            // Run the model on a number

            int[] input = new int[]{2};

            output = model.runOn(input);
            assertNotNull(output);

            result = (int[]) output.get("output");
            assertTrue(result instanceof int[]);

            assertEquals(1, result.length);
            assertEquals(11f, result[0], epsilon);

            // Run the model on a dictionary

            Map<String, Object> input_dict = new HashMap<>();
            input_dict.put("input", new int[]{2});

            output = model.runOn(input_dict);
            assertNotNull(output);

            result = (int[]) output.get("output");
            assertTrue(result instanceof int[]);

            assertEquals(1, result.length);
            assertEquals(11f, result[0], epsilon);

            // try running on input of of the wrong length, should throw IllegalArgumentException

            try {
                input = new int[]{1, 2, 3, 4, 5};
                model.runOn(input);
                fail();
            } catch (IllegalArgumentException e) {
            }

        } catch (ModelBundleException | ModelException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void test1In1OutNumberModel() {
        try {
            ModelBundle bundle = ModelBundle.bundleWithAsset(testContext, "1_in_1_out_number_test.tiobundle");
            assertNotNull(bundle);

            PytorchModel model = (PytorchModel)bundle.newModel();
            assertNotNull(model);
            model.load();

            // Ensure inputs and outputs return correct count

            assertEquals(1, model.getIO().getInputs().size());
            assertEquals(1, model.getIO().getOutputs().size());

            Map<String, Object> output;
            float[] result;

            // Run the model on a number

            float[] input = new float[]{2};

            output = model.runOn(input);
            assertNotNull(output);

            result = (float[]) output.get("output");
            assertTrue(result instanceof float[]);

            assertEquals(1, result.length);
            assertEquals(11f, result[0], epsilon);

            // Run the model on a dictionary

            Map<String, Object> input_dict = new HashMap<>();
            input_dict.put("input", new float[]{2});

            output = model.runOn(input_dict);
            assertNotNull(output);

            result = (float[]) output.get("output");
            assertTrue(result instanceof float[]);

            assertEquals(1, result.length);
            assertEquals(11f, result[0], epsilon);

            // try running on input of of the wrong length, should throw IllegalArgumentException

            try {
                input = new float[]{1, 2, 3, 4, 5};
                model.runOn(input);
                fail();
            } catch (IllegalArgumentException e) {
            }

        } catch (ModelBundleException | ModelException e) {
            e.printStackTrace();
            fail();
        }
    }


    @Test
    public void test1x1VectorsModel() {
        try {
            ModelBundle bundle = ModelBundle.bundleWithAsset(testContext, "1_in_1_out_vectors_test.tiobundle");
            assertNotNull(bundle);

            PytorchModel model = (PytorchModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            // Ensure inputs and outputs return correct count

            assertEquals(1, model.getIO().getInputs().size());
            assertEquals(1, model.getIO().getOutputs().size());

            float[] expected = new float[]{2, 4, 6, 8};
            float[] input = new float[]{1, 2, 3, 4};

            Map<String, Object> output;
            float[] result;

            // Run the model on a vector

            output = model.runOn(input);
            assertNotNull(output);

            result = (float[]) output.get("output");
            assertTrue(result instanceof float[]);

            assertEquals(4, result.length);
            assertTrue(Arrays.equals(expected, result));

            // Run the model on a dictionary

            Map<String, Object> input_dict = new HashMap<>();
            input_dict.put("input", input);

            output = model.runOn(input_dict);
            assertNotNull(output);

            result = (float[]) output.get("output");
            assertTrue(result instanceof float[]);

            assertEquals(4, result.length);
            assertTrue(Arrays.equals(expected, result));

            // try running on input of of the wrong length, should throw IllegalArgumentException

            try {
                input = new float[]{1, 2, 3, 4, 5};
                model.runOn(input);
                fail();
            } catch (IllegalArgumentException e) {
            }

            // try running on input of of the wrong length, should throw IllegalArgumentException

            try {
                input = new float[]{1};
                model.runOn(input);
                fail();
            } catch (IllegalArgumentException e) {
            }

        } catch (ModelBundleException | ModelException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void test2x2VectorsModel() {
        try {
            ModelBundle bundle = ModelBundle.bundleWithAsset(testContext, "2_in_2_out_vectors_test.tiobundle");
            assertNotNull(bundle);

            PytorchModel model = (PytorchModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            // Ensure inputs and outputs return correct count

            assertEquals(2, model.getIO().getInputs().size());
            assertEquals(2, model.getIO().getOutputs().size());

            Map<String, Object> inputs = new HashMap<>();
            inputs.put("input1", new float[]{1, 2, 3, 4});
            inputs.put("input2", new float[]{10, 20, 30, 40});

            Map<String, Object> output = model.runOn(inputs);
            assertNotNull(output);

            assertEquals(2, output.size());
            assertTrue(output.containsKey("output1"));
            assertTrue(output.containsKey("output2"));

            float[] result1 = (float[])output.get("output1");
            float[] result2 = (float[])output.get("output2");

            assertTrue(result1 instanceof float[]);
            assertTrue(result2 instanceof float[]);

            assertEquals(1, result1.length);
            assertEquals(1, result2.length);

            assertEquals(110, result1[0], epsilon);
            assertEquals(220, result2[0], epsilon);

            // Try running on input of of the wrong length, should throw IllegalArgumentException

            try {
                inputs = new HashMap<>();
                inputs.put("input1", new float[]{1, 2, 3, 5, 6});
                inputs.put("input2", new float[]{10, 20, 30});
                model.runOn(inputs);
                fail();
            } catch (IllegalArgumentException e) {
            }

            // Try running on input of of the wrong length, should throw IllegalArgumentException

            try {
                inputs = new HashMap<>();
                inputs.put("input1", new float[]{1});
                inputs.put("input2", new float[]{10, 20, 30});
                model.runOn(inputs);
                fail();
            } catch (IllegalArgumentException e) {
            }

        } catch (ModelBundleException | ModelException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void test2x2MatricesModel() {
        try {
            ModelBundle bundle = ModelBundle.bundleWithAsset(testContext, "2_in_2_out_matrices_test.tiobundle");
            assertNotNull(bundle);

            PytorchModel model = (PytorchModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            // Ensure inputs and outputs return correct count

            assertEquals(2, model.getIO().getInputs().size());
            assertEquals(2, model.getIO().getOutputs().size());

            float[] expected1 = new float[]{
                    6f, 8f, 10f, 12f,
                    60f, 80f, 100f, 120f,
                    600f, 800f, 1000f, 1200f,
                    6000f, 8000f, 10000f, 12000f
            };

            float[] expected2 = new float[]{
                    18f, 36f, 54f, 72f,
                    180f, 360f, 540f, 720f,
                    1800f, 3600f, 5400f, 7200f,
                    18000f, 36000f, 54000f, 72000f
            };

            float[] input1 = new float[]{
                    1f, 2f, 3f, 4f,
                    10f, 20f, 30f, 40f,
                    100f, 200f, 300f, 400f,
                    1000f, 2000f, 3000f, 4000f
            };

            float[] input2 = new float[]{
                    5, 6, 7, 8,
                    50, 60, 70, 80,
                    500, 600, 700, 800,
                    5000, 6000, 7000, 8000
            };

            Map<String, Object> inputs = new HashMap<>();
            inputs.put("input1", input1);
            inputs.put("input2", input2);

            Map<String, Object> output = model.runOn(inputs);
            assertNotNull(output);

            assertEquals(2, output.size());
            assertTrue(output.containsKey("output1"));
            assertTrue(output.containsKey("output2"));

            float[] result1 = (float[])output.get("output1");
            float[] result2 = (float[])output.get("output2");

            assertEquals(16, result1.length);
            assertEquals(16, result2.length);

            assertArrayEquals(expected1, result1, epsilon);
            assertArrayEquals(expected2, result2, epsilon);

            // Try running on input of of the wrong length, should throw IllegalArgumentException

            try {
                inputs = new HashMap<>();
                inputs.put("input1", new float[]{5, 6, 7, 8});
                inputs.put("input2", new float[]{5, 6, 7, 8});
                model.runOn(inputs);
                fail();
            } catch (IllegalArgumentException e) {
            }

        } catch (ModelBundleException | ModelException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void test2x1MatricesModel() {
        try {
            ModelBundle bundle = ModelBundle.bundleWithAsset(testContext, "2_in_1_out_matrices_test.tiobundle");
            assertNotNull(bundle);

            PytorchModel model = (PytorchModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            // Ensure inputs and outputs return correct count

            assertEquals(2, model.getIO().getInputs().size());
            assertEquals(1, model.getIO().getOutputs().size());

            float[] expected = new float[]{
                    6f, 8f, 10f, 12f,
                    60f, 80f, 100f, 120f,
                    600f, 800f, 1000f, 1200f,
                    6000f, 8000f, 10000f, 12000f
            };


            float[] input1 = new float[]{
                    1f, 2f, 3f, 4f,
                    10f, 20f, 30f, 40f,
                    100f, 200f, 300f, 400f,
                    1000f, 2000f, 3000f, 4000f
            };

            float[] input2 = new float[]{
                    5, 6, 7, 8,
                    50, 60, 70, 80,
                    500, 600, 700, 800,
                    5000, 6000, 7000, 8000
            };

            Map<String, Object> inputs = new HashMap<>();
            inputs.put("input1", input1);
            inputs.put("input2", input2);

            Map<String, Object> output = model.runOn(inputs);
            assertNotNull(output);

            assertEquals(1, output.size());
            assertTrue(output.containsKey("output"));

            float[] result1 = (float[])output.get("output");

            assertEquals(16, result1.length);

            assertArrayEquals(expected, result1, epsilon);

            // Try running on input of of the wrong length, should throw IllegalArgumentException

            try {
                inputs = new HashMap<>();
                inputs.put("input1", new float[]{5, 6, 7, 8});
                inputs.put("input2", new float[]{5, 6, 7, 8});
                model.runOn(inputs);
                fail();
            } catch (IllegalArgumentException e) {
            }

        } catch (ModelBundleException | ModelException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void test1x2MatricesModel() {
        try {
            ModelBundle bundle = ModelBundle.bundleWithAsset(testContext, "1_in_2_out_matrices_test.tiobundle");
            assertNotNull(bundle);

            PytorchModel model = (PytorchModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            // Ensure inputs and outputs return correct count

            assertEquals(1, model.getIO().getInputs().size());
            assertEquals(2, model.getIO().getOutputs().size());

            float[] expected1 = new float[]{
                    1f, 2f, 3f, 4f,
                    10f, 20f, 30f, 40f,
                    100f, 200f, 300f, 400f,
                    1000f, 2000f, 3000f, 4000f
            };

            float[] expected2 = new float[]{
                    2f, 4f, 6f, 8f,
                    20f, 40f, 60f, 80f,
                    200f, 400f, 600f, 800f,
                    2000f, 4000f, 6000f, 8000f
            };

            float[] input1 = new float[]{
                    1f, 2f, 3f, 4f,
                    10f, 20f, 30f, 40f,
                    100f, 200f, 300f, 400f,
                    1000f, 2000f, 3000f, 4000f
            };



            Map<String, Object> inputs = new HashMap<>();
            inputs.put("input", input1);

            Map<String, Object> output = model.runOn(inputs);
            assertNotNull(output);

            assertEquals(2, output.size());
            assertTrue(output.containsKey("output1"));
            assertTrue(output.containsKey("output2"));

            float[] result1 = (float[])output.get("output1");
            float[] result2 = (float[])output.get("output2");

            assertEquals(16, result1.length);
            assertEquals(16, result2.length);

            assertArrayEquals(expected1, result1, epsilon);
            assertArrayEquals(expected2, result2, epsilon);

            // Try running on input of of the wrong length, should throw IllegalArgumentException

            try {
                inputs = new HashMap<>();
                inputs.put("input1", new float[]{5, 6, 7, 8});
                inputs.put("input2", new float[]{5, 6, 7, 8});
                model.runOn(inputs);
                fail();
            } catch (IllegalArgumentException e) {
            }

        } catch (ModelBundleException | ModelException e) {
            e.printStackTrace();
            fail();
        }
    }


    @Test
    public void test3x3MatricesModel() {
        try {
            ModelBundle bundle = ModelBundle.bundleWithAsset(testContext, "1_in_1_out_tensors_test.tiobundle");
            assertNotNull(bundle);

            PytorchModel model = (PytorchModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            // Ensure inputs and outputs return correct count

            assertEquals(1, model.getIO().getInputs().size());
            assertEquals(1, model.getIO().getOutputs().size());

            float[] expected = new float[]{
                    2, 3, 4, 5, 6, 7, 8, 9, 10,
                    12, 22, 32, 42, 52, 62, 72, 82, 92,
                    103, 203, 303, 403, 503, 603, 703, 803, 903
            };

            float[] input = new float[]{
                    1, 2, 3, 4, 5, 6, 7, 8, 9,
                    10, 20, 30, 40, 50, 60, 70, 80, 90,
                    100, 200, 300, 400, 500, 600, 700, 800, 900
            };

            Map<String, Object> output = model.runOn(input);
            assertNotNull(output);

            float[] result = (float[]) output.get("output");
            assertTrue(result instanceof float[]);

            assertEquals(27, result.length);
            assertTrue(Arrays.equals(expected, result));

            // Try running on input of of the wrong length, should throw IllegalArgumentException

            try {
                model.runOn(new float[]{5, 6, 7, 8});
                fail();
            } catch (IllegalArgumentException e) {
            }

        } catch (ModelBundleException | ModelException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testPixelBufferIdentityModel() {
        try {
            ModelBundle bundle = ModelBundle.bundleWithAsset(testContext, "1_in_1_out_pixelbuffer_identity_test.tiobundle");
            assertNotNull(bundle);

            PytorchModel model = (PytorchModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            // Ensure inputs and outputs return correct count

            assertEquals(1, model.getIO().getInputs().size());
            assertEquals(1, model.getIO().getOutputs().size());

            int width = 224;
            int height = 224;

            Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            Canvas canvas = new Canvas(bmp);
            Paint paint = new Paint();
            //paint.setColor(Color.RED);
            paint.setColor(Color.rgb(89, 0, 84));
            canvas.drawRect(0F, 0F, width, height, paint);

            Map<String, Object> output = model.runOn(bmp);
            assertNotNull(output);

            Bitmap outputBitmap = (Bitmap) output.get("image");
            assertTrue(outputBitmap instanceof Bitmap);

            // Inspect pixel buffer bytes

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    assertEquals((bmp.getPixel(x, y)) & 0xFF, (outputBitmap.getPixel(x, y)) & 0xFF, epsilon = 1);
                    assertEquals((bmp.getPixel(x, y) >> 8) & 0xFF, (outputBitmap.getPixel(x, y) >> 8) & 0xFF, epsilon = 1);
                    assertEquals((bmp.getPixel(x, y) >> 16) & 0xFF, (outputBitmap.getPixel(x, y) >> 16) & 0xFF, epsilon = 1);
                }
            }

        } catch (ModelBundleException | ModelException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testPixelBufferNormalizationTransformationModel() {
        try {
            ModelBundle bundle = ModelBundle.bundleWithAsset(testContext, "1_in_1_out_pixelbuffer_normalization_test.tiobundle");
            assertNotNull(bundle);

            PytorchModel model = (PytorchModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            // Ensure inputs and outputs return correct count

            assertEquals(1, model.getIO().getInputs().size());
            assertEquals(1, model.getIO().getOutputs().size());

            int width = 224;
            int height = 224;

            Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            Canvas canvas = new Canvas(bmp);
            Paint paint = new Paint();
            //paint.setColor(Color.RED);
            paint.setColor(Color.rgb(89, 0, 84));
            canvas.drawRect(0F, 0F, width, height, paint);

            Map<String, Object> output = model.runOn(bmp);
            assertNotNull(output);

            Bitmap outputBitmap = (Bitmap) output.get("image");
            assertTrue(outputBitmap instanceof Bitmap);

            // Inspect pixel buffer bytes

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    assertEquals((bmp.getPixel(x, y)) & 0xFF, (outputBitmap.getPixel(x, y)) & 0xFF, epsilon = 1);
                    assertEquals((bmp.getPixel(x, y) >> 8) & 0xFF, (outputBitmap.getPixel(x, y) >> 8) & 0xFF, epsilon = 1);
                    assertEquals((bmp.getPixel(x, y) >> 16) & 0xFF, (outputBitmap.getPixel(x, y) >> 16) & 0xFF, epsilon = 1);
                }
            }

        } catch (ModelBundleException | ModelException e) {
            e.printStackTrace();
            fail();
        }
    }

    //region MobileNet tests

    @Test
    public void testMobileNetClassificationModel_asset() {
        try {
            ModelBundle bundle = ModelBundle.bundleWithAsset(testContext, "mobilenet_v2_1.4_224.tiobundle");
            assertNotNull(bundle);

            PytorchModel model = (PytorchModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            InputStream stream = testContext.getAssets().open("example-image.jpg");
            Bitmap bitmap = BitmapFactory.decodeStream(stream);

            Map<String, Object> output = model.runOn(bitmap);
            assertNotNull(output);

            Map<String, Float> classification = (Map<String, Float>)output.get("classification");
            assertTrue(classification instanceof Map);

            List<Map.Entry<String, Float>> top5 = ClassificationHelper.topN(classification, 5);
            Map.Entry<String, Float> top = top5.get(0);
            String label = top.getKey();

            assertEquals("rocking chair", label);

        } catch (ModelBundleException | ModelException | IOException e) {
            e.printStackTrace();
            fail();
        }
    }
/*
    @Test
    public void testQuantizedMobileNetClassificationModel_asset() {
        try {
            ModelBundle bundle = new ModelBundle(testContext, "mobilenet_v1_1.0_224_quant.tiobundle");
            assertNotNull(bundle);

            TFLiteModel model = (TFLiteModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            InputStream stream = testContext.getAssets().open("example-image.jpg");
            Bitmap bitmap = BitmapFactory.decodeStream(stream);

            Map<String,Object> output = model.runOn(bitmap);
            assertNotNull(output);

            Map<String, Float> classification = (Map<String, Float>)output.get("classification");
            assertTrue(classification instanceof Map);

            List<Map.Entry<String, Float>> top5 = ClassificationHelper.topN(classification, 5);
            Map.Entry<String, Float> top = top5.get(0);
            String label = top.getKey();

            assertEquals("rocking chair", label);

        } catch (ModelBundleException | ModelException | IOException e) {
            e.printStackTrace();
            fail();
        }
    }
*/
    //endRegion

    // FILES

    @Test
    public void testMobileNetClassificationModel_file() {
        try {
            ModelBundle bundle = bundleForFile("mobilenet_v2_1.4_224.tiobundle");
            assertNotNull(bundle);

            PytorchModel model = (PytorchModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            InputStream stream = testContext.getAssets().open("example-image.jpg");
            Bitmap bitmap = BitmapFactory.decodeStream(stream);

            Map<String, Object> output = model.runOn(bitmap);
            assertNotNull(output);

            Map<String, Float> classification = (Map<String, Float>)output.get("classification");
            assertTrue(classification instanceof Map);

            List<Map.Entry<String, Float>> top5 = ClassificationHelper.topN(classification, 5);
            Map.Entry<String, Float> top = top5.get(0);
            String label = top.getKey();

            assertEquals("rocking chair", label);

        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (ModelBundleException | ModelException e) {
            e.printStackTrace();
            fail();
        }
    }
/*
    @Test
    public void testQuantizedMobileNetClassificationModel_file() {
        try {
            ModelBundle bundle = bundleForFile("mobilenet_v1_1.0_224_quant.tiobundle");
            assertNotNull(bundle);

            TFLiteModel model = (TFLiteModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            InputStream stream = testContext.getAssets().open("example-image.jpg");
            Bitmap bitmap = BitmapFactory.decodeStream(stream);

            Map<String,Object> output = model.runOn(bitmap);
            assertNotNull(output);

            Map<String, Float> classification = (Map<String, Float>)output.get("classification");
            assertTrue(classification instanceof Map);

            List<Map.Entry<String, Float>> top5 = ClassificationHelper.topN(classification, 5);
            Map.Entry<String, Float> top = top5.get(0);
            String label = top.getKey();

            assertEquals("rocking chair", label);

        } catch (IOException e) {
            e.printStackTrace();
            fail();
        } catch (ModelBundleException | ModelException e) {
            e.printStackTrace();
            fail();
        }
    }*/
}
