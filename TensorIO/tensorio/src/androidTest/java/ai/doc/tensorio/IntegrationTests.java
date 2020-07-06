/*
 * IntegrationTests.java
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

package ai.doc.tensorio;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.support.test.InstrumentationRegistry;

import com.github.fge.jsonschema.core.exceptions.ProcessingException;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.doc.tensorio.TIOModel.TIOModelBundle;
import ai.doc.tensorio.TIOModel.TIOModelBundleException;
import ai.doc.tensorio.TIOModel.TIOModelBundleValidator;
import ai.doc.tensorio.TIOModel.TIOModelException;
import ai.doc.tensorio.TIOTensorflowLiteModel.TIOTFLiteModel;
import ai.doc.tensorio.TIOUtilities.TIOClassificationHelper;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class IntegrationTests {
    private Context context = InstrumentationRegistry.getTargetContext();
    private float epsilon = 0.01f;

    @Before
    public void setUp() throws Exception {

    }

    @After
    public void tearDown() throws Exception {

    }

    @Test
    public void test1In1OutNumberModel() {
        try {
            TIOModelBundle bundle = new TIOModelBundle(context, "1_in_1_out_number_test.tfbundle");
            assertNotNull(bundle);

            TIOTFLiteModel model = (TIOTFLiteModel) bundle.newModel();
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
            assertTrue(output instanceof Map);

            result = (float[]) output.get("output");
            assertTrue(result instanceof float[]);

            assertEquals(1, result.length);
            assertEquals(25f, result[0], epsilon);

            // Run the model on a dictionary

            Map<String, float[]> input_dict = new HashMap<>();
            input_dict.put("input", new float[]{2});

            output = model.runOn(input_dict);
            assertTrue(output instanceof Map);

            result = (float[]) output.get("output");
            assertTrue(result instanceof float[]);

            assertEquals(1, result.length);
            assertEquals(25f, result[0], epsilon);

            // try running on input of of the wrong length, should throw IllegalArgumentException

            try {
                input = new float[]{1, 2, 3, 4, 5};
                model.runOn(input);
                fail();
            } catch (IllegalArgumentException e) {

            }

        } catch (TIOModelBundleException | TIOModelException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void test1x1VectorsModel() {
        try {
            TIOModelBundle bundle = new TIOModelBundle(context, "1_in_1_out_vectors_test.tfbundle");
            assertNotNull(bundle);

            TIOTFLiteModel model = (TIOTFLiteModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            // Ensure inputs and outputs return correct count

            assertEquals(1, model.getIO().getInputs().size());
            assertEquals(1, model.getIO().getOutputs().size());

            float[] expected = new float[]{2, 2, 4, 4};
            float[] input = new float[]{1, 2, 3, 4};

            Map<String, Object> output;
            float[] result;

            // Run the model on a vector

            output = model.runOn(input);
            assertTrue(output instanceof Map);

            result = (float[]) output.get("output");
            assertTrue(result instanceof float[]);

            assertEquals(4, result.length);
            assertTrue(Arrays.equals(expected, result));

            // Run the model on a dictionary

            Map<String, float[]> input_dict = new HashMap<>();
            input_dict.put("input", input);

            output = model.runOn(input_dict);
            assertTrue(output instanceof Map);

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

        } catch (TIOModelBundleException | TIOModelException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void test2x2VectorsModel() {
        try {
            TIOModelBundle bundle = new TIOModelBundle(context, "2_in_2_out_vectors_test.tfbundle");
            assertNotNull(bundle);

            TIOTFLiteModel model = (TIOTFLiteModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            // Ensure inputs and outputs return correct count

            assertEquals(2, model.getIO().getInputs().size());
            assertEquals(2, model.getIO().getOutputs().size());

            Map<String, float[]> inputs = new HashMap<>();
            inputs.put("input1", new float[]{1, 2, 3, 4});
            inputs.put("input2", new float[]{10, 20, 30, 40});

            Map<String, Object> output = model.runOn(inputs);
            assertTrue(output instanceof Map);

            assertEquals(2, output.size());
            assertTrue(output.containsKey("output1"));
            assertTrue(output.containsKey("output2"));

            float[] result1 = (float[])output.get("output1");
            float[] result2 = (float[])output.get("output2");

            assertTrue(result1 instanceof float[]);
            assertTrue(result2 instanceof float[]);

            assertEquals(1, result1.length);
            assertEquals(1, result2.length);

            assertEquals(240, result1[0], epsilon);
            assertEquals(64, result2[0], epsilon);

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

        } catch (TIOModelBundleException | TIOModelException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void test2x2MatricesModel() {
        try {
            TIOModelBundle bundle = new TIOModelBundle(context, "2_in_2_out_matrices_test.tfbundle");
            assertNotNull(bundle);

            TIOTFLiteModel model = (TIOTFLiteModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            // Ensure inputs and outputs return correct count

            assertEquals(2, model.getIO().getInputs().size());
            assertEquals(2, model.getIO().getOutputs().size());

            float[] expected1 = new float[]{
                    56f, 72f, 56f, 72f,
                    5600f, 7200f, 5600f, 7200f,
                    560000f, 720000f, 560000f, 720000f,
                    56000000f, 72000000f, 56000000f, 72000000f
            };

            float[] expected2 = new float[]{
                    18f, 18f, 18f, 18f,
                    180f, 180f, 180f, 180f,
                    1800f, 1800f, 1800f, 1800f,
                    18000f, 18000f, 18000f, 18000f
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

            Map<String, float[]> inputs = new HashMap<>();
            inputs.put("input1", input1);
            inputs.put("input2", input2);

            Map<String,Object> output = model.runOn(inputs);
            assertTrue(output instanceof Map);

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

        } catch (TIOModelBundleException | TIOModelException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void test3x3MatricesModel() {
        try {
            TIOModelBundle bundle = new TIOModelBundle(context, "1_in_1_out_tensors_test.tfbundle");
            assertNotNull(bundle);

            TIOTFLiteModel model = (TIOTFLiteModel) bundle.newModel();
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

            Map<String,Object> output = model.runOn(input);
            assertTrue(output instanceof Map);

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

        } catch (TIOModelBundleException | TIOModelException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testPixelBufferIdentityModel() {
        try {
            TIOModelBundle bundle = new TIOModelBundle(context, "1_in_1_out_pixelbuffer_identity_test.tfbundle");
            assertNotNull(bundle);

            TIOTFLiteModel model = (TIOTFLiteModel) bundle.newModel();
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
            assertTrue(output instanceof Map);

            Bitmap outputBitmap = (Bitmap) output.get("image");
            assertTrue(outputBitmap instanceof Bitmap);

            // Inspect pixel buffer bytes

            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    assertEquals((bmp.getPixel(x, y)) & 0xFF, (outputBitmap.getPixel(x, y)) & 0xFF, epsilon = 1);
                    assertEquals((bmp.getPixel(x, y) >> 8) & 0xFF, (outputBitmap.getPixel(x, y) >> 8) & 0xFF, epsilon = 1);
                    assertEquals((bmp.getPixel(x, y) >> 16) & 0xFF, (outputBitmap.getPixel(x, y) >> 16) & 0xFF, epsilon = 1);
                }
            }

            // Try running on input image of wrong size, should throw IllegalArgumentException

            try {
                Bitmap small = Bitmap.createScaledBitmap(bmp, 128, 128, true);
                model.runOn(small);
                fail();
            } catch (IllegalArgumentException e) {
            }

        } catch (TIOModelBundleException | TIOModelException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testPixelBufferNormalizationTransformationModel() {
        try {
            TIOModelBundle bundle = new TIOModelBundle(context, "1_in_1_out_pixelbuffer_normalization_test.tfbundle");
            assertNotNull(bundle);

            TIOTFLiteModel model = (TIOTFLiteModel) bundle.newModel();
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

            Map<String,Object> output = model.runOn(bmp);
            assertTrue(output instanceof Map);

            Bitmap outputBitmap = (Bitmap) output.get("image");
            assertTrue(outputBitmap instanceof Bitmap);

            // Inspect pixel buffer bytes

            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    assertEquals((bmp.getPixel(x, y)) & 0xFF, (outputBitmap.getPixel(x, y)) & 0xFF, epsilon = 1);
                    assertEquals((bmp.getPixel(x, y) >> 8) & 0xFF, (outputBitmap.getPixel(x, y) >> 8) & 0xFF, epsilon = 1);
                    assertEquals((bmp.getPixel(x, y) >> 16) & 0xFF, (outputBitmap.getPixel(x, y) >> 16) & 0xFF, epsilon = 1);
                }
            }

            // Try running on input image of wrong size, should throw IllegalArgumentException

            try {
                Bitmap small = Bitmap.createScaledBitmap(bmp, 128, 128, true);
                model.runOn(small);
                fail();
            } catch (IllegalArgumentException e) {
            }

        } catch (TIOModelBundleException | TIOModelException e) {
            e.printStackTrace();
            fail();
        }
    }

    //region MobileNet tests

    @Test
    public void testMobileNetClassificationModel() {
        try {
            TIOModelBundle bundle = new TIOModelBundle(context, "mobilenet_v2_1.4_224.tiobundle");
            assertNotNull(bundle);

            TIOTFLiteModel model = (TIOTFLiteModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            InputStream stream = context.getAssets().open("example-image.jpg");
            Bitmap bitmap = BitmapFactory.decodeStream(stream);

            // TODO: Vision pipeline for resizing and normalizing bitmaps
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap,224,224,true);

            Map<String,Object> output = model.runOn(resizedBitmap);
            assertTrue(output instanceof Map);

            Map<String, Float> classification = (Map<String, Float>)output.get("classification");
            assertTrue(classification instanceof Map);

            List<Map.Entry<String, Float>> top5 = TIOClassificationHelper.topN(classification, 5);
            Map.Entry<String, Float> top = top5.get(0);
            String label = top.getKey();

            assertEquals("rocking chair", label);

        } catch (TIOModelBundleException | TIOModelException | IOException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testQuantizedMobileNetClassificationModel() {
        try {
            TIOModelBundle bundle = new TIOModelBundle(context, "mobilenet_v1_1.0_224_quant.tiobundle");
            assertNotNull(bundle);

            TIOTFLiteModel model = (TIOTFLiteModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            InputStream stream = context.getAssets().open("example-image.jpg");
            Bitmap bitmap = BitmapFactory.decodeStream(stream);

            // TODO: Vision pipeline for resizing and normalizing bitmaps
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap,224,224,true);

            Map<String,Object> output = model.runOn(resizedBitmap);
            assertTrue(output instanceof Map);

            Map<String, Float> classification = (Map<String, Float>)output.get("classification");
            assertTrue(classification instanceof Map);

            List<Map.Entry<String, Float>> top5 = TIOClassificationHelper.topN(classification, 5);
            Map.Entry<String, Float> top = top5.get(0);
            String label = top.getKey();

            assertEquals("rocking chair", label);

        } catch (TIOModelBundleException | TIOModelException | IOException e) {
            e.printStackTrace();
            fail();
        }
    }

    //endRegion

    //region Validation Tests

    @Test
    public void testTIOModelBundleValidator() {
        Context context = InstrumentationRegistry.getTargetContext();

        try {
            String[] assets = context.getAssets().list("");
            for(String s: assets){
                if (s.endsWith(".tfbundle")){
                    InputStream inputStream = context.getAssets().open(s + "/model.json");
                    int size = inputStream.available();
                    byte[] buffer = new byte[size];
                    inputStream.read(buffer);
                    inputStream.close();
                    String modelJSON = new String(buffer, "UTF-8");
                    assertEquals(true, TIOModelBundleValidator.ValidateTFLite(context, modelJSON));
                }
            }
        } catch(IOException  | ProcessingException ex) {
            ex.printStackTrace();
            fail();
        }
    }

    //endRegion
}
