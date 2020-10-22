/*
 * TensorFlowModelTrainingTest.java
 * TensorIO
 *
 * Created by Philip Dow on 10/21/2020
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

import org.hamcrest.core.IsEqual;
import org.hamcrest.core.IsNot;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.FileSystemException;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import ai.doc.tensorio.core.data.Batch;
import ai.doc.tensorio.core.data.Placeholders;
import ai.doc.tensorio.core.model.Model;
import ai.doc.tensorio.core.modelbundle.ModelBundle;
import ai.doc.tensorio.core.utilities.AndroidAssets;
import androidx.test.platform.app.InstrumentationRegistry;

import static org.junit.Assert.*;

public class TensorFlowModelTrainingTest {

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

            // the loss values are vanishingly small with the lisi model and normal looking with the default
            assertNotEquals(losses[0], losses[1]);
            assertNotEquals(losses[1], losses[2]);
            assertNotEquals(losses[2], losses[3]);

        } catch (ModelBundle.ModelBundleException | Model.ModelException | IOException e) {
            fail();
        }
    }

    @Test
    public void testCatsVsDogsTrainBatched() {
        try {
            // Prepare Model

            ModelBundle bundle = bundleForFile("cats-vs-dogs-train.tiobundle");
            assertNotNull(bundle);

            TensorFlowModel model = (TensorFlowModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            // Prepare Input

            InputStream stream1 = testContext.getAssets().open("cat.jpg");
            Bitmap bitmap1 = BitmapFactory.decodeStream(stream1);

            float[] labels1 = {
                    0
            };

            Batch.Item input1 = new Batch.Item();
            input1.put("image", bitmap1);
            input1.put("labels", labels1);

            InputStream stream2 = testContext.getAssets().open("dog.jpg");
            Bitmap bitmap2 = BitmapFactory.decodeStream(stream2);

            float[] labels2 = {
                    1
            };

            Batch.Item input2 = new Batch.Item();
            input2.put("image", bitmap2);
            input2.put("labels", labels2);

            String[] keys = {"image", "labels"};
            Batch batch = new Batch(keys);
            batch.add(input1);
            batch.add(input2);

            // Train Model

            float[] losses = new float[4];
            int epochs = 4;

            for (int epoch = 0; epoch < epochs; epoch++) {

                Map<String,Object> output = model.trainOn(batch);
                assertNotNull(output);

                float loss = ((float[]) Objects.requireNonNull(output.get("sigmoid_cross_entropy_loss/value")))[0];
                losses[epoch] = loss;
            }

            // the loss values are vanishingly small with the lisi model and more normal looking with the default
            assertNotEquals(losses[0], losses[1]);
            assertNotEquals(losses[1], losses[2]);
            assertNotEquals(losses[2], losses[3]);

        } catch (ModelBundle.ModelBundleException | Model.ModelException | IOException e) {
            fail();
        }
    }

    @Test
    public void testCatsVsDogsTrainWithPlaceholder() {
        try {
            // Prepare Model

            ModelBundle bundle = bundleForFile("cats-vs-dogs-train-with-placeholder.tiobundle");
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

            // Prepare Placeholders

            Placeholders placeholders1 = new Placeholders();
            placeholders1.put("placeholder_adam_learning_rate", new float[]{0.0001f});

            Placeholders placeholders2 = new Placeholders();
            placeholders2.put("placeholder_adam_learning_rate", new float[]{0.001f});

            // Train Model with First Learning Rate

            float[] losses1 = new float[4];
            int epochs1 = 4;

            for (int epoch = 0; epoch < epochs1; epoch++) {

                Map<String,Object> output = model.trainOn(input, placeholders1);
                assertNotNull(output);

                float loss = ((float[]) Objects.requireNonNull(output.get("sigmoid_cross_entropy_loss/value")))[0];
                losses1[epoch] = loss;
            }

            // Reset the Model

            model.unload();

            // Train Model with Second Learning Rate

            float[] losses2 = new float[4];
            int epochs2 = 4;

            for (int epoch = 0; epoch < epochs2; epoch++) {

                Map<String,Object> output = model.trainOn(input, placeholders2);
                assertNotNull(output);

                float loss = ((float[]) Objects.requireNonNull(output.get("sigmoid_cross_entropy_loss/value")))[0];
                losses2[epoch] = loss;
            }

            // Compare Losses
            // Ensures that setting the placeholder is having some effect
            // Loss values are fairly normal here: [1.20, 1.16, 1.12, 1.08] vs [0.88, 0.59, 0.37, 0.23]

            assertThat(losses1, IsNot.not(IsEqual.equalTo(losses2)));

        } catch (ModelBundle.ModelBundleException | Model.ModelException | IOException e) {
            fail();
        }
    }

    @Test
    public void testCatsVsDogsTrainBatchedWithPlaceholder() {
        try {
            // Prepare Model

            ModelBundle bundle = bundleForFile("cats-vs-dogs-train-with-placeholder.tiobundle");
            assertNotNull(bundle);

            TensorFlowModel model = (TensorFlowModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            // Prepare Input

            InputStream stream1 = testContext.getAssets().open("cat.jpg");
            Bitmap bitmap1 = BitmapFactory.decodeStream(stream1);

            float[] labels1 = {
                    0
            };

            Batch.Item input1 = new Batch.Item();
            input1.put("image", bitmap1);
            input1.put("labels", labels1);

            InputStream stream2 = testContext.getAssets().open("dog.jpg");
            Bitmap bitmap2 = BitmapFactory.decodeStream(stream2);

            float[] labels2 = {
                    1
            };

            Batch.Item input2 = new Batch.Item();
            input2.put("image", bitmap2);
            input2.put("labels", labels2);

            String[] keys = {"image", "labels"};
            Batch batch = new Batch(keys);
            batch.add(input1);
            batch.add(input2);

            // Prepare Placeholders

            Placeholders placeholders1 = new Placeholders();
            placeholders1.put("placeholder_adam_learning_rate", new float[]{0.0001f});

            Placeholders placeholders2 = new Placeholders();
            placeholders2.put("placeholder_adam_learning_rate", new float[]{0.001f});

            // Train Model

            // Train Model with Second Learning Rate

            float[] losses1 = new float[4];
            int epochs1 = 4;

            for (int epoch = 0; epoch < epochs1; epoch++) {

                Map<String,Object> output = model.trainOn(batch, placeholders1);
                assertNotNull(output);

                float loss = ((float[]) Objects.requireNonNull(output.get("sigmoid_cross_entropy_loss/value")))[0];
                losses1[epoch] = loss;
            }

            // Reset the Model

            model.unload();

            // Train Model with Second Learning Rate

            float[] losses2 = new float[4];
            int epochs2 = 4;

            for (int epoch = 0; epoch < epochs2; epoch++) {

                Map<String,Object> output = model.trainOn(batch, placeholders2);
                assertNotNull(output);

                float loss = ((float[]) Objects.requireNonNull(output.get("sigmoid_cross_entropy_loss/value")))[0];
                losses2[epoch] = loss;
            }

            // Compare Losses
            // Ensures that setting the placeholder is having some effect
            // Loss values here are vanishingly small: [-1.25E8, -1.35E8, -1.45E8, -1.57E8] vs [-1.87 E8, -1.27E8, -3.84E8, -5.01E8]
            // Something about the dog image, losses are vanishingly small training with the dog image alone

            assertThat(losses1, IsNot.not(IsEqual.equalTo(losses2)));

        } catch (ModelBundle.ModelBundleException | Model.ModelException | IOException e) {
            fail();
        }
    }

}
