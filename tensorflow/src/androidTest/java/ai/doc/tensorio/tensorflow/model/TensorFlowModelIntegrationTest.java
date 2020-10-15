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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.FileSystemException;
import java.util.Map;

import ai.doc.tensorflow.DataType;
import ai.doc.tensorflow.SavedModelBundle;
import ai.doc.tensorflow.Tensor;

import ai.doc.tensorio.core.model.Model;
import ai.doc.tensorio.core.modelbundle.FileModelBundle;
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

    /** Create a direct native order byte buffer with floats **/

    private ByteBuffer byteBufferWithFloats(float[] floats) {
        int size = floats.length * 4; // dims x bytes for dtype

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

            Map<String, Object> outputs = model.runOn(input);
            assertNotNull(outputs);

            // Read Output

            float[] output = (float[]) outputs.get("output");
            assertEquals(output[0], 25, epsilon);

        } catch (ModelBundle.ModelBundleException | Model.ModelException | IOException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void test1x1NumberModelDirectly() {
        try {

            // Prepare Model

            ModelBundle tioBundle = bundleForFile("1_in_1_out_number_test.tiobundle");
            assertNotNull(tioBundle);

            // Direct TensorFlow Test TODO: Make Tensor/IO Test

            File modelDir = ((FileModelBundle)tioBundle).getModelFile();
            SavedModelBundle model = new SavedModelBundle(modelDir);
            assertNotNull(model);

            // Prepare Inputs

            Tensor input = new Tensor(DataType.FLOAT32, new int[]{1}, "input");
            ByteBuffer buffer = byteBufferWithFloats(new float[]{2});
            input.setBytes(buffer);

            // Prepare Outputs

            Tensor output = new Tensor(DataType.FLOAT32, new int[]{1}, "output");

            // Run Model

            Tensor[] inputs = {input};
            Tensor[] outputs = {output};

            model.run(inputs, outputs);

            // Read Output

            ByteBuffer out = output.getBytes();
            assertEquals(out.getFloat(), 25, epsilon);

        } catch (ModelBundle.ModelBundleException | IOException e) {
            e.printStackTrace();
            fail();
        }
    }

    // TODO: Support for batch channel in shapes

    @Test
    public void testCatsVsDogsPredict() {
        try {
            // ModelBundle bundle = ModelBundle.bundleWithAsset(testContext, "cats-vs-dogs-predict.tiobundle");
            ModelBundle bundle = bundleForFile("cats-vs-dogs-predict.tiobundle");
            assertNotNull(bundle);

            TensorFlowModel model = (TensorFlowModel) bundle.newModel();
            assertNotNull(model);
            model.load();

            InputStream stream = testContext.getAssets().open("cat.jpg");
            Bitmap bitmap = BitmapFactory.decodeStream(stream);

            Map<String,Object> output = model.runOn(bitmap);
            assertNotNull(output);

        } catch (ModelBundle.ModelBundleException | Model.ModelException | IOException e) {
            fail();
        }
    }

}