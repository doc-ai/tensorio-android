package ai.doc.example;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;

import java.io.IOException;
import java.io.InputStream;
import java.util.AbstractMap;
import java.util.Arrays;
import java.util.Map;
import java.util.PriorityQueue;

import ai.doc.tensorio.TIOLayerInterface.TIOVectorLayerDescription;
import ai.doc.tensorio.TIOModel.TIOModel;
import ai.doc.tensorio.TIOModel.TIOModelBundle;
import ai.doc.tensorio.TIOModel.TIOModelBundleException;
import ai.doc.tensorio.TIOModel.TIOModelBundleManager;
import ai.doc.tensorio.TIOModel.TIOModelException;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            TIOModelBundleManager manager = new TIOModelBundleManager(getApplicationContext(), "");

            // load the model
            TIOModelBundle bundle = manager.bundleWithId("mobilenet-v2-100-224-unquantized");
            TIOModel model = bundle.newModel();
            model.load();

            // Load the image
            InputStream bitmap = getAssets().open("picture2.jpg");
            Bitmap bMap = BitmapFactory.decodeStream(bitmap);
            final Bitmap scaled = Bitmap.createScaledBitmap(bMap, 224, 224, false);

            // Create a background thread
            HandlerThread mHandlerThread = new HandlerThread("HandlerThread");
            mHandlerThread.start();
            Handler mHandler = new Handler(mHandlerThread.getLooper());


            mHandler.post(() -> {
                // Run the model on the input
                float[] result = new float[0];

                try {
                    result = (float[]) model.runOn(scaled);
                } catch (TIOModelException e) {
                    e.printStackTrace();
                }

                Log.i("result", Arrays.toString(result));

                // Build a PriorityQueue of the predictions
                PriorityQueue<Map.Entry<Integer, Float>> pq = new PriorityQueue<>(10, (o1, o2) -> (o2.getValue()).compareTo(o1.getValue()));
                for (int i = 0; i < 1001; i++) {
                    pq.add(new AbstractMap.SimpleEntry<>(i, result[i]));
                }

                // Show the 10 most likely predictions
                String[] labels = ((TIOVectorLayerDescription) model.descriptionOfOutputAtIndex(0)).getLabels();
                for (int i = 0; i < 10; i++) {
                    Map.Entry<Integer, Float> e = pq.poll();
                    Log.i(labels[e.getKey()], "" + e.getValue());
                }
            });


        } catch (IOException e) {
            e.printStackTrace();
        } catch (TIOModelBundleException e) {
            e.printStackTrace();
        } catch (TIOModelException e) {
            e.printStackTrace();
        }
    }
}
