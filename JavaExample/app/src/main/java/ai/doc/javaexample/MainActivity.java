package ai.doc.javaexample;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.doc.tensorio.TIOModel.TIOModel;
import ai.doc.tensorio.TIOModel.TIOModelBundle;
import ai.doc.tensorio.TIOModel.TIOModelBundleException;
import ai.doc.tensorio.TIOModel.TIOModelBundleManager;
import ai.doc.tensorio.TIOModel.TIOModelException;
import ai.doc.tensorio.TIOUtilities.TIOClassificationHelper;

public class MainActivity extends AppCompatActivity {

    private String TAG = "MainActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            TIOModelBundleManager manager = new TIOModelBundleManager(getApplicationContext(), "");
            Set<String> ids = manager.getBundleIds();
            System.out.println(ids);

            // load the model
            TIOModelBundle bundle = manager.bundleWithId("mobilenet-v2-100-224-unquantized");
            TIOModel model = bundle.newModel();

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
                    Map<String,Object> output = model.runOn(scaled);
                    Map<String, Float> classification = (Map<String, Float>)output.get("classification");
                    List<Map.Entry<String, Float>> top5 = TIOClassificationHelper.topN(classification, 5);

                    for (Map.Entry<String, Float> entry : top5) {
                        Log.i(TAG, entry.getKey() + ":" + entry.getValue());
                    }
                } catch (TIOModelException e) {
                    e.printStackTrace();
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        } catch (TIOModelBundleException e) {
            e.printStackTrace();
        }
    }
}
