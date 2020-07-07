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

import ai.doc.tensorio.TIOModel.TIOModelBundle;
import ai.doc.tensorio.TIOModel.TIOModelBundleException;
import ai.doc.tensorio.TIOModel.TIOModelException;
import ai.doc.tensorio.TIOTFLiteModel.TIOTFLiteModel;
import ai.doc.tensorio.TIOUtilities.TIOClassificationHelper;

public class MainActivity extends AppCompatActivity {

    private String TAG = "MainActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            // Load the Model

            TIOModelBundle bundle = new TIOModelBundle(getApplicationContext(), "mobilenet_v2_1.4_224.tiobundle");
            TIOTFLiteModel model = (TIOTFLiteModel) bundle.newModel();

            // Load the Test Image

            InputStream bitmap = getAssets().open("picture2.jpg");
            Bitmap bMap = BitmapFactory.decodeStream(bitmap);
            final Bitmap scaled = Bitmap.createScaledBitmap(bMap, 224, 224, false);

            // Create a Background Thread

            HandlerThread mHandlerThread = new HandlerThread("HandlerThread");
            mHandlerThread.start();
            Handler mHandler = new Handler(mHandlerThread.getLooper());

            // Execute the Model

            mHandler.post(() -> {
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

        } catch (IOException | TIOModelBundleException e) {
            e.printStackTrace();
        }
    }
}
