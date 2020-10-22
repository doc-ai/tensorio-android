/*
 * MainActivity.java
 * TensorIO
 *
 * Created by Philip Dow on 7/8/2020
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

package com.docai.javaexample;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import androidx.appcompat.app.AppCompatActivity;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;

import ai.doc.tensorio.core.modelbundle.ModelBundle;
import ai.doc.tensorio.core.modelbundle.ModelBundle.ModelBundleException;
import ai.doc.tensorio.core.model.Model.ModelException;
import ai.doc.tensorio.tflite.model.TFLiteModel;
import ai.doc.tensorio.core.utilities.ClassificationHelper;

public class MainActivity extends AppCompatActivity {

    private final Handler main = new Handler(Looper.getMainLooper());
    private final String TAG = "MainActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            // Load the Model

            ModelBundle bundle = ModelBundle.bundleWithAsset(getApplicationContext(), "mobilenet_v2_1.4_224.tiobundle");
            TFLiteModel model = (TFLiteModel) bundle.newModel();

            // Load the Test Image

            InputStream stream = getAssets().open("elephant.jpg");
            Bitmap bitmap = BitmapFactory.decodeStream(stream);

            ImageView imageView = findViewById(R.id.imageView);
            imageView.setImageBitmap(bitmap);

            stream.close();

            // Create a Background Thread

            HandlerThread mHandlerThread = new HandlerThread("HandlerThread");
            mHandlerThread.start();
            Handler mHandler = new Handler(mHandlerThread.getLooper());

            // Execute the Model

            mHandler.post(() -> {
                try {
                    Map<String,Object> output = model.runOn(bitmap);
                    Map<String, Float> classification = (Map<String, Float>)output.get("classification");
                    List<Map.Entry<String, Float>> top5 = ClassificationHelper.topN(classification, 5, 0.1f);

                    for (Map.Entry<String, Float> entry : top5) {
                        Log.i(TAG, entry.getKey() + ":" + entry.getValue());
                    }

                    main.post( () -> {
                        TextView textView = findViewById(R.id.textView);
                        textView.setText(formattedResults(top5));
                    });

                } catch (ModelException e) {
                    e.printStackTrace();
                }
            });

        } catch (IOException | ModelBundleException e) {
            e.printStackTrace();
        }
    }

    private String formattedResults(List<Map.Entry<String, Float>> results) {
        StringBuilder b = new StringBuilder();

        for (Map.Entry<String, Float> entry : results) {
            b.append(entry.getKey());
            b.append(": ");
            b.append(entry.getValue());
            b.append("\n");
        }

        b.setLength(b.length() - 1);

        return b.toString();
    }
}
